"""Track a take's things through one gripper video with SAM 3, in SAM 3's own environment.

Run by keep_masks with the Python of the environment SAM 3 is installed in
(umi config sam3_python) - SAM 3 wants a newer torch than LeRobot allows, so
it is a process of its own, and this file imports nothing of pinkrab's. It
reads a spec (JSON), crops each frame of the video to the dataset's window
and size, and tracks the things, each named in words (SAM 3's concept
prompt: every instance of it). It writes each frame's labels, the thing's
label where SAM 3 found it, to an .npz, and says how far it has got, a JSON
line at a time on stdout.

    python sam_track.py spec.json

spec: {"video": mp4, "window": [x, y, w, h], "size": [height, width],
       "resolution": 672, "new_thing_score": 0.4, "thing_score": 0.3, "out": npz,
       "things": [{"label": 3, "text": "black cube"}, ...]}

Or kept running, the model loaded once, for one request after another, a
JSON line each:

    python sam_track.py --serve

a spec as above tracks that video, its lines said as they come and the last
{"done": true}; {"image": jpg, "texts": [...], "out": npz, "resolution": 672,
"new_thing_score": 0.4, "thing_score": 0.3} looks for the things in one
picture - the Curate page's preview, as the things are named - and says
{"ok": true, "scores": [[...], ...]} with each text's masks in the npz as t0,
t1, ... (instances, height, width). A request that fails says {"error": "..."}.

Asked over ssh, from another machine, the files come and go in the lines
themselves: "video_data" or "image_data" (base64) in place of a path, and
with no "out" the npz comes back as "data" (base64) in the last line. The
service ends when its asker goes - its stdin closed - even halfway through
a video, so a dropped connection does not leave a GPU busy.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np


def say(**fields) -> None:
    print(json.dumps(fields), flush=True)


def frames_into(folder: Path, video: str, window, size) -> int:
    """The video's frames as the dataset keeps them, as JPEGs in `folder`."""
    x, y, w, h = window
    height, width = size
    cap = cv2.VideoCapture(video)
    n = 0
    while True:
        ok, picture = cap.read()
        if not ok:
            break
        cut = cv2.resize(picture[y:y + h, x:x + w], (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / f"{n:05d}.jpg"), cut, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    return n


# Frames of the picture backbone's answers kept for the next session to ask
# of the same frame: each is about 100 MB at 672.
BACKBONE_KEEP = 6


def reuse_backbone(predictor) -> None:
    """Keep the picture backbone's answers for the last few frames and give
    them again when the same frame is asked of it: the things are followed
    side by side, each in a session of its own, and the backbone - six
    tenths of a frame's time - would otherwise see every frame once for
    every thing. A frame is known by a thin sample of its pixels and what was
    asked of it."""
    import hashlib
    from collections import OrderedDict

    backbone = predictor.model.detector.backbone
    forward = backbone.forward_image
    store: OrderedDict = OrderedDict()

    def cached(samples, *args, **kwargs):
        key = (tuple(samples.shape), args, tuple(sorted(kwargs.items())),
               hashlib.blake2b(samples[..., ::7, ::7].detach().cpu().numpy().tobytes(), digest_size=16).digest())
        if key in store:
            store.move_to_end(key)
        else:
            store[key] = forward(samples, *args, **kwargs)
            if len(store) > BACKBONE_KEEP:
                store.popitem(last=False)
        return store[key]

    backbone.forward_image = cached


def predictor_for(resolution: int):
    """SAM 3's video predictor at `resolution`, its backbone's answers reused."""
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    predictor = Sam3VideoPredictor(resolution=resolution)
    reuse_backbone(predictor)
    # The bar for dropping a thing the detector has not seen for a while,
    # which a backward pass lifts and the next video puts back.
    predictor.unmatched = predictor.model.hotstart_unmatch_thresh
    return predictor


def track(predictor, spec: dict, tell=say) -> None:
    """One video's things tracked as `spec` says, the labels written to its
    out, saying how far it has got; ends by saying done."""
    height, width = spec["size"]
    things = spec["things"]
    if len(things) > 8:
        raise ValueError("at most 8 things")
    with tempfile.TemporaryDirectory() as tmp:
        n = frames_into(Path(tmp), spec["video"], spec["window"], spec["size"])
        if n == 0:
            raise ValueError("the video has no frames")
        # Where each thing was found, a bit each, put together into labels
        # at the end in the things' order - a later thing over an earlier.
        found = np.zeros((n, height, width), dtype=np.uint8)
        seen = np.zeros((len(things), n), dtype=bool)
        model = predictor.model
        model.score_threshold_detection = float(spec.get("thing_score", 0.3))
        model.new_det_thresh = float(spec.get("new_thing_score", 0.4))
        model.hotstart_unmatch_thresh = predictor.unmatched

        def follow(k: int, start: int, direction: str):
            """Thing k named on frame `start` and followed one way, a frame at a time.

            Each session keeps its own copy of the video's frames, in this
            machine's memory rather than the GPU's: a thing each, side by side,
            they are some 1.2 GB a session for a 7 s video at 672, and with the
            model they do not fit an 8 GB GPU. Closed however the following
            ends, so a video that fails leaves nothing on the GPU for the next."""
            session = predictor.handle_request({"type": "start_session", "resource_path": tmp,
                                                "offload_video_to_cpu": True})["session_id"]
            try:
                predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": start,
                                          "text": things[k]["text"]})
                for response in predictor.handle_stream_request(
                        {"type": "propagate_in_video", "session_id": session, "start_frame_index": start,
                         "propagation_direction": direction}):
                    i = response["frame_index"]
                    masks = response["outputs"]["out_binary_masks"]
                    if len(masks):
                        found[i][np.any(np.stack([np.asarray(m, dtype=bool) for m in masks]), axis=0)] |= 1 << k
                        seen[k, i] = True
                    yield i
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session})

        # Every thing named on the first frame and followed forward side by
        # side, so each frame goes through the backbone once - SAM 3 finds a
        # thing that comes into view on the way.
        for frames in zip(*(follow(k, 0, "forward") for k in range(len(things)))):
            tell(stage="forward", frame=frames[0], frames=n)
        # Then each thing first found later than that, named there and
        # followed back to the start. Followed back, it is held by SAM 3's
        # tracker however long its detector stays under the bar - where it
        # went unseen is where it was small or half hidden - and nothing new
        # is taken up on the way.
        model.new_det_thresh, model.hotstart_unmatch_thresh = 1.0, n + 1
        for k in range(len(things)):
            first = int(np.argmax(seen[k])) if seen[k].any() else 0
            if first > 0:
                for i in follow(k, first, "backward"):
                    tell(stage="back", thing=k, frame=i, frames=n)
        labels = np.zeros((n, height, width), dtype=np.uint8)
        for k, thing in enumerate(things):
            labels[(found >> k) & 1 == 1] = thing["label"]
            tell(thing=k, things=len(things), done=True, found=int(seen[k].sum()), frames=n)
        np.savez_compressed(spec["out"], labels=labels)
    tell(done=True)


def look(predictor, ask: dict, tell=say) -> None:
    """One picture looked at as a one-frame video, with the bar the tracking
    starts a thing at: what is found here is what track() would start
    following from this frame."""
    predictor.model.new_det_thresh = float(ask.get("new_thing_score", 0.4))
    predictor.model.score_threshold_detection = float(ask.get("thing_score", 0.3))
    found, scores = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "00000.jpg").write_bytes(Path(ask["image"]).read_bytes())
        for k, text in enumerate(ask["texts"]):
            session = predictor.handle_request({"type": "start_session", "resource_path": tmp})["session_id"]
            try:
                out = predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0,
                                                "text": text})["outputs"]
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session})
            masks = [np.asarray(m, dtype=bool) for m in out["out_binary_masks"]]
            found[f"t{k}"] = np.stack(masks) if masks else np.zeros((0, 1, 1), dtype=bool)
            scores.append([round(float(v), 3) for v in out["out_probs"]])
    np.savez_compressed(ask["out"], **found)
    tell(ok=True, scores=scores)


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        track(predictor_for(int(spec.get("resolution", 672))), spec)
    except ValueError as exc:
        say(error=str(exc))
        return 1
    return 0


def answered(predictor, ask: dict) -> None:
    """One request done: a video tracked or a picture looked at, its files
    taken from the request and its npz sent back in the last line where the
    asker is on another machine."""
    with tempfile.TemporaryDirectory() as tmp:
        for key, name in (("video_data", "video.mp4"), ("image_data", "image.jpg")):
            if key in ask:
                path = Path(tmp) / name
                path.write_bytes(base64.b64decode(ask.pop(key)))
                ask[key.removesuffix("_data")] = str(path)
        inline = "out" not in ask
        if inline:
            ask["out"] = str(Path(tmp) / "out.npz")
        last: dict = {}

        def hold(**fields) -> None:
            # The closing line waits for the npz to go with it.
            nonlocal last
            if inline and (fields.get("ok") or (fields.get("done") and "thing" not in fields)):
                last = fields
            else:
                say(**fields)
        (track if "video" in ask else look)(predictor, ask, hold)
        if last:
            say(**last, data=base64.b64encode(Path(ask["out"]).read_bytes()).decode())


def serve() -> int:
    """Requests one JSON line at a time off stdin, the model loaded once for
    all of them: a video to track (a spec, as main() reads) or a picture to
    look at. stdin is read on a thread of its own, so its end - the asker
    gone - ends the process at once, not after the request in hand."""
    lines: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            lines.put(line)
        os._exit(0)
    threading.Thread(target=read, daemon=True).start()
    predictor = None
    while True:
        line = lines.get()
        try:
            ask = json.loads(line)
            if predictor is None:
                predictor = predictor_for(int(ask.get("resolution", 672)))
            answered(predictor, ask)
        except Exception as exc:  # noqa: BLE001 - one bad request is an answer, not the end of the service
            say(error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(serve() if sys.argv[1:] == ["--serve"] else main())
