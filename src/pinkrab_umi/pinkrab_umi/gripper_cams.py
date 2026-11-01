"""The two gripper cameras, read for the page and the recorder.

Either set: the handhelds' (`handheld`), recorded from, or the follower
arms' (`follower`), what a policy is shown. The same model, camera.yaml's
`gripper` profile, named in a file of their own - see `camera_file`.

Each is a UVC camera opened with OpenCV and read on a thread of its own, in
the mode its profile names.
What the page gets is a smaller JPEG of every other frame, cut as a dataset
cuts it (`gripper_view`, about this unit's measured lens axis) so the page
shows what a policy is shown - or the whole frame while the lens is not
measured; what the recorder gets is the frame itself, with the host clock it
arrived on. Every frame is also written to shared memory as it arrives, for
any other process - a policy, for the arms' (frame_share).

Which camera is which hand is decided by where it is plugged in, because two
of one model may report one serial number: `pinkrab-umi cameras` records the
socket of each in the workspace's cameras.yaml, `pinkrab cameras` the arms'
in follower_cameras.yaml. The top camera's socket is in cameras.yaml too
(`pinkrab-umi cameras --top`); it is never a gripper camera, whichever set is
looked for, even when it is the same model. A profile's `usb` ids, when it
names them, keep other cameras out of the list; without them every USB camera
is one. A side whose camera is not found is
reported as absent and retried, not an error - a handheld can be unplugged
while the top camera runs. On macOS the retry sees only the cameras that
were on the bus when the process started (see `_avfoundation_devices`): one
plugged in later needs the live view started again.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2

from pinkrab_core.dataset import gripper_shape, gripper_view, lens_centre, lens_command

from .config import SETTINGS
from .frame_share import FrameWriter

RETRY_S = 3.0
# How long the main run loop turns before the device list is read.
PLUG_SETTLE_S = 0.2
STALE_S = 1.0

# The two sets of gripper cameras, and the command that names each one's sockets.
CAMERA_SETS = {"handheld": "pinkrab-umi cameras", "follower": "pinkrab cameras"}
# The top camera's name in cameras.yaml, and the command that names its socket.
TOP = "top"
TOP_COMMAND = "pinkrab-umi cameras --top"


def camera_file(which: str = "handheld") -> Path:
    """The file naming one set's sockets. Two files, not one: the sets may be
    plugged into the same two sockets, one at a time."""
    from pinkrab_core.config import SETTINGS as CORE

    if which not in CAMERA_SETS:
        raise ValueError(f"no such camera set: {which!r} - one of {', '.join(CAMERA_SETS)}")
    return CORE.cameras_file if which == "handheld" else CORE.follower_cameras_file


def _named(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    """The sockets a camera file names, by name; one it does not name is left out.

    On macOS a value is the front of the camera's unique id; on Linux a
    device path (the udev symlink the pairing tool writes, or a by-path link).
    """
    try:
        import yaml
        data = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - missing or unreadable: nothing named
        return {}
    if not isinstance(data, dict):
        return {}
    key = "mac_id" if sys.platform == "darwin" else "device"
    entries = {name: data.get(name) for name in names}
    return {name: str(entry[key]) for name, entry in entries.items()
            if isinstance(entry, dict) and entry.get(key)}


def mapping(which: str = "handheld") -> dict[str, str]:
    """Which socket is which side, as the set's file says; a side it does
    not name is left out."""
    return _named(camera_file(which), ("left", "right"))


def top_socket() -> str | None:
    """The top camera's socket, as cameras.yaml names it, or None."""
    return _named(camera_file("handheld"), (TOP,)).get(TOP)


def same_socket(a: str, b: str) -> bool:
    """Whether two stored sockets are one camera. macOS stores the front of the
    unique id; Linux a device link - the udev name or a by-path link - and two
    links are one camera when they lead to the same /dev/videoN."""
    if sys.platform == "darwin":
        return a == b
    return Path(a).resolve() == Path(b).resolve()


def _avfoundation_devices(fresh: bool = False) -> list[tuple[int, str]]:
    """(index, unique id) for every video device, in the order OpenCV's
    AVFoundation backend numbers them: video and muxed devices sorted by
    unique id. The order AVFoundation lists them in is not that, and not even
    the same from one process to the next - indexing it opened the other
    hand's camera, or a phone's.

    The list is the process's own copy, and it follows plugs only while the
    main thread's run loop turns: run it for a moment first. A plain process
    never does, so without this the list stayed what it was at the first call -
    a camera plugged in after that was never seen. On another thread there is
    nothing to turn, and the list is the one the main thread last saw - the
    same one OpenCV opens by, so the index still holds.

    `fresh` asks system_profiler instead, a process of its own and so always
    current: for naming sockets from a thread, where the index is not opened."""
    found = None
    if not fresh:
        try:
            import AVFoundation as av  # pyobjc-framework-AVFoundation
            if threading.current_thread() is threading.main_thread():
                from Foundation import NSDate, NSRunLoop
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(PLUG_SETTLE_S))
            devices = [*av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo),
                       *av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed)]
            found = [str(d.uniqueID()) for d in devices]
        except Exception:  # noqa: BLE001 - fall back to the profiler's listing
            found = None
    if found is None:
        try:
            raw = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                                 capture_output=True, text=True, timeout=10.0).stdout
            cameras = json.loads(raw).get("SPCameraDataType", [])
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return []
        found = [str(c.get("spcamera_unique-id", "")) for c in cameras]
    return list(enumerate(sorted(found)))


def _usb_unique_ids(usb: tuple[int, int] | None) -> set[str] | None:
    """The unique ids AVFoundation would give the USB devices on the bus now -
    those with these vendor and product ids, or every one - from the
    IORegistry (None if it cannot be read).

    The camera stack can keep a device the bus no longer has: one whose link
    dropped mid-stream stayed listed, by AVFoundation and system_profiler
    alike, until UVCAssistant was restarted. The registry is the bus itself.
    """
    try:
        done = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True,
                              text=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode or not done.stdout.strip():
        return None
    text = done.stdout
    found = set()
    for block in text.split("+-o ")[1:]:
        fields = dict(re.findall(r'"(idVendor|idProduct|locationID)" = (\d+)', block))
        if not {"idVendor", "idProduct", "locationID"} <= set(fields):
            continue
        ids = (int(fields["idVendor"]), int(fields["idProduct"]))
        if usb is None or ids == usb:
            found.add(f"{int(fields['locationID']):#x}{ids[0]:04x}{ids[1]:04x}")
    return found


def candidates(fresh: bool = False, top: bool = False) -> list[dict]:
    """Every camera of the gripper profile on the bus now, whichever hand it is
    on - less the top camera, named in cameras.yaml, unless `top` asks for it,
    and then every camera of the top profile with them.

    Each entry: `device` (the index to open on macOS, the path on Linux), and
    `id` (what the mapping matches: the unique id's front on macOS, the
    by-path link on Linux), and `port` (a human-readable socket description).
    `fresh` (macOS) lists from a separate process, current from any thread;
    its `device` is then not an index to open.
    """
    found = _modules(fresh, SETTINGS.gripper_camera.usb)
    if top:
        ids = {str(cam["id"]) for cam in found}
        found += [cam for cam in _modules(fresh, SETTINGS.top_camera.usb) if str(cam["id"]) not in ids]
        return found
    above = top_socket()
    return [cam for cam in found if above is None or not same_socket(above, str(cam["id"]))]


# A USB camera's unique id on macOS: its location, then its vendor and product
# id - 0x21410000 32e4 5234 for a module in socket 0x2141. A camera built in,
# or a phone's, has another kind.
_USB_UNIQUE = re.compile(r"^0x[0-9a-f]+[0-9a-f]{8}$")


def _modules(fresh: bool, usb: tuple[int, int] | None) -> list[dict]:
    """Every USB camera on the bus now - of these vendor and product ids, or
    any; see `candidates`."""
    found: list[dict] = []
    if sys.platform == "darwin":
        tail = None if usb is None else f"{usb[0]:04x}{usb[1]:04x}"
        on_bus = _usb_unique_ids(usb)
        for index, unique in _avfoundation_devices(fresh):
            unique = unique.lower()
            if not _USB_UNIQUE.match(unique) or (tail is not None and not unique.endswith(tail)):
                continue
            if on_bus is not None and unique not in on_bus:
                continue
            found.append({"device": index, "id": unique[:6], "port": f"USB location {unique[:6]}"})
        return found
    by_path = Path("/dev/v4l/by-path")
    if by_path.is_dir():
        seen: set[Path] = set()
        for link in sorted(by_path.iterdir()):
            if not link.name.endswith("index0"):
                continue
            target = link.resolve()
            ids = _usb_ids(target)
            if ids is None or (usb is not None and ids != usb):
                continue
            # One camera, two links: the kernel gives a socket a `usb-` link and
            # a `usbv2-` one to the same node. One name per camera, the first.
            if target in seen:
                continue
            seen.add(target)
            found.append({"device": str(link), "id": str(link), "port": link.name})
    return found


def _usb_ids(node: Path) -> tuple[int, int] | None:
    """The USB vendor and product id of the device behind /dev/videoN, read
    from sysfs: the video node's device is the UVC interface, its parent the
    USB device. Not from /dev/v4l/by-id, whose names come from the serial
    number - two of one model may share one, and then they share one link."""
    usb = Path("/sys/class/video4linux") / node.name / "device" / ".."
    try:
        return (int((usb / "idVendor").read_text(), 16), int((usb / "idProduct").read_text(), 16))
    except (OSError, ValueError):
        return None


def find_top() -> tuple[int | str, str] | None:
    """(the device to open, its socket) of the top camera, if it is on the bus
    where cameras.yaml says."""
    wanted = top_socket()
    if wanted is None:
        return None
    for cam in _modules(False, SETTINGS.top_camera.usb):
        if same_socket(wanted, str(cam["id"])):
            return (cam["device"] if sys.platform == "darwin" else wanted), str(cam["id"])
    return None


def find_devices(which: str = "handheld") -> dict[str, int | str]:
    """Which device index (macOS) or path (Linux) each side's camera is on now."""
    found: dict[str, int | str] = {}
    wanted = mapping(which)
    if sys.platform == "darwin":
        for cam in candidates():
            for side, prefix in wanted.items():
                if prefix and str(cam["id"]).startswith(prefix):
                    found[side] = cam["device"]
        return found
    for side, path in wanted.items():
        if path and Path(path).exists():
            found[side] = path
    return found


class GripperCamera(threading.Thread):
    """One side's camera: opened when found, read until it goes away."""

    def __init__(self, side: str, which: str = "handheld") -> None:
        super().__init__(name=f"gripper-{side}", daemon=True)
        self.side = side
        self.which = which
        self.stopping = threading.Event()
        # Set by `retry` (and by `stop`): a wait between attempts ends at once,
        # and an open camera is let go and found again.
        self._wake = threading.Event()
        self._lock = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._stamp: float | None = None
        self._fps = 0.0
        self._mean = 0.0
        self._note = "looking for the camera"
        self._device: int | str | None = None
        self._id: str | None = None
        # This unit's name in lens.yaml, and where its lens axis lands - read
        # each time the camera opens, so a fit made meanwhile is picked up.
        self.lens_name = side if which == "handheld" else f"follower_{side}"
        self._centre: tuple[float, float] | None = None
        # Called with (side, picture, host_ts) for every frame, when set: the
        # recorder's way in, on this thread, so it must not block.
        self.sink = None
        # Every frame, for other processes: a policy reads the arms'.
        self.share = FrameWriter(f"{which}_{side}")

    # --- what the page and the recorder read ----------------------------------
    def jpeg_after(self, seen: int, timeout: float) -> tuple[bytes | None, int]:
        with self._lock:
            if self._seq == seen and not self.stopping.is_set():
                self._lock.wait(timeout)
            return self._jpeg, self._seq

    def retry(self) -> None:
        """Look for the camera and open it again, now - the Refresh on the page.

        On macOS a camera plugged in after the live view started is not among
        the devices this process can see, and this finds the same nothing;
        the note then says to restart the live view.
        """
        self._wake.set()

    def _rest(self) -> None:
        """Wait out RETRY_S between attempts, or less when asked to retry."""
        self._wake.wait(RETRY_S)
        self._wake.clear()

    def stats(self) -> dict:
        with self._lock:
            fresh = self._stamp is not None and time.monotonic() - self._stamp < STALE_S
            return {"running": fresh, "fps": round(self._fps, 1) if fresh else None,
                    "mean": round(self._mean, 1) if fresh else None,
                    "device": self._device if fresh else None, "id": self._id if fresh else None,
                    "note": "" if fresh else self._note,
                    "lens": None if self._centre else
                    f"whole frame - lens not measured: `{lens_command(self.lens_name, 'capture')}`"}

    # --- the thread ------------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        finally:
            self.share.close()

    def _run(self) -> None:
        while not self.stopping.is_set():
            found = candidates()
            wanted = mapping(self.which).get(self.side, "")
            match = next((c for c in found if sys.platform != "darwin" or str(c["id"]).startswith(wanted)), None) \
                if wanted else None
            device = None if match is None else match["device"]
            if sys.platform != "darwin":
                device = find_devices(self.which).get(self.side)
            if device is not None:
                # Said each time it opens, with what this process sees on the
                # bus: the order OpenCV numbers the devices in is this
                # process's, and it has not always matched another user's.
                print(f"  gripper {self.side} ({self.which}): opening device {device} = {match['id'] if match else '?'}  "
                      f"(seen: {', '.join(f'{c['device']}={c['id']}' for c in found)})", flush=True)
                self._id = None if match is None else str(match["id"])
            if not wanted:
                self._say(f"no socket named for this side - run `{CAMERA_SETS[self.which]}`")
                self._rest()
                continue
            if device is None:
                self._say("no camera on this side - plugged in after the live view started? restart it"
                          if sys.platform == "darwin" else
                          "no camera on this side - is it plugged in where it was?")
                self._rest()
                continue
            self._centre = lens_centre(self.lens_name)
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2
            cap = cv2.VideoCapture(device, backend)
            profile = SETTINGS.gripper_camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*profile.fourcc))
            cap.set(cv2.CAP_PROP_FPS, profile.fps)
            if not cap.isOpened():
                self._say(f"the camera at {device} would not open")
                cap.release()
                self._rest()
                continue
            self._device = device
            self._read(cap)
            cap.release()
            if not self.stopping.is_set():
                self._say("opening it again" if self._wake.is_set() else "the camera stopped answering")
            self._rest()

    def _say(self, note: str) -> None:
        with self._lock:
            self._note = note

    def _read(self, cap: cv2.VideoCapture) -> None:
        count = 0
        window_started = time.monotonic()
        window_frames = 0
        misses = 0
        while not self.stopping.is_set() and not self._wake.is_set():
            ok, picture = cap.read()
            now = time.monotonic()
            if not ok:
                misses += 1
                if misses > 30:
                    return
                continue
            misses = 0
            count += 1
            window_frames += 1
            sink = self.sink
            if sink is not None:
                sink(self.side, picture, time.time())
            self.share.write(picture, now)
            if now - window_started >= 1.0:
                fps = window_frames / (now - window_started)
                window_started, window_frames = now, 0
            else:
                fps = None
            jpeg = None
            if count % SETTINGS.gripper_stream_every == 0:
                h, w = picture.shape[:2]
                if self._centre is not None:
                    small = gripper_view(picture, gripper_shape(w, h), self._centre)
                else:
                    small = cv2.resize(picture, (SETTINGS.gripper_stream_width,
                                                 int(h * SETTINGS.gripper_stream_width / w)),
                                       interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg = buf.tobytes() if ok2 else None
            with self._lock:
                self._stamp = now
                if fps is not None:
                    self._fps = fps
                    self._mean = float(picture[::8, ::8].mean())
                if jpeg is not None:
                    self._jpeg, self._seq = jpeg, self._seq + 1
                    self._lock.notify_all()

    def stop(self) -> None:
        self.stopping.set()
        self._wake.set()
        with self._lock:
            self._lock.notify_all()


class GripperCameras:
    """Both sides of one set, started and stopped together."""

    def __init__(self, which: str = "handheld") -> None:
        camera_file(which)   # an unknown set is said here, not on a thread
        self.which = which
        self.sides = {side: GripperCamera(side, which) for side in ("left", "right")}

    def start(self) -> None:
        for cam in self.sides.values():
            cam.start()

    def stop(self) -> None:
        for cam in self.sides.values():
            cam.stop()
        # Waited for: a capture released while the interpreter is already
        # going down takes the process with it (measured, a segfault on exit).
        for cam in self.sides.values():
            cam.join(timeout=3.0)

    def stats(self) -> dict:
        return {side: cam.stats() for side, cam in self.sides.items()}

    def retry(self, side: str) -> None:
        self.sides[side].retry()
