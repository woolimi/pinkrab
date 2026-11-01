# pinkrab_umi

Where the handheld gripper was, from the top camera over the desk.

The UMI gripper collects demonstrations without the robot: you hold it, you do the
task, and what gets recorded is the gripper's pose over time. That pose has to be
measured against the desk rather than against the camera, which is not held to
where it was put - a stand knocked, a camera turned. So it is measured in two
pieces and multiplied:

```
desk_T_hand(t)  =  inv(cam_T_desk(t))  @  cam_T_hand(t)
                   ----------------      -------------
                   the marker dome       the marker solid
                   standing on the desk  on the hand
```

Both halves come off the same frame of the top camera - a global-shutter
AR0234 module with a 120-degree rectilinear lens, the gripper cameras' part.
Each is a marker solid solved as one pose from
every face in view. Nothing is carried from one frame to the next: no tracker,
no map, no drift, and the same frames give the same answer every time. The desk
solid stands on its flat cut on a flat desk, so its z is up and every hand pose
is level with the desk. A frame that does not show the desk solid has no
world pose: a take is placed in the robot's world through the solid
(`desk_solid.py`), so it has to be in the picture with the solids while a take
is made. Teleop follows each hand's motion in the camera's own frame and
needs no solid.

```bash
pinkrab-umi marker --solid desk ./print           # the desk solid - print, cut, glue
pinkrab-umi marker --solid handheld_left ./print  # a hand's solid
pinkrab-umi live                                 # watch it solve; takes are recorded from the page
pinkrab-umi bench <session>                      # re-judge a recorded take
```

## The desk solid

The eighteen-square solid cut flat under its side squares, standing on its
cut: the hands' size and construction - 30 mm faces, 21 mm markers on 28 mm
printed tiles - with ids 32 to 44 on its thirteen squares. From wherever the
head looks at least two are in view, and two faces of a solid are not
coplanar: together they pin the pose outright, where one square alone fits two
poses. The tiles and the solid pocketed for them are in `print/`
(`marker_tiles_desk_*.stl`, `marker_solid_tiles_desk.stl`); `marker --solid desk`
writes the paper sheet and the paper-pocket solid. Standing on its cut, its origin is 15 mm above the desk.

Set it on the desk in front of your chest, where the head sees it past the
hand. It must not move during a take: it is the world.

A take records the desk solid it was made with in its `meta.json` (`desk`:
its face size, its yaw and how far its origin stood above the desk), and the
studio's replay reckons where the solid stood in the robot's world from that
solid, the start of a policy run from that height. Resolve skips a take made
with another size of desk solid than today's.

## The marker solid

A 50 mm solid with a `DICT_4X4_50` marker on five faces and a stalk socket in the
sixth. Five rather than one because a single square is planar: its pose has a
twofold ambiguity that noise flips between, and near face-on the solver picks
between the two on nothing. Two faces of a solid are not coplanar, so together they
pin the pose outright.

| | |
|---|---|
| Solid edge | 50 mm |
| Paper tile | 48 mm, in a 48.4 mm pocket 0.4 mm deep |
| Marker (black square) | 36 mm - 6 mm a module, 6 mm of quiet zone |
| Face ids | +Z 0, +X 1, -X 2, +Y 3, -Y 4 |
| Up | side faces toward +Z, top face toward +Y |
| Stalk socket | 8 x 8 mm square, 15 mm deep, in -Z |

Only the 36 mm goes into `solvePnP`. Everything else is fit.

**Print at 100%.** Not fit-to-page, not shrink-to-fit. A page scaled by three
percent gives markers three percent small, and a marker three percent small gives
a solid three percent further away - on every frame, smoothly, in a trajectory that
looks entirely correct. The sheet carries a 100 mm ruler bar; measure it before
cutting anything.

**Matte paper.** The camera sees a glossy print as a mirror of whatever lamp
is behind you, and a blown-out highlight across a marker is a face that stops
decoding at exactly the angles you were about to use.

**The bar at the top of each tile is which way up.** PnP returns the marker's own
frame, so a tile glued a quarter turn round yields a pose wrong by a right angle
whenever that face is the one being seen - and the residual is perfect, because
the fit is perfect. It is the geometry behind the fit that is wrong. Check the
bars before the glue dries.

The socket is square so the handle cannot turn in it. A solid that has rotated on
its stalk since the last take reports a pose wrong by however far it turned, with
nothing in the frame to say so.

`marker` writes a vector PDF (exact by construction - verified to 0.001% of the
36 mm at 600 dpi), a 600 dpi PNG of the same page, and a watertight STL.

### Two more solids, for the hands

The handheld carries a solid of its own on a peg out of its wrist, which runs
the roof's outline 15 mm further down (`wrist_mm`) and so leaves room under the
camera's back for its cable: the whole eighteen-square solid with 30 mm faces,
sixteen of them marked with 21 mm markers on 28 mm printed tiles, the same
socket. It hangs by one blank face and carries the driver board's mount on the
other. There is one per hand, told
apart by id - the left 0 to 15, the right 16 to 31 - so two in the same frame
are never confused. The tiles and the solid pocketed for them are in `print/`
(`marker_tiles_handheld_*.stl`, `marker_solid_tiles_handheld_*.stl`); `marker`
writes the paper sheet and the paper-pocket solid.

    pinkrab-umi marker --solid handheld_left ./print
    pinkrab-umi marker --solid handheld_right ./print

A take records in its `meta.json` the hand solids' face sizes (`hands`) and
where each sat on its handheld (`solid_T_tcp`, the tool centre in the solid's
frame), and convert and the studio's replay find the tool centre with those; a
take from before it kept `solid_T_tcp` hung its solids against the roof on the
bare peg. Convert and resolve skip a take that does not record the sizes; the
shelf's replay places no hand for it - a fail for a take set at the chest,
unjudged for one from before.

Every solid is a `SolidSpec` in `config.py`, and the solver is handed the same
spec the sheet was printed from.

## The gripper cameras

Two pairs of one camera model: the handhelds', recorded from, and the follower
arms', shown to a policy. Every module reports the same serial number, so a
camera is named by the USB socket it is plugged into, one plug at a time:

```bash
pinkrab-umi cameras                  # the handhelds' - cameras.yaml
pinkrab-umi cameras --top            # the top camera, the same part - cameras.yaml too
pinkrab cameras                      # the follower arms' - follower_cameras.yaml
pinkrab-umi live --cameras follower  # the arms' pair in the live view (the policy page asks for it)
```

The top camera is one more module of the part, so its socket is named the
same way - unplug it, plug it back - and every naming of the hands' cameras
leaves it out: it stays plugged in while they are named.

The two files may name the same sockets: the rig has two camera ports, the
handhelds' cameras plugged in to record and the arms' to run a policy. A live
view opens one pair and records with it what that pair is for: a demonstration
(`kind: demo` in the episode's `meta.json`) with the handhelds', and a
rollout (`kind: rollout`, a policy's run from Drive to its stop, the handhelds
putting the arms right where it went wrong - DAgger) with the arms'; asked for
the other kind, it refuses. A rollout's episode also has `robot.jsonl`, the
driven arms' joints and the gate's mode as the robot read them, each reading
marked with who had the arms (`controller`: `policy`, `human` - a correction -,
`hold` while the clutch held them, `none` otherwise), and its `meta.json`
names the arms (`arms`), the policy, the seconds each controller had them
(`controlled`) and the build of each arm those joints are of (`description`,
by arm, a hash of where the description puts the arm's joints). Convert learns
from the `human` stretches only. Convert and the replay take a rollout only on
those builds: on an arm with a link another length or a zero moved, the same
angles are another pose.

Convert makes a dataset of both hands from takes made with both: every frame
has both solids solved, both jaws read and a picture from each gripper, and
each hand's picture, steps, opening and absolute pose go under its own name
(`observation.images.gripper_left`, `left_dx`, `pose.tcp_left`, ...). The
state ends with where the right tool stands in the left tool's frame, which
rollout reads off the two arms the same way. Before it are the openings
alone, never a hand's last step: a policy shown its own last step learns to
repeat it and stops looking at the picture.

Each gripper camera's lens is measured against the ChArUco board in `print/lens_charuco_a4.pdf`
(`pinkrab-umi lens board` writes the same page): print it at 100 %, then
`pinkrab-umi lens capture left` and `pinkrab-umi lens fit left`, and the same
for the right. The arms' cameras are measured by `pinkrab lens capture|fit
left|right` with their own pair plugged in, and kept as `follower_left` and
`follower_right`: a lens is a unit, and each picture is cut about the lens
axis of the camera that took it. Until a camera is measured, the live view
shows it whole, a take from it does not convert, and a policy does not run
on it.

The top camera, whose picture every marker pose is solved in, is measured
against the same board the same way: `pinkrab-umi lens capture top`, opening
it as the live view does (the same mode and exposure), so with the live view
down. Hold the board still in many places, the picture's corners among them,
lifted off the desk and tilted 30-45 deg this way and that, near and far. A
view tilted a new way is kept as well as one landing somewhere new, and the
window shows the tilt. Then `pinkrab-umi lens fit top`: a pinhole with the
rational distortion (k1, k2, p1, p2, k3, k4, k5, k6 - the five of an ordinary
lens fall short this far off the axis), started from the lens's earlier fit
or else its nominal 120 degrees, kept as `top`. A board shown at one tilt only
leaves the focal length unknown, and the fit refuses it. The module carries
no numbers of its own, so the live view solves nothing until the lens is
fitted, and with the fit from its next start after. Each take's `meta.json`
says which fit (`camera_source`), and names the lens it is of
(`camera_unit`): the views kept in `lens/top` name it in their `camera.json`,
and a new lens is a new name - empty `lens/top` and capture again.

A fit made again of the same lens is a better lens than the one the takes
before it were solved with. A take keeps no top-camera video - its pictures
are of the room and the person, never shown to a policy - but each frame's
pose line keeps the corners of every marker read in it (`markers`), so
`pinkrab-umi resolve` (every take through the fitted lens not yet solved with
its fit, or the ones named) makes their poses again from those through the
live view's own solving, with the new fit, and keeps the recorded ones as
`poses.recorded.jsonl`. Solved again with the lens they were recorded
through, the corners give back the recorded poses within 0.06 mm (measured
10-31). A take recorded before the corners were kept has its video instead,
and is solved again from that. A take is only solved again when it was recorded
through the lens the fit is of, at the fit's picture size, and with the
solids as they are now: the
desk solid and its face size, the hand solids looked for and their face sizes
as the take records them (a take that does not record them is skipped). Such a
video is compressed, so a pose made
from it is about a millimetre from the live view's (4 mm at the 95th
percentile); the lens moved a fingertip on the desk by up to 6 mm. The setup
checklist - the studio's Before starting card, `pinkrab doctor` - carries
both steps: the top camera's lens with the line that fits it, and, once it
is fitted, how many takes through that lens are not yet solved with it. A take
resolve finds it cannot solve again with that fit it notes in the take
(`resolve_skipped`, with why and the solids it judged by); the card counts
those apart, and resolve tries them again once the solids it looks for
change.

## The live view

Two things can only be found with the solid in your hand and an answer on the
screen, so `live` puts the marker's picture and the solver's own numbers on a
page at `localhost:8731`.

The first is the glue. A patch turned a quarter of the way round is consistent
with itself, so a frame that sees only that face solves cleanly and lies about
where the solid is - the arrow printed on the sheet is there to stop that, and this
is how to tell whether it worked. Turn the solid until every pair of faces has been
seen together; the wrong one shows up as a reprojection error that jumps whenever
it joins the solve, and the page draws its outline red instead of green when the
frame gets thrown out.

The second is noise against distance. Hold the solid still and the window fills
with samples; read the standard deviation of x, y and z in millimetres and the
spread of the rotations in degrees, hit reset, move back half a metre and do it
again. The maximum rotation deviation is still the one to watch rather than the
mean: with the stream filter in front of the window it should stay in single
digits, and a large one means flips are getting through it.

The third is the take itself. **Start take** on the page opens a session on disk
- the PNG format bench reads - and feeds it until **stop
take**, at which point the closure verdict is on the page: start pose against end
pose, coverage beside it, PASS or FAIL against the thresholds in `config.py`. Seat
the solid on its fixture, start, hold still a moment, do thirty seconds of moving,
reseat it, stop. The picture is live the whole way, so a take that drifted out of
frame is a take you saw drift out of frame, not one bench convicts a minute
later. What the page shows is computed by `bench`'s own functions over the same
solved frames, so `pinkrab-umi bench <name>` on the session it wrote reproduces
the verdict digit for digit.

The browser only displays. The camera is opened here, the frames are solved here
with its lens as fitted against the board (lens.yaml `top`) - nothing is
solved until it is - and the page is sent pictures and numbers - a
page that ran its own detector would be a second answer to the same question,
computed from intrinsics somebody had to type in.

## The filter

A single visible face is a plane, and a plane in one image fits two poses -
mirrored about the image plane, tens of degrees apart, with residuals that
converge as the face comes square to the camera. The solver cannot rank them from
one frame; measured on this rig, a still solid read 85 degree rotation spikes
whenever a face sat near face-on, and a two-face view that shed one face at the
area gate flipped the same way in the very next frame.

So between the solver and everything downstream sits a stream filter. Frames with
two or more faces pass untouched - non-coplanar faces pin the pose outright - and
each one refreshes an anchor. A single-face frame whose two residuals clearly
rank keeps its own answer. One whose residuals do not rank is decided by the
anchor, the accepted pose moments ago, since a solid does not turn eighty degrees
in a thirtieth of a second; and when the anchor is older than a second, the frame
is refused rather than guessed at. Refusals become holes, and holes are visible -
nothing interpolates across them.

The same filter, cold from the first frame, runs in all three places a stream is
judged: the live display, a take's verdict, and `bench` over a session - so the
same frames in the same order give the same answer everywhere. Its thresholds
live in `config.py` beside the reprojection gate.

## The recording

One camera, 1920x1200 MJPG asked for at 60 fps - it sends about 50; asked for 30 it sends about 21 - read with OpenCV. Its exposure is
pinned (`top_exposure_us`, 10 ms, at most one frame's 16.7 ms) - the hand's blur budget, not the room's -
and its gain fitted at bring-up and then following the room. AVFoundation has
no manual exposure on the Mac, so both are set as UVC requests on the
module's control pipe through IOKit (`uvc.py`), beside the stream and without
root; on Linux they are the video node's V4L2 controls. The module counts
exposure in 1/60 ms, not the 100 us UVC names: its brightness stops rising
where the exposure fills the frame period.

Frames land as PNG plus a `frames.jsonl`, and the lens they were solved with
as `calibration.json` - `camera_source` says which fit. PNG rather than a video
container: a container would be a quarter the size and would re-quantise every
marker edge, which is the signal the whole pipeline reads. A lens fitted per
lens rather than read from a table: units differ by a percent or two, and a
percent of focal length is a percent of every distance reported.

Timestamps are the host's. The module stamps nothing of its own, so a frame
is its arrival less the latency the gripper cameras are stamped with
(`top_camera_latency_s`), the same module over the same bus.

## The verdict

Reprojection error says the marker fit the corners. It says nothing about a focal
length a percent out or about a desk solve leaning a degree - both of which
produce beautiful residuals and a wrong trajectory.

So the number that decides whether the rig is usable is closure. Take the gripper
off a fixture, move around for thirty seconds, put it back on the same fixture:
whatever pose difference comes out is the entire error budget, measured end to
end, with no ground truth beyond a lump of plastic that did not move. The live
page reports it the moment a take ends, and `bench` recomputes the same
millimetres and degrees from the session on disk, with coverage beside either - a
two-millimetre closure over a take that solved a third of its frames is not a
good take, it is a short one.

## Offline check

No camera, no solid, no room. The solid is rendered into synthetic frames from a
known pose, put through the same detector and solver and composition the real
thing uses, and compared against the pose it was rendered from.

```bash
python src/pinkrab_umi/test/offline_check.py
```

That closes a loop the residuals cannot. A face transform with a sign the wrong
way round still solves, and still reprojects to a fraction of a pixel; the pose is
simply wrong, by an amount that looks like a hand moving.

## Not here yet

- **Gripper width.** The solid gives the pose. What the jaws were doing is the
  other half of a UMI demonstration and is not measured yet.
