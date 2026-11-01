# pinkrab_sim

The UMI loop in MuJoCo, end to end: a Hermit with no arm behind it makes the
takes, the recorder's own format holds them, `pinkrab-umi convert` turns them into
a dataset, and the arm in the same scene runs what was trained on it through the
rollout's own `Driver`. Every step between a hand moving and an arm moving is the
code the rig runs; what the simulation replaces is the room - the lens, the desk,
the blocks and the person.

```bash
pinkrab-sim lens                                   # where the gripper camera must point, per lens
pinkrab-sim demos --count 100 --batch a            # takes the left arm could have made
pinkrab-sim convert a --out sim-a                  # into a LeRobot dataset
pinkrab-remote launch "this machine" sim-a --policy act --name act-sim-a
pinkrab-sim swap ~/pinkrab/train/act-sim-a/checkpoints/last/pretrained_model
pinkrab-sim evaluate ~/pinkrab/train/act-sim-a/checkpoints/last/pretrained_model --videos
pinkrab-sim oracle                                 # the arm replaying takes: the ceiling
```

Rendering is headless (EGL). Takes land in `~/pinkrab/sim/episodes/`, evaluation
reports in `~/pinkrab/sim/evals/`.

Both hands' takes - the right hand putting its block in its bowl, the left
held ready, the task the rig is proven on:

```bash
pinkrab-sim demos --scene bowls --count 200 --batch b --workers 6
pinkrab-sim convert b --out sim-b                  # both hands, the studio's state
pinkrab-sim evaluate <checkpoint> --scene bowls    # both arms, started where the rig starts them
pinkrab-sim oracle --scene bowls
```

## The scene

`scene.xml` compiled with MjSpec and never written back - the safety gate checks
poses against that file, and blocks on its desk would change its answers. On top
of it:

- **the gripper cameras**, on each arm and on the Hermit, placed from
  `camera.yaml` by the same construction the bracket is cut from
  (`lens.claw_T_camera`). The bracket itself is hidden from them, as it is behind
  the lens in the room;
- **the blocks**: the top two pieces of the stacking pyramid, the white cap and
  the green frustum it goes on, free bodies with their measured size and mass;
- **the Hermit**: a mocap body laid out exactly as the gripper link of the arm
  that follows it, wearing that hand's printed body, jaw and marker solid. Its
  camera and jaw hinge are where the arm's are, which is the premise of the one
  camera spec for both. Its TCP is at the arm's and in the handheld's convention,
  which the arm's tool frame is `TOOL_TURN` of;
- **the desk solid's frame** (not drawn, so no camera sees a stand-in for it)
  and **the top camera**, where every recorded pose is measured from and seen
  from;
- **servos as the bench has them**: scene.xml's gain of 15 N m/rad lets an arm
  reaching over the desk sag 15 mm under its own weight, so the task scene uses
  80 (an STS3215 stalls near 3 N m and gives all of it within a couple of
  degrees), the rotor felt through the 1:345 gearbox as 0.006 kg m^2 of
  armature - without it the 12 g jaw rings faster than the 2 ms step - and the
  gripper's squeeze capped at the quarter of stall torque `Torque_Limit` 250
  leaves. Contact is an elliptic cone with `impratio` 10.

## Both hands

The `bowls` scene puts a 40 mm block and a 100 mm bowl on the side of each
working hand (`two_hand_working`: the right hand, as the rig is proven, or
both) and a Hermit in each hand; a hand not at work is held still where its
arm stands ready, a pose the arm can take, the jaw open. The bowls are a floor and a ring of wall
slabs, plain shapes MuJoCo collides with inside and out. Each hand's take is
planned as one hand's is - the block taken square to a face, held by its
side against the claw, let go just over its bowl - and the two run at once,
the shorter one holding still at its end; a take lets go with the hand
drawing back and rising, as the shelf wants. A take is kept only if each arm
could follow its hand and the two arms, posed frame by frame as they would
follow, never meet; and only if every working hand's block ends in its bowl.

The desk solid stands where the rig reckons one set at the operator's chest
(`desk_solid.world_T_desk`), and the takes say so (`placement: chest`). So the
shelf judges them on the arms as it judges the rig's, convert carries them
into the robot's world the same way, and `evaluate` starts both arms where
the rig would - the policy page's own start, not one of the simulation's.

## The fisheye

MuJoCo renders pinhole cameras, and a 200 degree lens sees 100 degrees off its
axis. Each frame is drawn from five 90 degree pinhole views with one centre - the
front half of a cube map - and every fisheye pixel is looked up in the face its
ray passes through, by a table made once per lens. A lens is Kannala-Brandt,
r = f theta (1 + k1 theta^2 + ... + k4 theta^8), which with every k zero is
equidistant; past the image circle, or past where a measured fit stops
growing, nothing is drawn.
The scene's headlight is turned off, because a light that moves with the camera
would light the five faces five ways.

`pinkrab-sim lens` solves the pitch and yaw that put the shut fingertips on
the image's centre line a quarter of the way up from the bottom, for each lens in
`config.LENSES` and each one the rig has measured into the workspace's lens.yaml
(`pinkrab-umi lens fit`), listed as ar0234_<hand>. For the module `camera.yaml`
was solved against, it gives back camera.yaml's own angles.

## The takes

A take is planned the way a hand makes one: down along the tool's axis to the
cap, shut, up, across, down onto the green piece, open, away - with a hand's
slow wander on top. The cap is held against the claw's inner face, where the
jaw meets its sides at 0.27 to 0.30 of its travel. A plan is kept only if

- the left arm can follow every frame of it with `track()` from the frame
  before - the rollout's own solver and step - with no joint asked to jump;
- the arm's claw, standing where the Hermit stands, stays out of the desk;
- the cap, let go a few millimetres above the green piece, ends up on it.

`--viewpoint opposite` plans the takes a person sitting across the table would
make, facing the robot. No arm on this board can make them, and the first check
rejects every one.

What is written is what `pinkrab-umi live` writes, plus `truth.jsonl`: the TCP as
it was, beside what the tracker saw of it (the tracker's noise is
`pose_noise_m`, `pose_noise_rad` in `config.py`).

## The checks

```bash
python src/pinkrab_sim/test/sim_check.py
```

The lens against camera.yaml, the camera inside its bracket, the Hermit's TCP
where it is held, the arm holding a Hermit's pose with its camera and jaw hinge
where the Hermit's are, and one take through convert against its truth - once with no
tracker noise, where the dataset's `pose.tcp_<hand>` and actions have to be the truth,
and once with it. And one take of both hands: both blocks in their bowls, the shelf
passing both hands on their arms, each hand's `pose.tcp` and the relative pose
in the state the truth, and the rig's start putting each arm where its hand
began.

## Evaluation

`swap` asks a checkpoint about frames of its own dataset three ways - as
recorded, with another frame's picture, and with another frame's state - and
prints the cosine between the directions of the answers and against the
recorded step. A policy that answers the same with another picture is not
looking at it. A policy that reads a command, trained on a
dataset of several, is asked a fifth way, with another of the dataset's
commands: one that answers the same is not listening to what it is told.

`evaluate` lays the things out as a take could have been made on them - a layout
a demonstration was planned for and the arm could follow - puts the arm where the
rig puts it before a policy drives (`demo_start`, or a recorded take's first pose
with `--start episode`), and counts three things per episode: the cap picked,
lifted clear of the desk, on the green piece at the end.

The grasp is judged two ways. By default the cap is welded to the claw once the
jaw has shut on it where the takes held it, and let go when the jaw opens; while
welded it leaves the claw's and the jaw's collisions and keeps the desk's and the
green piece's. What that measures is whether the policy went to the cap, shut,
carried and let go in the right place. A block bound for a bowl is welded once
the claw and the jaw both touch it. `--physics` leaves the grasp to contact:
a pyramid in a jaw that swings about a hinge tips as it is carried, and lands off
the green piece more often than on it.

`oracle` is the ceiling for both: the arm made to follow planned takes exactly,
tool pose by tool pose through `track()`, no policy. Welded, it places nineteen
caps in twenty; by contact alone it lifts nine caps
in ten and places about a third.
A policy's failures are its own only below those.
