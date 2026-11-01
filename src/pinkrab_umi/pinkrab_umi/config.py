"""Single source of truth for the top-camera tracking slice.

Everything the pipeline measures is downstream of two numbers that only exist
once: how big the printed marker is, and what the camera's intrinsics are. Get
either wrong and nothing fails - the trajectory just comes out scaled, which
looks exactly like a correct trajectory of a different gripper. So the printed
geometry lives here and is read by both the sheet generator and the pose solver,
and the intrinsics are read off the device rather than typed.

The workspace root is pinkrab_core's; sessions land under it beside the
calibration and the recorded trajectories, because they describe this rig too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.config import CameraProfile


@dataclass(frozen=True)
class SolidSpec:
    """One printed solid: its size, and the three lengths that follow from it.

    The three are one rule. The patch is the face less a millimetre a side, so
    the print has somewhere to be cut without eating the marker's white border.
    The marker is six eighths of the patch: 4x4_50 is six modules across and the
    detector wants a module of white outside it, so the patch is eight modules
    and the black square six. The marker side is the quantity solvePnP is given,
    and the only length here that changes the answer rather than the fit.
    """

    name: str
    face_mm: float
    patch_mm: float
    marker_mm: float
    # Ids follow the face order from this base, so a second solid in the same
    # room - the other hand's - is told apart by its numbers alone.
    first_marker_id: int = 0
    # Which solid the faces are on: a rhombicuboctahedron - eighteen squares
    # forty-five degrees apart, so at least two are always inside the angle a
    # marker still reads at, its eight triangles carrying nothing - whole, or
    # cut to the dome the handhelds carry. Every face is a square, so the
    # sheet, the pockets and the solver are one code; a shape contributes its
    # list of faces and how far each sits from the centre.
    shape: str = "rhombicuboctahedron_dome"
    # Whether the blank faces get stalk sockets. A solid that hangs from a peg
    # needs them; one that stands on a desk is plainer without.
    sockets: bool = True
    # The letter cut into each of the solid's eight triangles, which carry no
    # marker: which hand the solid belongs on, readable from any side.
    hand_letter: str = ""
    # Whether the solid hangs on the left handheld, the right one's mirror
    # image: every frame on it is the right's reflected (`Settings._as_hung`).
    mirrored: bool = False

    def at_face(self, face_mm: float) -> SolidSpec:
        """The same solid at another face size, the patch and marker following
        by the rule above."""
        from dataclasses import replace

        patch = face_mm - 2.0
        return replace(self, face_mm=face_mm, patch_mm=patch, marker_mm=patch * 6.0 / 8.0)


# The marker solid on each handheld. A six-faced box showed one face alone from about a
# third of the directions a hand holds it in even at 1280 wide, and one face is
# a plane: a plane fits two poses, and the arm follows both in turn. Eighteen
# squares forty-five degrees apart never do that. The whole solid is kept:
# it hangs from the roof by its blank forward face, the one the work hides,
# and the face opposite the roof - the one toward the hand - is blank too and
# carries the driver board's mount. A dome with the hand's half cut away
# showed two faces from the attitude a hand reaches down in, and two faces
# measure twice as loosely as four. The other sixteen are marked, 30 mm faces
# with 21 mm of marker on printed tiles, whose fit was measured on the P2S. Ids 0..15 on the left and 16..31 on
# the right, so the two can share a room.
HANDHELD_LEFT_SOLID = SolidSpec("handheld_left", 30.0, 28.0, 21.0, first_marker_id=0,
                                shape="rhombicuboctahedron", hand_letter="L", mirrored=True)
HANDHELD_RIGHT_SOLID = SolidSpec("handheld_right", 30.0, 28.0, 21.0, first_marker_id=16,
                                 shape="rhombicuboctahedron", hand_letter="R")
# The domes glued before the whole solid, until those are replaced. The same
# faces under the same ids as they were printed with.
DOME_LEFT_SOLID = SolidSpec("dome_left", 36.0, 34.0, 25.5, first_marker_id=0,
                            shape="rhombicuboctahedron_dome", mirrored=True)
DOME_RIGHT_SOLID = SolidSpec("dome_right", 36.0, 34.0, 25.5, first_marker_id=12,
                             shape="rhombicuboctahedron_dome")
# The solid on the desk: the world. The eighteen-square solid cut flat under
# its side squares, standing on that cut, so its +Z is up and its frame IS the
# world. Thirteen marked faces on the half the head can see, the +Y side among
# them since no stalk enters it; the half under the desk would carry nothing.
# Two faces are never coplanar, so from wherever the head looks one solid pins
# the pose outright. The hands' size and the hands' tiles - 30 mm faces with
# 21 mm of marker on 28 mm printed tiles
# - read by the same head from the same distance as the hands are. Ids
# 32..44, clear of both hands' 0..31, in the same 50-id dictionary. A take
# keeps the face size it was recorded with (meta.json "desk"), and where the
# solid stood in the robot's world is reckoned from that solid's size
# (desk_solid.world_T_desk).
DESK_SOLID = SolidSpec("desk", 30.0, 28.0, 21.0, first_marker_id=32,
                       shape="desk_dome", sockets=False)

# The faces of each shape, by outward normal, in the order that assigns ids.
# The six axial faces are named for their axis, the twelve edge faces for the
# two axes their normal lies between. This order IS the id assignment: a
# reordered list renumbers the faces, and a renumbered face is a pose that is
# confidently wrong by a right angle on every frame that sees it. The whole
# solid lists the dome's faces first, in the dome's order, then the four the
# dome cut away.
SHAPE_FACES: dict[str, tuple[str, ...]] = {
    "rhombicuboctahedron": (
        "z_plus", "x_plus", "x_minus", "y_minus",
        "xp_yp", "xp_ym", "xm_yp", "xm_ym",
        "xp_zp", "xm_zp", "yp_zp", "ym_zp",
        "xp_zm", "xm_zm", "yp_zm", "ym_zm",
    ),
    # the same solid with everything under the side squares cut away, and the
    # forward face left blank
    "rhombicuboctahedron_dome": (
        "z_plus", "x_plus", "x_minus", "y_minus",
        "xp_yp", "xp_ym", "xm_yp", "xm_ym",
        "xp_zp", "xm_zp", "yp_zp", "ym_zp",
    ),
    # the dome with nothing to hang from: every one of its thirteen squares
    # marked, the +Y side included, since nothing enters it
    "desk_dome": (
        "z_plus", "x_plus", "x_minus", "y_minus",
        "xp_yp", "xp_ym", "xm_yp", "xm_ym",
        "xp_zp", "xm_zp", "yp_zp", "ym_zp",
        "y_plus",
    ),
}
# Half the side of a square face, times this, is how far the face sits from the
# centre: 1+sqrt2 for the rhombicuboctahedron (all eighteen of its squares are
# the same distance out - it is one insphere for the squares).
SHAPE_INRADIUS_PER_HALF_FACE: dict[str, float] = {
    "rhombicuboctahedron": 1.0 + 2.0 ** 0.5,
    "rhombicuboctahedron_dome": 1.0 + 2.0 ** 0.5,
    "desk_dome": 1.0 + 2.0 ** 0.5,
}
# Which face the stalk enters, so the solid can be held; it carries no marker.
# Both hang by the face the camera never sees - the one that faces the work.
# A shape not listed here has no blank face: it stands, and is marked all over.
SHAPE_BLANK_FACE: dict[str, str] = {
    "rhombicuboctahedron": "y_plus",
    "rhombicuboctahedron_dome": "y_plus",
}
# Which face carries the driver board's mount, for a shape that gives it a
# whole face: a second stalk socket, the same size as the roof's. The dome
# gives it the flat cut instead.
SHAPE_BOARD_FACE: dict[str, str] = {
    "rhombicuboctahedron": "z_minus",
}
# The dome is the whole solid cut flat this far below its centre, in half-faces:
# just under the four side squares, so those stay whole.
SHAPE_CUT_PER_HALF_FACE: dict[str, float] = {
    "rhombicuboctahedron_dome": 1.0,
    "desk_dome": 1.0,
}

# Where the marker solid hangs on the handheld, and therefore where the tool
# centre is when the solid is somewhere. It goes on a square peg out of the
# end of the handheld's wrist, pointing down the tool, set toward the camera
# side so the solid's face on the peg stands flush with the wrist's
# camera-side face - so this is not a measurement: it is the peg's place
# (Settings.peg_mm, wrist_mm) and direction in the claw's mesh
# frame (meshes/handheld_body_*.stl) put together with that frame's place
# against the gripper frame (the description).
#
# The peg is square, so a solid goes on it four ways and only one of them is
# this one: the solid's +X along the claw's +X, its marker face +Z toward the
# camera side, which is the way the printed faces are meant to land. A solid
# pushed on a quarter turn out is a pose wrong by ninety degrees on every frame,
# and the ids in the live view are what say which way it went on.
BARE_PEG_MM = (-2.8, -0.25, 6.5)         # where the bare peg left the roof's middle, before the wrist
PEG_DIR = (0.0, 0.0, -1.0)               # ...and which way it points
CLAW_T_TCP_MM = (-7.90, 0.00, 109.08)    # the gripper frame, in the claw's mesh frame
# The claw's frame written in the solid's axes, per shape - rows are the solid's
# x, y, z as claw vectors read down the columns: for the dome, solid x is claw x,
# solid y is claw +z (back up the peg, since it hangs by its blank +Y face) and
# solid z is claw -y (the camera side).
SHAPE_R_CLAW: dict[str, tuple[tuple[float, float, float], ...]] = {
    "rhombicuboctahedron": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "rhombicuboctahedron_dome": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}

@dataclass(frozen=True)
class Settings:
    # --- the printed solid -------------------------------------------------
    # Every solid this package prints or solves, and which one is meant now.
    # Each has its own sheet and STL; the geometry the solver is given comes
    # from the same spec the sheet was printed from, so they cannot drift apart.
    solids: tuple[SolidSpec, ...] = (HANDHELD_LEFT_SOLID, HANDHELD_RIGHT_SOLID,
                                     DOME_LEFT_SOLID, DOME_RIGHT_SOLID, DESK_SOLID)
    solid: str = "handheld_left"
    # The solid that is the world: solved in every frame beside the hands', and
    # every hand pose is reported in its frame. See DESK_SOLID.
    desk_solid: str = "desk"

    # Depth of the pocket the paper sits in. 0.4 mm is two layers at 0.2 mm and
    # about the thickness of paper plus glue, so the patch ends up flush: a patch
    # standing proud catches on the jaws and peels.
    recess_mm: float = 0.4
    # Per-side slack in the pocket. Paper cut by hand is never exactly 48 mm.
    recess_clearance_mm: float = 0.2
    # A solid for printed marker tiles in place of the paper: the pocket a tile deep,
    # its (-x, -y) corner filled so a tile with that corner cut this far along
    # each edge goes in one way round, and crush ribs on its walls - two a wall,
    # standing this far off it, this wide where they meet it - that the tile
    # presses flat going in and is held by. They stop short of the face so the
    # tile starts in square. Nought, the paper's plain pocket.
    tile_key_mm: float = 0.0
    tile_rib_mm: float = 0.0
    tile_rib_width_mm: float = 0.8
    tile_rib_lead_mm: float = 0.3
    # The face's name, cut into the middle of the pocket's floor - read while
    # gluing, covered by the paper after. Glyph height, stroke, and depth below
    # the floor.
    name_mm: float = 4.0
    name_stroke_mm: float = 0.7
    name_depth_mm: float = 0.6
    hand_letter_mm: float = 9.0
    hand_letter_stroke_mm: float = 1.2

    # Arrow printed at the top edge of each patch, inside the cut line. The
    # marker's own frame is what PnP returns, so a patch glued a quarter turn
    # round yields a confidently wrong solid pose that no residual catches. Kept to
    # the outermost 1.5 mm, so the marker keeps more than half a module of white
    # inside it even on the hands' 28 mm patches.
    up_mark_mm: float = 1.5

    # The stalk socket in the blank face: a square blind hole, so the handle
    # cannot rotate in it. A round one would, and a solid that has turned on its
    # stalk since the last take is a solid whose pose is wrong by however far it
    # turned - with nothing in the frame to say so.
    stalk_mm: float = 8.0
    stalk_depth_mm: float = 15.0
    # How far below the handheld's roof its solid hangs: the body's wrist runs
    # the roof's outline this much further down and the peg leaves its end,
    # the room under the camera's back for its cable to leave toward the hand.
    wrist_mm: float = 15.0
    # Where the peg's axis meets the roof's underside, claw mesh frame: toward
    # the camera side by as much as puts the solid's face on the peg flush
    # with the wrist's camera-side face (build_handheld_body.py PEG_X, PEG_Y,
    # PEG_Z).
    peg_mm: tuple[float, float, float] = (-2.8, -9.2, 6.5)

    # Which library reads the squares. "aruco" is OpenCV's own pipeline -
    # adaptive threshold, contour, quad fit - and "apriltag" is the AprilTag
    # library's, which clusters image gradients and fits quads on a graph. The
    # second finds faces the first loses: measured on rendered frames of this
    # solid at 70 cm with a blurred hand, 2.75 faces against 1.50 for the same
    # family read by OpenCV. It costs a dependency (pupil-apriltags) and about
    # half a millisecond a frame, and it only reads the APRILTAG families, so it
    # goes with a dictionary below that starts DICT_APRILTAG_.
    #
    # Both are here because the sheet on the desk decides which one is right,
    # and swapping the sheet is a reprint and a re-glue of every face.
    detector: str = "aruco"

    # 4x4_50 over the larger dictionaries: 32 ids is all this needs, and fewer
    # bits per marker means each module is bigger on the sensor, which is what
    # decides whether a face 600 mm away decodes at all.
    aruco_dict: str = "DICT_4X4_50"


    # --- the printed sheet ------------------------------------------------
    page_mm: tuple[float, float] = (210.0, 297.0)  # A4
    # Printed ruler. A page scaled to "fit" silently shrinks every marker by a few
    # percent and every distance the pipeline reports with it, and the only way to
    # notice is to put a ruler on the paper.
    ruler_mm: float = 100.0

    # --- the top camera --------------------------------------------------
    # Its model - what it is opened as, its exposure's unit, how late its
    # frames arrive - is camera.yaml's `top` profile (top_camera, below). It
    # is told from the gripper cameras by its socket, the `top` entry of
    # cameras.yaml in the workspace (`pinkrab-umi cameras --top`). Every marker
    # pose is solved in its picture, through its lens as fitted against the
    # board (`pinkrab-umi lens fit top`); it carries no numbers of its own, so
    # until it is fitted it solves nothing.
    #
    # Its exposure, pinned: the hand's blur budget, not the room's. A marker
    # read over a long exposure is a marker that vanishes when it moves -
    # 10 ms smears a hand at 200 mm/s by 2 mm, under the 3.5 mm of a marker
    # module. Shorter is too dark in a room lit at night even at the full
    # gain: the desk solid, still, was tracked in every frame at 10 ms, 93 %
    # at 8.3 and 66 % at 6.7 (measured with the camera 0.5-0.7 m off).
    top_exposure_us: int = 10000
    # The gain is fitted at bring-up to put the picture's bright end here, and
    # then follows the room: every this many seconds the bright end is measured
    # and the gain nudged toward the target, within the module's range. Slow on
    # purpose: a hand or a sheet of paper passing the lens must not swing the
    # picture the detector is reading. 0 turns the following off.
    top_target_p99: float = 160.0
    top_gain_follow_s: float = 2.0
    top_gain_deadband: float = 0.08     # leave the gain alone within this fraction of target

    # --- the gripper cameras ----------------------------------------------
    # One on each handheld, looking down the jaw; their model is camera.yaml's
    # `gripper` profile (gripper_camera, below). The two are told apart by where
    # they are plugged in, which differs from one rig to the next: `pinkrab-umi
    # cameras` records it in the workspace's cameras.yaml, and until it has,
    # neither side has a camera.
    #
    # The stream to the page is a smaller JPEG, made from every other frame:
    # two full-size encodes at 30 would be the loop's whole budget. Once the
    # lens is measured it is the dataset's picture instead (GRIPPER_WIDTH);
    # this is the width of a whole frame while it is not.
    gripper_stream_width: int = 960
    gripper_stream_every: int = 2
    # Every camera's frames, for any other process: each frame, whole, in
    # shared memory under this name - {camera} `top`, or a gripper set and side
    # (`follower_left`) - written by the thread reading the camera as it
    # arrives (frame_share); a policy reads the arms' each tick. A frame older
    # than this is no picture, and a policy holds: some seven frames at 30
    # fps, past any one late frame.
    camera_share_name: str = "pinkrab_cam_{camera}"
    camera_share_stale_s: float = 0.25
    # How long after the jaw's angle is read it is stamped on this host's
    # clock, for laying the handheld's streams on the top camera's exposures:
    # half a reader period at teleop_fps plus one serial read. The cameras'
    # are their profiles' (gripper_camera_latency_s, top_camera_latency_s).
    jaw_latency_s: float = 0.012
    # A rollout's robot readings, the same way: the driver reads the bus at
    # arm_poll_hz and the bridge passes the latest on at its push_hz (both 60),
    # each holding a reading up to a period before the studio stamps it - half
    # a period each on average.
    robot_state_latency_s: float = 0.017

    # --- the gripper lens, measured ----------------------------------------
    # A ChArUco board on one A4 sheet, landscape: 9 x 6 squares of 30 mm with
    # 21 mm markers from a dictionary the solids do not use, so a solid in view
    # is never taken for the board. A 5x5 marker is 7 modules with its border,
    # 3 mm each, and its 4.5 mm inset in the square is 9 pixels at two a
    # millimetre: OpenCV's own drawing of the board at that scale is exact,
    # and the PDF is that drawing, pixel for pixel.
    lens_board_squares: tuple[int, int] = (9, 6)
    lens_board_square_mm: float = 30.0
    lens_board_marker_mm: float = 21.0
    lens_board_dict: str = "DICT_5X5_100"
    lens_page_mm: tuple[float, float] = (297.0, 210.0)
    # A view is kept when this many corners are found in it, the board has
    # stopped moving, and enough of its corners fall where few kept views have
    # put one - a fisheye is only known where it has been shown the board.
    lens_min_corners: int = 12
    lens_views: int = 30
    lens_gap_s: float = 0.7
    lens_still_px: float = 1.5
    lens_grid: tuple[int, int] = (8, 5)
    lens_cell_full: int = 3
    lens_new_fraction: float = 0.25
    # A view the board has not moved from since the last kept one - by this
    # much, median corner - is that view again, and not kept.
    lens_moved_px: float = 20.0
    # No frame for this long and the camera has stopped, not slowed.
    lens_stall_s: float = 3.0
    # The top camera's views, kept the same way: more of
    # them, since a board a desk away covers less of each, and a view whose
    # board is tilted this far from every kept one's is new as well - an
    # ordinary lens is known from the board's tilts as much as from where it
    # lands.
    top_lens_views: int = 40
    top_lens_new_tilt_deg: float = 15.0
    # The board's facings - ways it faced, each this far from the rest - a
    # capture asks for and a fit refuses fewer than.
    top_lens_min_facings: int = 4
    # A board shown flat on the desk, however many places it is slid to, puts
    # every view at the same tilt, and the focal length is then hardly known
    # (+-16 %, measured); a fit whose focal length the views know
    # no better than this fraction is refused.
    top_lens_max_f_sd: float = 0.01
    # The top lens's nominal horizontal field of view is its profile's
    # (top_lens_nominal_hfov_deg, below).

    # --- solving ----------------------------------------------------------
    # A face seen edge-on gives four nearly collinear corners and a pose that
    # swings; below this the face is dropped rather than trusted.
    min_face_area_px: float = 100.0
    # One face whose corners disagree with the rest: left out when the solve
    # over the others fits this fraction as well or better - a clean frame loses
    # a few percent by dropping a face, a frame carrying one bad face loses
    # most of its residual. Only tried above this many pixels: below it every
    # face is within its own noise and there is nothing to explain.
    face_outlier_gain: float = 0.5
    face_outlier_min_px: float = 0.75
    # Above this the solve is reported but flagged and the frame not used: a
    # face read off another solid, a patch glued askew. A face_size_mm that is
    # not the solid's hardly shows here - a solid scaled whole fits as well,
    # only nearer or further. With the top camera on the head the hands are
    # 25-40 cm away and their markers 25-60 px across: the residual of a good
    # solve then runs 1.2-1.6 px and reaches 2.7 px, and a frame near the top
    # of that moves no more from its neighbours than the rest (0.2-1.4 mm,
    # 0.2-1.2 deg).
    max_reproj_px: float = 3.5
    # Faces the reader found as squares but could not decode are read again
    # where the solid's pose, from the faces that did decode, says they are
    # (pnp._recover). What it buys is a face count that holds from frame to
    # frame instead of flickering at the edge of reading.
    recover_faces: bool = True
    # On the live picture, a face that was drawn a moment ago and is not read
    # this frame stays drawn, dimmer, for this many frames. Drawing only: the
    # pose is solved from what this frame reads. A face at the edge of reading
    # comes and goes every few frames, and an outline that blinks with it reads
    # as a pose that blinks, which it is not.
    draw_hold_frames: int = 8
    # The top camera's picture also shows each hand the Hermit where the
    # robot's ready pose would hold it - a take starts there, and a hand that
    # only waits waits there - and turns it green once the hand is held within
    # this of it (pinkrab_umi.ready_guide). Near enough that takes start alike,
    # not that the hand lands on it: a hand anywhere this near the ready pose
    # is one the arm meets without swinging shoulder_pan round, five times in
    # six.
    ready_guide_mm: float = 100.0
    ready_guide_deg: float = 20.0

    # --- the stream filter (single-face ambiguity) ------------------------
    # A single face fits two poses. When the worse solution's residual is more
    # than this many times the better one's, the frame ranks its own answers and
    # is trusted alone; nearer than this, the two are indistinguishable and the
    # choice goes to continuity. Near face-on the ratio approaches one.
    filter_ambiguity_ratio: float = 2.0
    # How long the last accepted pose can vouch for an ambiguous frame. A second
    # covers the flicker of a face dropping out at the area gate and back; much
    # longer and the solid may genuinely be somewhere else by now.
    filter_anchor_max_age_s: float = 1.0
    # The fastest the solid is allowed to have turned between accepted frames. A
    # wrist can flick at a few hundred degrees a second; a flip lands the pose
    # roughly a mirror away in one frame, which at 30 fps reads as thousands.
    filter_max_rot_speed_dps: float = 480.0
    # How much of each accepted pose the smoothed pose takes. One is no
    # smoothing. At 0.3 and 30 fps the average settles in three frames and the
    # standing noise of a still solid - about a millimetre and half a degree at
    # arm's length - comes out at 0.45 of itself. Lower is calmer and later.
    filter_ema_alpha: float = 0.3

    # --- the live view ----------------------------------------------------
    # The port the live browser view listens on. One above the studio's 8730, so
    # both can be up at once on the rig: the studio is the robot and this is the
    # camera watching the gripper, and a take is set up with both open.
    live_port: int = 8731

    # How many solved frames the live statistics are taken over. At 30 fps this is
    # eight seconds - long enough that the standard deviation of a solid held still
    # has settled, short enough that moving the solid to the next distance clears
    # the window before the arm is back out of shot.
    live_window_frames: int = 240

    # The longest take the live view will record before stopping it itself. A
    # closure take is half a minute of moving plus a still start and end; four
    # minutes is room for a slow one, and a forgotten one stops filling the disk
    # at ninety megabytes a minute instead of overnight.
    take_max_seconds: float = 240.0

    # --- the desk solid in the robot's world --------------------------------
    # The desk solid has to stand on the table, and the operator sits a little
    # back from the table's edge, so it stands out in front of their chest. The
    # robot reckons it against its board, midway between the arms - the chest's
    # place - and the hands this much further out than the solid: a hand
    # against the solid is carried this far further out, where the arms work.
    # The studio's hands view, the shelf's replay and a policy run's start
    # take it (desk_solid.world_T_desk).
    # Set so the solid laid against the near edge of the task mat is reckoned
    # against that same edge on the robot's desk, where the mat lies with its
    # near edge on the shoulder pan axes, 105.5 mm out from the board: that
    # less the solid's 72.4 mm depth (30 mm faces).
    desk_reckoned_back_m: float = 0.033

    # --- closure thresholds ----------------------------------------------
    # A take that starts and ends on the same physical fixture should come back to
    # the same pose. What it comes back off by is the whole error budget of the
    # pipeline - detection, intrinsics, and the sheet solve together - so this is the
    # number that says whether the rig is usable, not any single stage's residual.
    closure_translation_mm: float = 10.0
    closure_rotation_deg: float = 3.0
    # Frames with no usable face are a hole in the demonstration. Below this the
    # take is not worth keeping however good the frames that did solve.
    min_coverage: float = 0.90

    # --- what a take must hold to be trained on ------------------------------
    # The shelf judges each take by these, after UMI's own dataset plan, which
    # drops a video with more than ten frames lost to tracking or fewer than
    # sixty tracked, at 60 fps: stated here in seconds, so they hold at any rate.
    # The working hand is the one whose jaw moved; a frame is tracked when its
    # solid solved, as UMI counts tracking, and the jaw's reading is counted
    # apart (keep_warn_jaw_read).
    keep_max_lost_s: float = 10 / 60
    # A hole this short or shorter in a take - a frame or few without a hand's
    # solid, the desk solid or a gripper picture just then - is bridged when
    # it is converted rather than cut: each pose interpolated across it from
    # the frames either side, each picture the last one taken. Two hands and
    # three cameras make one such frame every second or two, and a cut there
    # leaves two short pieces, one of them starting mid-task. The shelf lets
    # a take lose this much (keep_max_lost_s).
    convert_max_hole_s: float = 10 / 60

    # --- what of the gripper pictures is kept (keep_masks) -------------------
    # SAM 3 tracks the task's things in an environment of its own: it wants a
    # newer torch than LeRobot allows. This is that environment's Python, and
    # the size SAM 3 reads each picture at (a multiple of 336: 1008 is its
    # own, 672 keeps a Hermit's things at a third of the time).
    sam3_env: str = "sam3-env"
    # What that environment is made of, on a machine the Curate page sets it
    # up on (the install page's command, the same three): SAM 3 ported to run
    # on MPS and CUDA alike, and the torch it needs.
    sam3_packages: tuple[str, ...] = (
        "torch>=2.15", "torchvision", "sam3[video] @ git+https://github.com/woolimi/sam3-silicon-mac.git")
    sam3_resolution: int = 672
    # SAM 3's bars, in place of its own 0.7 and 0.5: a thing starts being
    # followed where it scores sam3_new_thing_score and is held where it
    # scores sam3_thing_score. Its own bars lose a small thing seen from
    # afar - a block a fisheye sees across the mat scores about 0.3.
    sam3_new_thing_score: float = 0.4
    sam3_thing_score: float = 0.3
    # What is kept of a gripper picture is grown by this much before the rest
    # is painted over, so an edge drawn or tracked a little short of the
    # thing does not eat into it (backgrounds.kept).
    keep_grow_px: int = 6
    keep_min_held_s: float = 60 / 60
    # UMI filters what the robot cannot follow by its kinematics and gives no
    # number: this share is ours - of the working hand's frames on the replay,
    # those the replay could not place left out, the ones the arm follows: no
    # miss, jump or contact, a joint at its limit being only pointed out;
    # none for a hand the replay never placed. A frame is not placed
    # when the solid was not solved, which keep_max_lost_s judges, or when only
    # the jaw went unread, which keep_warn_jaw_read warns of.
    keep_min_feasible: float = 0.80
    # UMI warns, and keeps the take, below this share of frames with the
    # gripper's state read.
    keep_warn_jaw_read: float = 0.90
    # A take whose jaw opened and closed by less than this share of its travel
    # grasped nothing.
    keep_min_jaw_spread: float = 0.20
    # A hand that let go of what it carried - its jaw's last opening - draws
    # back off it and rises, each by at least this much, before the take
    # stops: a take that stops at the release teaches a policy to hover over
    # what it put down. Back is along the tool's approach at the release,
    # level with the desk. 0 leaves it unjudged, for a task that ends held. A
    # hand that ends back at the ready pose (within ready_guide_mm and
    # ready_guide_deg) has drawn off however it went.
    keep_min_retract_mm: float = 20.0
    # Every camera of a take recorded it whole: each video at this share of
    # its rate at least, never this long without a frame, and its picture
    # changing - a camera that sent one picture over and over, or none, gives
    # a policy nothing to look at. The change is the most the picture moves
    # between samples a third of a second apart, in grey levels.
    keep_min_stream_share: float = 0.8
    keep_max_frame_gap_s: float = 0.25
    keep_min_picture_change: float = 1.0
    # A hand lets go of what it carried by opening its jaw wider than it
    # carried at: an arm has to open its own to let go, and a take whose jaw
    # only shut - the object slipping from a slack hand - teaches it to carry
    # the object off again. False for a task that lets go by squeezing.
    keep_release_opens: bool = True
    # An arm the replay swings this far round shoulder_pan from the ready pose
    # to follow its hand is pointed out: held where the arm meets it only
    # turned aside - too high, too near or too flat - or a task that reaches
    # far to one side.
    keep_warn_pan_from_ready_deg: float = 60.0

    # --- derived ----------------------------------------------------------
    # The cameras, as camera.yaml's profiles state them - read there, so a
    # camera is changed in that file alone.
    @property
    def top_camera(self) -> CameraProfile:
        return CORE.camera_profile("top")

    @property
    def gripper_camera(self) -> CameraProfile:
        return CORE.camera_profile("gripper")

    @property
    def top_camera_width(self) -> int:
        return self.top_camera.width

    @property
    def top_camera_height(self) -> int:
        return self.top_camera.height

    @property
    def fps(self) -> int:
        """The top camera's rate: the clock of every pose solved."""
        return self.top_camera.fps

    @property
    def top_exposure_unit_us(self) -> float:
        return self.top_camera.exposure_unit_us

    @property
    def top_camera_latency_s(self) -> float:
        return self.top_camera.latency_s

    @property
    def top_lens_nominal_hfov_deg(self) -> float:
        """Where a fit starts from, and what the board's tilt is shown with
        while views are kept - never a pose."""
        if self.top_camera.hfov_deg is None:
            raise ValueError(f"{CORE.camera_file}: the top profile names no hfov_deg for a fit to start from")
        return self.top_camera.hfov_deg

    @property
    def gripper_camera_usb(self) -> tuple[int, int] | None:
        return self.gripper_camera.usb

    @property
    def gripper_camera_width(self) -> int:
        return self.gripper_camera.width

    @property
    def gripper_camera_height(self) -> int:
        return self.gripper_camera.height

    @property
    def gripper_camera_fps(self) -> int:
        return self.gripper_camera.fps

    @property
    def gripper_camera_latency_s(self) -> float:
        return self.gripper_camera.latency_s

    @property
    def solid_spec(self) -> SolidSpec:
        for spec in self.solids:
            if spec.name == self.solid:
                return spec
        raise KeyError(f"no solid named {self.solid!r}; have {[c.name for c in self.solids]}")

    def for_solid(self, name: str) -> Settings:
        """The same settings, meant for another of the solids."""
        from dataclasses import replace
        chosen = replace(self, solid=name)
        chosen.solid_spec  # raises early if the name is not one of ours
        return chosen

    def at_face(self, name: str, face_mm: float) -> Settings:
        """The same settings, meant for the solid `name` with `face_mm` faces."""
        from dataclasses import replace
        spec = self.for_solid(name).solid_spec
        if abs(face_mm - spec.face_mm) > 1e-6:
            spec = spec.at_face(face_mm)
        return replace(self, solid=name, solids=tuple(spec if c.name == name else c for c in self.solids))

    def as_recorded(self, name: str, meta: dict) -> Settings:
        """The hand solid `name` as the take whose meta.json is `meta` had it.

        A pose solved live is of the solid on the handheld then, so its tool
        centre is found with that solid's size, which the take records under
        "hands". KeyError when the take does not say: its poses could be of
        any size."""
        return self.at_face(name, float((meta.get("hands") or {})[name]))

    def recorded_solid_T_tcp(self, name: str, meta: dict):
        """The tool centre in hand solid `name`'s frame as the take whose
        meta.json is `meta` had it, a 4x4 in metres.

        A take keeps it under "solid_T_tcp". One from before it did keeps only
        its solids' face sizes, under "hands", and hung them against the roof
        on the bare peg in its middle. KeyError when the take says neither."""
        import numpy as np
        from dataclasses import replace

        kept = (meta.get("solid_T_tcp") or {}).get(name)
        if kept is not None:
            return np.array(kept, dtype=float)
        return replace(self.as_recorded(name, meta), wrist_mm=0.0, peg_mm=BARE_PEG_MM).solid_T_tcp

    @property
    def face_size_mm(self) -> float:
        return self.solid_spec.face_mm

    @property
    def patch_mm(self) -> float:
        return self.solid_spec.patch_mm

    @property
    def marker_mm(self) -> float:
        return self.solid_spec.marker_mm

    @property
    def first_marker_id(self) -> int:
        return self.solid_spec.first_marker_id

    @property
    def shape(self) -> str:
        return self.solid_spec.shape

    @property
    def marker_faces(self) -> tuple[str, ...]:
        return SHAPE_FACES[self.shape]

    @property
    def blank_face(self) -> str | None:
        return SHAPE_BLANK_FACE.get(self.shape)

    @property
    def inradius_mm(self) -> float:
        """Centre to any marker face."""
        return self.face_size_mm / 2.0 * SHAPE_INRADIUS_PER_HALF_FACE[self.shape]

    @property
    def mount_out_mm(self) -> float:
        """Centre to the face the stalk enters - and so to the surface it hangs from.

        Every shape hangs by one of its square faces, so this is the inradius.
        """
        return self.inradius_mm

    @property
    def board_face(self) -> str | None:
        """The face whose socket takes the driver board's mount, if a face does."""
        return SHAPE_BOARD_FACE.get(self.shape)

    @property
    def socket_faces(self) -> tuple[str, ...]:
        """Every face with a stalk socket in it and no marker on it."""
        if not self.solid_spec.sockets or self.blank_face is None:
            return ()
        return (self.blank_face,) + ((self.board_face,) if self.board_face else ())

    @property
    def cut_mm(self) -> float | None:
        """Centre to the flat cut, below it, for a shape that has one."""
        k = SHAPE_CUT_PER_HALF_FACE.get(self.shape)
        return None if k is None else self.face_size_mm / 2.0 * k

    @property
    def board_out_mm(self) -> float:
        """Centre to the surface the board's mount plugs into, down the solid's -Z.

        A whole solid gives it a face, at the inradius; the dome gives it the cut.
        """
        return self.inradius_mm if self.board_face else float(self.cut_mm)

    @property
    def face_ids(self) -> dict[str, int]:
        """'z_plus' -> 0, ... - ids follow the face order rather than being typed."""
        return {name: self.first_marker_id + i for i, name in enumerate(self.marker_faces)}

    @property
    def id_faces(self) -> dict[int, str]:
        return {i: name for name, i in self.face_ids.items()}

    @property
    def pocket_mm(self) -> float:
        """Side of the recess cut into each face."""
        return self.patch_mm + 2 * self.recess_clearance_mm

    @property
    def mesh_dir(self) -> Path:
        """The handheld's printed parts, as the studio draws them.

        Installed under share/ by colcon, beside the package in a checkout:
        searched for rather than counted up to, as the description is.
        """
        here = Path(__file__).resolve()
        for parent in here.parents:
            for candidate in (parent / "share" / "pinkrab_umi" / "meshes", parent / "pinkrab_umi" / "meshes"):
                if candidate.is_dir():
                    return candidate
        return here.parents[1] / "meshes"

    @property
    def headcam_dir(self) -> Path:
        """Where recorded sessions land - beside the calibration, outside the repo.

        A session is thousands of PNGs of a particular room on a particular head.
        It describes this rig the way the calibration does, so it lives where that
        does and is never committed.
        """
        return CORE.workspace / "headcam"

    def session_dir(self, session: str) -> Path:
        return self.headcam_dir / session

    @property
    def lens_dir(self) -> Path:
        """The board views a lens is fitted from, a directory per camera."""
        return CORE.workspace / "lens"

    @property
    def episodes_dir(self) -> Path:
        """Where recorded episodes land: pinkrab_core's (`episodes_dir`)."""
        return CORE.episodes_dir

    @property
    def sam3_python(self) -> Path:
        """The Python of SAM 3's environment, in the workspace (sam3_env)."""
        return CORE.workspace / self.sam3_env / "bin" / "python"

    def camera_share_lock(self, camera: str) -> Path:
        """The file whose lock guards a camera's shared frame (frame_share)."""
        return CORE.workspace / "frame_share" / f"{camera}.lock"

    @property
    def curate_file(self) -> Path:
        """The Curate page's choice of machine for SAM 3, kept across restarts."""
        return CORE.workspace / "curate.json"

    # The videos: H.264 through PyAV at a constant quality, a keyframe every
    # second. OpenCV's own writer has no quality knob and wrote 1080p at some
    # 5 Mbit/s - a 7 s episode was 14 MB; at crf 30 and the scale below an
    # 8.8 s take's two gripper videos are 1.1 MB.
    # A gripper video is kept at `episode_sensor_scale` of the sensor's
    # pixels, the scale a dataset keeps its window at (GRIPPER_WIDTH across
    # the window's 1560 sensor pixels): the 1280x960 mode's frame is kept at
    # 800x600, and the window is cut from it without scaling. SAM 3's keep
    # labels and the randomised pictures come out of it as out of the frame
    # as read. Written on a thread per video; a frame that finds its queue
    # this full is dropped and counted rather than making the camera wait.
    # `episode_codecs` is the fallback order for OpenCV's writer when PyAV
    # cannot encode here.
    episode_codec: str = "libx264"
    episode_crf: int = 30
    episode_sensor_scale: float = 0.5
    episode_preset: str = "veryfast"
    episode_gop: int = 30
    episode_codecs: tuple[str, ...] = ("avc1", "mp4v")
    episode_queue_frames: int = 90

    @property
    def solid_on_peg_mm(self):
        """Where this solid's centre sits in the claw's frame, on the peg.

        The peg is one size, one place and one direction whatever goes on it, so
        a solid's centre is the wrist and its mounting face's distance out along
        the peg.
        """
        import numpy as np

        return np.array(self.peg_mm, dtype=float) + np.array(PEG_DIR, dtype=float) * (self.wrist_mm + self.mount_out_mm)

    def _as_hung(self, right):
        """A frame on the right handheld, as the handheld this solid hangs on
        has it, both in that handheld's own claw frame.

        The left handheld is the right one's mirror image across the claw's
        y = 0 plane - its claw is the right claw mirrored, as its arm is - so
        each of its frames is the right's reflected across that plane and
        turned along its own x to stay right-handed. Its tool centre comes out
        where the description puts the left gripper frame, turned half round
        about the approach as TOOL_TURN has it, and its solid on the mirrored
        peg, turned the same half round about the peg."""
        import numpy as np

        if not self.solid_spec.mirrored:
            return right
        across = np.diag([1.0, -1.0, 1.0, 1.0])
        own_x = np.diag([-1.0, 1.0, 1.0, 1.0])
        return across @ right @ own_x

    @property
    def claw_T_solid(self):
        """The solid on its peg, in the claw's frame, as a 4x4 in metres."""
        import numpy as np

        out = np.eye(4)
        out[:3, :3] = np.array(SHAPE_R_CLAW[self.shape], dtype=float)
        out[:3, 3] = self.solid_on_peg_mm / 1000.0
        return self._as_hung(out)

    @property
    def solid_T_tcp(self):
        """The tool centre in the solid's frame, as a 4x4 in metres.

        What a solved pose has to be composed with to become a gripper pose:
        cam_T_solid @ solid_T_tcp is where the tool centre was. Derived rather
        than written down, because the one thing it depends on that is not
        fixed is which solid is on the peg.
        """
        import numpy as np

        claw_tcp = np.eye(4)
        claw_tcp[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        claw_tcp[:3, 3] = np.array(CLAW_T_TCP_MM, dtype=float) / 1000.0
        return np.linalg.inv(self.claw_T_solid) @ self._as_hung(claw_tcp)


SETTINGS = Settings()
