"""The robot's side: what runs on the machine the arms are bolted to.

Two halves, split by which machine they belong on rather than by what they do.
This one wants the rig - the serial ports, the description, and a MuJoCo it can
check a pose against. `desktop.launch.py` wants a screen and a route to the
studio. Today both run on one laptop and the split costs nothing; the day the
rig gets a machine of its own, the line is already drawn.

Two nodes. The gate takes `/joint_commands`, holds each pose against the joint
bands, the scene and a ramp, and puts what survives on
`/joint_commands_checked`. The driver holds the serial ports, reads both arms
onto `/joint_states`, and writes only what arrives on that checked topic - so
there is no route from a command to a servo that misses the gate.

Torque is off when this comes up and stays off until somebody calls
`/set_torque`. The arms hang on a wall; bringing up a launch file should not
change what they are doing. Taking it down does: on Ctrl-C the driver walks
any arm that is holding torque home, and lets go once it is there. The driver
is given longer than launch's usual five seconds before SIGTERM for that walk.

With no arms plugged in the driver says so and stops, and the gate carries on
alone - which is a robot that cannot move, and is the right thing to be left
with on a machine that has no rig attached.

    ros2 launch pinkrab_bringup robot.launch.py
    ros2 launch pinkrab_bringup robot.launch.py max_speed_dps:=120
    ros2 launch pinkrab_bringup robot.launch.py max_speed_dps:=0    # no ramp
    ros2 launch pinkrab_bringup robot.launch.py driver:=false   # the gate alone
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from pinkrab_core.config import SETTINGS

DESCRIPTION = FindPackageShare("pinkrab_description")


def generate_launch_description() -> LaunchDescription:
    scene = LaunchConfiguration("scene")
    max_speed_dps = LaunchConfiguration("max_speed_dps")
    use_driver = LaunchConfiguration("driver")

    return LaunchDescription([
        DeclareLaunchArgument(
            "scene",
            default_value=PathJoinSubstitution([DESCRIPTION, "mjcf", "scene.xml"]),
            description="The scene the gate checks a pose against - the studio's own.",
        ),
        # The number itself lives in pinkrab_core's config and is read from there,
        # not copied. Written out here it would go on being passed explicitly
        # after somebody changed the config, which is a launch file quietly
        # overriding the setting it looks like it is honouring.
        DeclareLaunchArgument(
            "max_speed_dps", default_value=str(SETTINGS.max_joint_speed_dps),
            description="The most a joint may be driven per second, degrees. Omit "
                        f"it for the project's own {SETTINGS.max_joint_speed_dps}; "
                        "0 sends every pose through whole.",
        ),
        DeclareLaunchArgument(
            "driver", default_value="true",
            description="Open the servo bus. false brings up the gate on its own.",
        ),
        # Taken from the installed description rather than from a checkout, so
        # `colcon build` proves the package ships the scene it claims to.
        Node(
            package="pinkrab_safety",
            executable="gate",
            output="screen",
            parameters=[{"scene": scene, "max_speed_dps": max_speed_dps}],
        ),
        Node(
            package="pinkrab_driver",
            executable="driver",
            output="screen",
            condition=IfCondition(use_driver),
            # Room to walk the arms home on the way down; see the driver.
            sigterm_timeout=str(SETTINGS.park_timeout_s + 4.0),
        ),
    ])
