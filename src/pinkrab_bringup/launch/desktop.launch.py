"""The operator's side: the robot in RViz, relayed from the studio.

This is the half that runs where a person is sitting. Nothing here opens the
servo bus - the bridge wants a network route to the studio, RViz wants a screen,
and the handheld loop wants the top camera. `robot.launch.py` is the other
half, on the machine the arms are bolted to.

The handheld loop comes up with it and stays up. Which arm it drives is decided
by the solid in frame, and whether it drives at all is the teleop mode - so
starting and stopping a session is a service call rather than a process being
launched and killed. The top camera is still started from the studio's page:
on macOS it needs root, and launch cannot type a password.

`display.launch.py` shows the description with a slider window - it poses a picture
and nothing else knows about it. This one shows the robot the studio is actually
running: the bridge republishes the simulation's joint positions, so RViz, rosbag
and anything else on the graph see the same arms the page does.

The studio is not started here. It owns the simulation and outlives any one ROS
session:

    pinkrab-studio                                     # in its own terminal
    ros2 launch pinkrab_bringup desktop.launch.py

    ros2 launch pinkrab_bringup desktop.launch.py studio:=http://192.168.1.20:8730
    ros2 launch pinkrab_bringup desktop.launch.py rviz:=false
    ros2 launch pinkrab_bringup desktop.launch.py bridge:=false   # the studio's own is up

The studio is not started here. It is a console script with no rclpy in it, it
outlives any one ROS session, and it brings up its own bridge and handheld
loop - so it is run from its own terminal, and this launch is then only wanted
for RViz:

    pinkrab-studio            # then switch the page to real
    ros2 launch pinkrab_bringup desktop.launch.py bridge:=false teleop:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from pinkrab_core.config import SETTINGS

DESCRIPTION = FindPackageShare("pinkrab_description")


def generate_launch_description() -> LaunchDescription:
    studio = LaunchConfiguration("studio")
    use_rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    mount = LaunchConfiguration("mount")

    # Wrapped as a string on purpose. `Command` hands back the whole expanded URDF,
    # and a parameter left unwrapped is parsed as YAML - the description's own colons
    # then make it a mapping, and the launch dies before a node starts.
    robot_description = ParameterValue(Command([
        "xacro ", PathJoinSubstitution([DESCRIPTION, "urdf", "pinkrab.urdf.xacro"]),
        " mount:=", mount,
    ]), value_type=str)

    return LaunchDescription([
        # Every node below on the robot's middleware, whatever this shell
        # names: two on one graph pass topics but zero the fields of a service
        # call. Our own nodes set it themselves; RViz and the publishers take
        # it from here.
        SetEnvironmentVariable("RMW_IMPLEMENTATION", SETTINGS.ros_middleware),
        DeclareLaunchArgument(
            "studio", default_value="http://127.0.0.1:8730",
            description="Where the studio is serving.",
        ),
        DeclareLaunchArgument(
            "rviz", default_value="true", description="Open RViz as well.",
        ),
        DeclareLaunchArgument(
            "bridge", default_value="true",
            description="Run the bridge. false when `pinkrab-studio --real` has "
                        "already brought up its own - two bridges are two copies "
                        "of every command. The studio also brings up the handheld "
                        "loop; pass teleop:=false with this.",
        ),
        DeclareLaunchArgument(
            "teleop", default_value="true",
            description="Run the handheld loop. It drives only in the handheld mode.",
        ),
        DeclareLaunchArgument(
            "live", default_value="http://127.0.0.1:8802",
            description="The pinkrab-umi live server the solid is solved on.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([DESCRIPTION, "rviz", "display.rviz"]),
            description="RViz layout.",
        ),
        DeclareLaunchArgument(
            "mount",
            default_value=PathJoinSubstitution([DESCRIPTION, "config", "mount.yaml"]),
            description="Installation geometry for this rig.",
        ),
        # No joint_state_publisher_gui here. The studio is the thing that decides
        # where the joints are, and two publishers on one topic is two robots.
        Node(
            package="pinkrab_bridge",
            executable="bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("bridge")),
            parameters=[{"studio": studio}],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        # The layout matters more than it looks. RViz remembers the last fixed
        # frame it was given, and this robot's root is `world`; opened on whatever
        # a previous session left behind, every link reports no transform and
        # nothing is drawn.
        Node(
            package="pinkrab_teleop",
            executable="teleop",
            output="screen",
            condition=IfCondition(LaunchConfiguration("teleop")),
            parameters=[{"live": LaunchConfiguration("live")}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            condition=IfCondition(use_rviz),
            arguments=["-d", rviz_config],
        ),
    ])
