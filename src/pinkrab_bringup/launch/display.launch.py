"""Show the Pinkrab model - robot_state_publisher, joint sliders, RViz.

This is how the wall mount gets checked against the real thing. Put the sliders at
zero and the rendered arms should sit where the real ones sit; if they do not, the
numbers in config/mount.yaml are wrong and every later stage inherits the error.

    ros2 launch pinkrab_bringup display.launch.py
    ros2 launch pinkrab_bringup display.launch.py mount:=/path/to/another_rig.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from pinkrab_core.config import SETTINGS

DESCRIPTION = FindPackageShare("pinkrab_description")


def generate_launch_description() -> LaunchDescription:
    mount = LaunchConfiguration("mount")
    rviz_config = LaunchConfiguration("rviz_config")

    # Passed through to xacro rather than read here: the description owns what the
    # file means, and this launch only decides which one to hand it.
    # Wrapped as a string on purpose. `Command` hands back the whole expanded URDF,
    # and a parameter left unwrapped is parsed as YAML - the description's own colons
    # then make it a mapping, and the launch dies before a node starts.
    robot_description = ParameterValue(Command([
        "xacro ",
        PathJoinSubstitution([DESCRIPTION, "urdf", "pinkrab.urdf.xacro"]),
        " mount:=", mount,
    ]), value_type=str)

    return LaunchDescription([
        # Every node below on the robot's middleware, whatever this shell
        # names: two on one graph pass topics but zero the fields of a service
        # call. Our own nodes set it themselves; RViz and the publishers take
        # it from here.
        SetEnvironmentVariable("RMW_IMPLEMENTATION", SETTINGS.ros_middleware),
        DeclareLaunchArgument(
            "mount",
            default_value=PathJoinSubstitution([DESCRIPTION, "config", "mount.yaml"]),
            description="Installation geometry for this rig.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([DESCRIPTION, "rviz", "display.rviz"]),
            description="RViz layout.",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        # Sliders, not a real joint source: this launch poses a picture and
        # nothing else knows about it. For the robot the studio is running, with
        # its joint positions on /joint_states, use desktop.launch.py.
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
        ),
    ])
