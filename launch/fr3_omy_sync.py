#!/usr/bin/python3

import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = WORKSPACE_ROOT / "launch" / "FR3_omy_bridge.py"


def generate_launch_description():
    port_name = LaunchConfiguration("port_name")

    bringup_launch = (
        Path(get_package_share_directory("open_manipulator_bringup"))
        / "launch"
        / "omy_l100_leader_ai.launch.py"
    )

    leader = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(bringup_launch)),
        launch_arguments={
            "port_name": port_name,
            "use_sim": "false",
            "use_mock_hardware": "false",
            "use_self_collision_avoidance": "true",
        }.items(),
    )

    fr3_bridge = ExecuteProcess(
        cmd=["/usr/bin/python3", str(BRIDGE_SCRIPT)],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "port_name",
            default_value="/dev/ttyUSB0",
            description="Serial port used by the OMY-L100 leader arm.",
        ),
        leader,
        fr3_bridge,
    ])


if __name__ == "__main__":
    launch_service = LaunchService(argv=sys.argv[1:])
    launch_service.include_launch_description(generate_launch_description())
    sys.exit(launch_service.run())
