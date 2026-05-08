import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    pkg_sim  = get_package_share_directory('astar_robot_sim')
    pkg_desc = get_package_share_directory('astar_robot_description')
    pkg_gz   = get_package_share_directory('gazebo_ros')

    world_file = os.path.join(pkg_sim,  'worlds', 'grid_world.world')
    urdf_file  = os.path.join(pkg_desc, 'urdf',   'astar_robot.urdf')
    planner = LaunchConfiguration('planner')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'planner',
            default_value='astar',
            description='Planner to use: astar or dstar'
        ),

        # Launch Gazebo with our world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gz, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'world': world_file}.items(),
        ),

        # Publish robot URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
        ),

        # Spawn robot into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'astar_robot',
                '-x', '0.5', '-y', '0.5', '-z', '0.1'
            ],
            output='screen',
        ),

        # Static transform: world -> map
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        ),

        # A* planner node
        Node(
            package='astar_robot_sim',
            executable='astar_planner',
            name='astar_planner',
            output='screen',
            condition=IfCondition(PythonExpression(["'", planner, "' == 'astar'"])),
        ),

        # D* Lite planner node
        Node(
            package='astar_robot_sim',
            executable='dstar_lite_planner',
            name='dstar_lite_planner',
            output='screen',
            condition=IfCondition(PythonExpression(["'", planner, "' == 'dstar'"])),
        ),

        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
