import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    pkg_tb3 = get_package_share_directory('turtlebot3_gazebo')
    planner = LaunchConfiguration('planner')

    return LaunchDescription([
        DeclareLaunchArgument(
            'planner',
            default_value='astar',
            description='Planner to use: astar or dstar'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_tb3, 'launch', 'turtlebot3_house.launch.py')
            ),
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        ),

        # A* planner
        Node(
            package='astar_robot_sim',
            executable='astar_planner',
            name='astar_planner',
            output='screen',
            condition=IfCondition(PythonExpression(["'", planner, "' == 'astar'"])),
        ),

        # D* Lite planner
        Node(
            package='astar_robot_sim',
            executable='dstar_lite_planner',
            name='dstar_lite_planner',
            output='screen',
            condition=IfCondition(PythonExpression(["'", planner, "' == 'dstar'"])),
        ),

        # Robot controller — follows selected planner path
        Node(
            package='astar_robot_sim',
            executable='robot_controller',
            name='robot_controller',
            output='screen',
            parameters=[{
                'path_topic': PythonExpression([
                    "'/dstar_path' if '", planner, "' == 'dstar' else '/astar_path'"
                ])
            }],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
