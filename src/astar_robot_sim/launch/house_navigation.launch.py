import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    pkg_tb3  = get_package_share_directory('turtlebot3_gazebo')
    pkg_sim  = get_package_share_directory('astar_robot_sim')
    pkg_gz   = get_package_share_directory('gazebo_ros')
    planner  = LaunchConfiguration('planner')

    world_file   = os.path.join(pkg_sim, 'worlds', 'house_with_actors.world')
    launch_dir   = os.path.join(pkg_tb3, 'launch')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose       = LaunchConfiguration('x_pose', default='-2.0')
    y_pose       = LaunchConfiguration('y_pose', default='-0.5')

    return LaunchDescription([
        DeclareLaunchArgument('planner',      default_value='astar'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose',       default_value='-2.0'),
        DeclareLaunchArgument('y_pose',       default_value='-0.5'),

        # Gazebo server with our custom world (house + actors)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gz, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world_file}.items(),
        ),

        # Gazebo client (GUI)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gz, 'launch', 'gzclient.launch.py')
            ),
        ),

        # TurtleBot3 robot state publisher (sets URDF / robot_description)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'robot_state_publisher.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # Spawn TurtleBot3 burger using the official spawn launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'spawn_turtlebot3.launch.py')
            ),
            launch_arguments={
                'x_pose': x_pose,
                'y_pose': y_pose,
            }.items(),
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

        # Robot controller
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
