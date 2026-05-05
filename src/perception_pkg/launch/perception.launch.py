"""
ROS 2 Launch file for perception pipeline.

Launches the perception node with parameters from YAML configuration.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from os.path import join


def generate_launch_description():
    """Generate launch description for perception pipeline."""
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=join(
            FindPackageShare('perception_pkg').find('perception_pkg'),
            'config',
            'params.yaml'
        ),
        description='Path to parameters YAML file'
    )
    
    # Create perception node
    perception_node = Node(
        package='perception_pkg',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        emulate_tty=True,
        arguments=[
            '--ros-args',
            '--log-level', 'info'
        ]
    )
    
    return LaunchDescription([
        params_file_arg,
        perception_node
    ])
