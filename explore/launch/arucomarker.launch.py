from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='explore',
            executable='aruco_detector_node.py',
            name='detector_node',
            output='screen'
        ),
        Node(
            package='explore',
            executable='aruco_boundary_node.py',
            name='boundary_node',
            output='screen'
        )
    ])
