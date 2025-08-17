#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('explore'),
                'launch',
                'navigation_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'true',
        }.items()
    )
    lidar_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('explore'),
                'launch',
                'lidar3d.launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'true',
        }.items()   
    )
    return  LaunchDescription([
        navigation_launch,
        lidar_slam,
    ])