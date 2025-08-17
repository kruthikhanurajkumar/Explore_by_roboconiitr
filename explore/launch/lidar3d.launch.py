from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    localization = LaunchConfiguration('localization')
    robot_ns = LaunchConfiguration('robot_ns')
    use_camera = LaunchConfiguration('use_camera')

    icp_odom_parameters = {
        'odom_frame_id': '/panther/odom',#change to your odom frame
        # 'guess_frame_id': 'odom',
        'OdomF2M/ScanSubtractRadius': '0.3',  # match voxel size
        'OdomF2M/ScanMaxSize': '10000'
    }

    rtabmap_parameters = {
        'subscribe_rgb': True,
        'subscribe_depth': False,
        'subscribe_rgbd': use_camera,
        'subscribe_scan_cloud': True,
        'subscribe_scan': True,
        'use_action_for_goal': True,
        'odom_sensor_sync': False,
        'delete_db_on_start': True,
        # RTAB-Map's parameters should be strings:
        'Mem/NotLinkedNodesKept': 'true',
        'Grid/RangeMin': '0.5',
        'Grid/NormalsSegmentation': 'true',
        'Grid/MaxGroundHeight': '0.10',
        'Grid/MaxObstacleHeight': '1',
        'Grid/RayTracing': 'true',
        'Grid/3D': 'true',
        'RGBD/OptimizeMaxError': '0.3',
    }

    shared_parameters = {
        'frame_id': '/panther/base_link',#change to your base link frame
        'use_sim_time': use_sim_time,
        'Reg/Strategy': '1',
        'Reg/Force3DoF': 'true',
        'Mem/NotLinkedNodesKept': 'false',
        'Icp/VoxelSize': '0.3',
        'Icp/MaxCorrespondenceDistance': '3',
        'Icp/PointToPlaneGroundNormalsUp': '0.9',
        'Icp/RangeMin': '0.5',
        'Icp/MaxTranslation': '1'
    }

    remappings = [#change these to match your topic names
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static'),
        ('odom', 'panther/odometry/filtered'),
        ('scan_cloud', '/panther/cx/lslidar_point_cloud'),
        ('rgb/image', '/panther/camera_front/image_raw'),
        ('rgb/camera_info', '/panther/camera_front/camera_info'),
        ('/scan', '/panther/cx/scan'),
        # ('depth/image', '/republished/panther/depth/image'),  # Uncomment and adjust if needed
    ]

    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='false', choices=['true', 'false'],
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'localization', default_value='false', choices=['true', 'false'],
            description='Launch rtabmap in localization mode (a map should have been already created).'),

        DeclareLaunchArgument(
            'robot_ns', default_value='',
            description='Robot namespace.'),

        DeclareLaunchArgument(
            'use_camera', default_value='false', choices=['true', 'false'],
            description='Use camera for global loop closure / re-localization.'),

        # RTAB-Map Nodes
        Node(
            condition=IfCondition(use_camera),
            package='rtabmap_sync', executable='rgbd_sync', output='screen',
            namespace=robot_ns,
            parameters=[{'approx_sync': True}, {'use_sim_time': use_sim_time}],
            remappings=remappings),

        Node(
            package='rtabmap_odom', executable='icp_odometry', output='screen',
            namespace=robot_ns,
            parameters=[icp_odom_parameters, shared_parameters, {'use_sim_time': use_sim_time}],
            remappings=remappings,
            arguments=["--ros-args", "--log-level", 'warn']),

        Node(
            condition=UnlessCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            namespace=robot_ns,
            parameters=[rtabmap_parameters, shared_parameters, {'use_sim_time': use_sim_time}],
            remappings=remappings,
            arguments=['-d']),

        Node(
            condition=IfCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            namespace=robot_ns,
            parameters=[rtabmap_parameters, shared_parameters,
                        {'Mem/IncrementalMemory': 'False',
                         'Mem/InitWMWithAllNodes': 'True'},
                        {'use_sim_time': use_sim_time}],
            remappings=remappings),


        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            namespace=robot_ns,
            parameters=[rtabmap_parameters, shared_parameters, {'use_sim_time': use_sim_time}],
            remappings=remappings),

    ])
