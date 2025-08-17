#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import tf2_ros
import threading
import time
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
import tf_transformations

from collections import deque

# point cloud helper
try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    # If sensor_msgs_py is not available, user should install it or provide an alternative.
    pc2 = None

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_detection_node')

        # Declare parameters for easy configuration
        self.declare_parameter('camera_topics', ['/front/color/image_raw','/rear/color/image_raw','right/color/image_raw','left/color/image_raw'])
        self.declare_parameter('camera_info_topics', ['/front/color/camera_info', '/rear/color/camera_info','right/color/camera_info', '/left/color/camera_info'])
        self.declare_parameter('marker_size', 0.15)
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('position_smoothing', 0.1)
        self.declare_parameter('tf_timeout', 0.5)
        self.declare_parameter('buffer_size', 10)
        self.declare_parameter('debug', False)

        # LiDAR / sensor fusion parameters (new)
        self.declare_parameter('lidar_topic', '/lidar/velodyne_points')
        self.declare_parameter('lidar_frame', 'lidar_velodyne')
        self.declare_parameter('lidar_search_radius', 0.75)  # meters
        self.declare_parameter('lidar_influence', 0.75)     # 0..1 how much lidar pulls the camera estimate toward the nearest point

        # Get parameters
        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        self.aruco_dict_name = self.get_parameter('aruco_dict').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.position_smoothing = self.get_parameter('position_smoothing').get_parameter_value().double_value
        self.tf_timeout = self.get_parameter('tf_timeout').get_parameter_value().double_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value

        # LiDAR params
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.lidar_frame = self.get_parameter('lidar_frame').get_parameter_value().string_value
        self.lidar_search_radius = self.get_parameter('lidar_search_radius').get_parameter_value().double_value
        self.lidar_influence = self.get_parameter('lidar_influence').get_parameter_value().double_value

        # ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.aruco_dict_name))
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_info = {}
        self.latest_images = {}
        self.lock = threading.Lock()

        # store latest LIDAR pointcloud message
        self.latest_lidar_msg = None
        self.latest_lidar_timestamp = None

        self.marker_pub = self.create_publisher(Float32MultiArray, '/detected_aruco_markers', 10)
        self.image_subs = []
        self.camera_info_subs = []

        # For each marker ID, we keep a queue of recent positions (for mean filtering/averaging)
        self.marker_buffers = {} # {marker_id: deque of (x, y, z)}

        for i, (img_topic, info_topic) in enumerate(zip(self.camera_topics, self.camera_info_topics)):
            camera_name = f"camera_{i}"
            img_sub = self.create_subscription(
                Image, img_topic,
                lambda msg, cam=camera_name: self.image_callback(msg, cam), 10)
            self.image_subs.append(img_sub)
            info_sub = self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, cam=camera_name: self.camera_info_callback(msg, cam), 10)
            self.camera_info_subs.append(info_sub)

        # Lidar subscription (added)
        self.lidar_sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, 5)

        self.processing_timer = self.create_timer(0.2, self.process_detections)
        self.get_logger().info("ArucoDetectionNode (with LiDAR fusion) initialized.")

    def camera_info_callback(self, msg, camera_name):
        with self.lock:
            self.camera_info[camera_name] = {
                'K': np.array(msg.k).reshape(3, 3),
                'D': np.array(msg.d),
                'frame_id': msg.header.frame_id
            }

    def image_callback(self, msg, camera_name):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_images[camera_name] = {
                    'image': cv_image,
                    'timestamp': msg.header.stamp,
                    'frame_id': msg.header.frame_id
                }
        except Exception as e:
            self.get_logger().error(f"Error processing image from {camera_name}: {e}")

    def lidar_callback(self, msg: PointCloud2):
        # Store latest LiDAR cloud for fusion. Minimal processing here.
        with self.lock:
            self.latest_lidar_msg = msg
            self.latest_lidar_timestamp = msg.header.stamp

    def detect_aruco(self, image):
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)
        detections = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                detections.append({
                    'id': int(marker_id),
                    'corners': corners[i]
                })
        return detections

    def estimate_3d_position(self, marker_corners, camera_name):
        if camera_name not in self.camera_info:
            return None
        camera_matrix = self.camera_info[camera_name]['K']
        dist_coeffs = self.camera_info[camera_name]['D']
        object_points = np.array([
            [-self.marker_size/2, -self.marker_size/2, 0],
            [self.marker_size/2, -self.marker_size/2, 0],
            [self.marker_size/2, self.marker_size/2, 0],
            [-self.marker_size/2, self.marker_size/2, 0]
        ], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points, marker_corners.astype(np.float32),
            camera_matrix, dist_coeffs
        )
        if success:
            return {
                'translation': tvec.flatten(),
                'rotation': rvec.flatten()
            }
        return None

    def transform_to_map_frame(self, position, camera_frame):
        try:
            # Use latest available transform
            point_camera = PoseStamped()
            point_camera.header.frame_id = camera_frame
            point_camera.header.stamp = rclpy.time.Time().to_msg()
            point_camera.pose.position.x = float(position['translation'][0])
            point_camera.pose.position.y = float(position['translation'][1])
            point_camera.pose.position.z = float(position['translation'][2])
            point_camera.pose.orientation.w = 1.0

            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                camera_frame,
                rclpy.time.Time(),  # Latest available
                timeout=rclpy.duration.Duration(seconds=self.tf_timeout)
            )
            # Extract translation
            tx, ty, tz = transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
            # Extract rotation as quaternion
            q = transform.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
            # Rotate the position vector
            pos_cam = np.array(position['translation'])
            rot_matrix = tf_transformations.quaternion_matrix(quat)[:3, :3]
            pos_map = np.dot(rot_matrix, pos_cam)
            # Add translation
            map_x = pos_map[0] + tx
            map_y = pos_map[1] + ty
            map_z = pos_map[2] + tz
            return {'x': map_x, 'y': map_y, 'z': map_z}
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None

    # New: find nearest point in the stored lidar cloud to a given position (in map frame).
    def find_nearest_lidar_point(self, map_pos):
        """Return nearest point (x,y,z) in map frame or None if unavailable or too far."""
        with self.lock:
            lidar_msg = self.latest_lidar_msg
        if lidar_msg is None:
            return None

        # We need to have a point cloud helper
        if pc2 is None:
            self.get_logger().warn("sensor_msgs_py.point_cloud2 not available; skipping LiDAR fusion.")
            return None

        try:
            # Step 1: transform map_pos into lidar_frame coordinates
            # Get transform from lidar_frame <- map_frame (we want map -> lidar)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.lidar_frame,
                    self.map_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.tf_timeout)
                )
                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z
                q = transform.transform.rotation
                quat = [q.x, q.y, q.z, q.w]
                rot_matrix = tf_transformations.quaternion_matrix(quat)[:3, :3]
            except Exception as e:
                # if transform fails, try using identity (skip fusion)
                self.get_logger().warn(f"Failed to lookup transform map->lidar: {e}")
                return None

            # map_pos vector
            pos_map = np.array([map_pos['x'], map_pos['y'], map_pos['z']])
            # pos in lidar frame: rot_matrix * (pos_map - translation_map_to_lidar)
            # Because transform is lidar_frame <- map_frame, the transform gives: p_lidar = R * p_map + t
            p_map = pos_map
            p_lidar = np.dot(rot_matrix, p_map) + np.array([tx, ty, tz])

            # Step 2: iterate over points in lidar_msg (in lidar_frame) and find nearest within radius
            nearest = None
            nearest_dist2 = float('inf')
            # Use read_points with skip_nans=True for safety
            for p in pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True):
                # p is tuple (x,y,z)
                dx = p[0] - p_lidar[0]
                dy = p[1] - p_lidar[1]
                dz = p[2] - p_lidar[2]
                dist2 = dx*dx + dy*dy + dz*dz
                if dist2 < nearest_dist2:
                    nearest_dist2 = dist2
                    nearest = p

            if nearest is None:
                return None

            nearest_dist = np.sqrt(nearest_dist2)
            if nearest_dist <= self.lidar_search_radius:
                # Convert nearest point back to map frame for fusion: p_map = R_inv*(p_lidar - t)
                # R_inv = R.T because R is rotation matrix
                R_inv = rot_matrix.T
                p_lidar_arr = np.array([nearest['x'], nearest['y'],nearest['z']], dtype=np.float64)
                p_map_from_lidar = np.dot(R_inv, (p_lidar_arr - np.array([tx, ty, tz])))
                return {'x': float(p_map_from_lidar[0]), 'y': float(p_map_from_lidar[1]), 'z': float(p_map_from_lidar[2]), 'dist': float(nearest_dist)}
            else:
                return None
        except Exception as e:
            self.get_logger().warn(f"Error while searching LiDAR points: {e}")
            return None

    # fuse camera-derived map_position with lidar nearest point (both in map frame)
    def fuse_with_lidar(self, map_pos):
        lidar_point = self.find_nearest_lidar_point(map_pos)
        if lidar_point is None:
            return map_pos, 0.0  # no lidar support, zero lidar confidence
        # Weighted average: fused = (1 - w)*camera + w*lidar_point
        w = float(self.lidar_influence)
        fused = {
            'x': (1.0 - w) * map_pos['x'] + w * lidar_point['x'],
            'y': (1.0 - w) * map_pos['y'] + w * lidar_point['y'],
            'z': (1.0 - w) * map_pos['z'] + w * lidar_point['z']
        }
        # Provide a simple confidence based on distance between camera and lidar (smaller dist -> higher confidence)
        dist = np.linalg.norm([map_pos['x'] - lidar_point['x'], map_pos['y'] - lidar_point['y'], map_pos['z'] - lidar_point['z']])
        lidar_conf = max(0.0, 1.0 - (dist / max(self.lidar_search_radius, 1e-6)))
        return fused, float(lidar_conf)

    def process_detections(self):
        with self.lock:
            current_images = self.latest_images.copy()
        marker_positions = {}  # id: list of (x,y,z, confidence)

        for camera_name, image_data in current_images.items():
            if camera_name not in self.camera_info:
                continue
            image = image_data['image']
            frame_id = self.camera_info[camera_name]['frame_id']
            aruco_detections = self.detect_aruco(image)

            for detection in aruco_detections:
                position_3d = self.estimate_3d_position(detection['corners'], camera_name)
                if position_3d is None:
                    continue

                map_position = self.transform_to_map_frame(position_3d, frame_id)
                if map_position is None:
                    continue

                fused_map_pos, lidar_conf = self.fuse_with_lidar(map_position)
                marker_id = detection['id']
                confidence = 1.0 + lidar_conf

                if marker_id not in marker_positions:
                    marker_positions[marker_id] = []
                marker_positions[marker_id].append((fused_map_pos['x'], fused_map_pos['y'], fused_map_pos['z'], confidence))

        # Pick **only one** detection per ID based on highest confidence
        for marker_id, pos_list in marker_positions.items():
            # Sort by confidence, pick best
            best_detection = max(pos_list, key=lambda p: p[3])
            x, y, z, conf = best_detection

            if marker_id not in self.marker_buffers:
                self.marker_buffers[marker_id] = deque(maxlen=self.buffer_size)
            self.marker_buffers[marker_id].append((x, y, z))

        self.publish_mean_marker_positions()

    def publish_mean_marker_positions(self):
        msg = Float32MultiArray()
        data = []
        for marker_id, buffer in self.marker_buffers.items():
            if not buffer:
                continue
            xs, ys, zs = zip(*buffer)
            mean_x = float(np.mean(xs))
            mean_y = float(np.mean(ys))
            mean_z = float(np.mean(zs))
            # confidence is simple for now - 1.0 (camera) + extra if LiDAR confirmed recently (optional)
            confidence = 1.0
            data.extend([
                float(marker_id),
                mean_x,
                mean_y,
                mean_z
            ])
        if data:
            msg.data = data
            self.marker_pub.publish(msg)
            self.get_logger().info(f"Published {len(data)//4} unique marker(s) mean positions.")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # cv2.destroyAllWindows()  # If using debug visualization

if __name__ == '__main__':
    main()
