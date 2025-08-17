#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Header
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
import numpy as np
import tf_transformations
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from scipy.spatial.distance import cdist
import tf2_ros
import tf2_geometry_msgs
import time
import math


class MapFriendlyBoundaryNode(Node):
    def __init__(self):
        super().__init__('map_friendly_boundary')

        # Parameters
        self.declare_parameter('grid_size', 2000)  # Larger grid for map-like behavior
        self.declare_parameter('resolution', 0.05)  # Finer resolution like typical maps
        self.declare_parameter('origin_x', -50.0)  # Larger coverage area
        self.declare_parameter('origin_y', -50.0)
        self.declare_parameter('min_connection_distance', 0.8)  # 80cm minimum
        self.declare_parameter('max_connection_distance', 10.0)  # 10m maximum
        self.declare_parameter('marker_topic', '/detected_aruco_markers')
        self.declare_parameter('robot_safety_distance', 0.75)  # Minimum distance from robot
        self.declare_parameter('robot_pose_topic', '/robot_pose')
        self.declare_parameter('use_amcl_pose', True)
        self.declare_parameter('line_thickness', 0.15)  # Smaller thickness of boundary lines in meters
        self.declare_parameter('marker_buffer_radius', 0.2)  # Small circle around each marker
        self.declare_parameter('map_bounds_x', 20.0)  # Known/mapped area bounds
        self.declare_parameter('map_bounds_y', 20.0)
        self.declare_parameter('update_rate', 2.0)  # Hz for publishing updates
        # Map correction parameters
        self.declare_parameter('enable_map_correction', True)  # Enable map-based correction
        self.declare_parameter('obstacle_search_radius', 1.0)  # Search radius for nearby obstacles
        self.declare_parameter('obstacle_threshold', 90)  # Minimum occupancy value to consider obstacle
        self.declare_parameter('correction_tolerance', 0.5)  # Max distance to shift boundaries
        # Marker persistence parameters
        self.declare_parameter('marker_timeout', 30.0)  # Time to keep markers without updates (seconds)
        self.declare_parameter('max_acute_angle', 60.0)  # Maximum acute angle between lines (degrees)
        self.declare_parameter('min_line_length', 0.5)  # Minimum line length in meters
        
        # Get parameters
        self.grid_size = self.get_parameter('grid_size').get_parameter_value().integer_value
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.origin_x = self.get_parameter('origin_x').get_parameter_value().double_value
        self.origin_y = self.get_parameter('origin_y').get_parameter_value().double_value
        self.min_connection_distance = self.get_parameter('min_connection_distance').get_parameter_value().double_value
        self.max_connection_distance = self.get_parameter('max_connection_distance').get_parameter_value().double_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.robot_safety_distance = self.get_parameter('robot_safety_distance').get_parameter_value().double_value
        self.robot_pose_topic = self.get_parameter('robot_pose_topic').get_parameter_value().string_value
        self.use_amcl_pose = self.get_parameter('use_amcl_pose').get_parameter_value().bool_value
        self.line_thickness = self.get_parameter('line_thickness').get_parameter_value().double_value
        self.marker_buffer_radius = self.get_parameter('marker_buffer_radius').get_parameter_value().double_value
        self.map_bounds_x = self.get_parameter('map_bounds_x').get_parameter_value().double_value
        self.map_bounds_y = self.get_parameter('map_bounds_y').get_parameter_value().double_value
        self.update_rate = self.get_parameter('update_rate').get_parameter_value().double_value
        # Map correction parameters
        self.enable_map_correction = self.get_parameter('enable_map_correction').get_parameter_value().bool_value
        self.obstacle_search_radius = self.get_parameter('obstacle_search_radius').get_parameter_value().double_value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').get_parameter_value().integer_value
        self.correction_tolerance = self.get_parameter('correction_tolerance').get_parameter_value().double_value
        # Marker persistence parameters
        self.marker_timeout = self.get_parameter('marker_timeout').get_parameter_value().double_value
        self.max_acute_angle = self.get_parameter('max_acute_angle').get_parameter_value().double_value
        self.min_line_length = self.get_parameter('min_line_length').get_parameter_value().double_value

        # QoS Profile for map-like behavior
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Publishers - changed topic name to be more map-like
        self.map_pub = self.create_publisher(OccupancyGrid, '/boundary_map', map_qos)

        # Subscribers
        self.marker_sub = self.create_subscription(
            Float32MultiArray,
            self.marker_topic,
            self.marker_callback,
            10
        )

        # Optional: Subscribe to existing map to merge boundaries
        self.base_map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.base_map_callback,
            map_qos
        )

        # Robot pose subscription
        if self.use_amcl_pose:
            self.robot_pose_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                '/amcl_pose',
                self.robot_pose_callback,
                10
            )
        else:
            self.robot_pose_sub = self.create_subscription(
                PoseStamped,
                self.robot_pose_topic,
                self.robot_pose_stamped_callback,
                10
            )

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_pose_received = False
        self.base_map = None
        self.base_map_info = None
        
        # Marker persistence storage - Dictionary with ID as key
        # Each entry: {'id': int, 'x': float, 'y': float, 'last_seen': float, 'corrected_x': float, 'corrected_y': float}
        self.persistent_markers = {}
        
        # Initialize map with unknown values (-1)
        self.map_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # Timer for periodic publishing and cleanup
        self.create_timer(1.0 / self.update_rate, self.publish_map)
        self.create_timer(5.0, self.cleanup_expired_markers)  # Cleanup every 5 seconds

        self.get_logger().info("Enhanced Map-Friendly Boundary Node initialized.")
        self.get_logger().info(f"Publishing boundary map at {self.update_rate} Hz on /boundary_map")
        self.get_logger().info(f"Marker timeout: {self.marker_timeout}s, Max angle: {self.max_acute_angle}°")

    def base_map_callback(self, msg):
        """Store base map to merge with boundaries"""
        self.base_map = msg
        self.base_map_info = msg.info
        
        # Apply map correction to current markers if enabled
        if self.enable_map_correction and self.persistent_markers:
            self.apply_map_corrections_to_persistent_markers()
            self.get_logger().debug(f"Applied map corrections to {len(self.persistent_markers)} persistent markers")
        
        self.get_logger().debug("Updated base map for merging")

    def robot_pose_callback(self, msg):
        """Handle AMCL pose updates"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_pose_received = True

    def robot_pose_stamped_callback(self, msg):
        """Handle PoseStamped updates"""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.robot_pose_received = True

    def get_robot_pose_from_tf(self):
        """Fallback: get robot pose from TF if direct pose not available"""
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
            self.robot_pose_received = True
            return True
        except Exception as e:
            self.get_logger().warn(f"Could not get robot pose from TF: {e}")
            return False

    def marker_callback(self, msg):
        """Update markers and store them persistently with IDs"""
        if not msg.data:  # Skip empty messages
            return

        # Try to get robot pose if not received
        if not self.robot_pose_received:
            self.get_robot_pose_from_tf()

        current_time = time.time()
        
        # Parse markers - assuming format: [id, x, y, angle, id, x, y, angle, ...]
        data = msg.data
        new_marker_ids = set()

        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                marker_id = int(data[i])
                x = float(data[i+1])
                y = float(data[i+2])
                # angle = float(data[i+3])  # Not used currently but available
                
                new_marker_ids.add(marker_id)
                
                # Update or add marker
                if marker_id in self.persistent_markers:
                    # Update existing marker
                    self.persistent_markers[marker_id].update({
                        'x': x,
                        'y': y,
                        'last_seen': current_time
                    })
                else:
                    # Add new marker
                    self.persistent_markers[marker_id] = {
                        'id': marker_id,
                        'x': x,
                        'y': y,
                        'last_seen': current_time,
                        'corrected_x': x,
                        'corrected_y': y
                    }
        
        # Apply map corrections if enabled
        if self.enable_map_correction and self.base_map is not None:
            self.apply_map_corrections_to_persistent_markers()
            
        self.get_logger().debug(f"Updated {len(new_marker_ids)} markers. Total persistent: {len(self.persistent_markers)}")

    def cleanup_expired_markers(self):
        """Remove markers that haven't been seen for too long"""
        current_time = time.time()
        expired_ids = []
        
        for marker_id, marker_data in self.persistent_markers.items():
            if current_time - marker_data['last_seen'] > self.marker_timeout:
                expired_ids.append(marker_id)
        
        for marker_id in expired_ids:
            del self.persistent_markers[marker_id]
            
        if expired_ids:
            self.get_logger().info(f"Removed {len(expired_ids)} expired markers: {expired_ids}")

    def apply_map_corrections_to_persistent_markers(self):
        """Apply map corrections to all persistent markers"""
        if not self.enable_map_correction or not self.base_map:
            return
            
        for marker_id, marker_data in self.persistent_markers.items():
            corrected_pos = self.find_nearby_obstacle(marker_data['x'], marker_data['y'])
            
            if corrected_pos:
                original_dist = np.sqrt((marker_data['x'] - corrected_pos[0])**2 + 
                                      (marker_data['y'] - corrected_pos[1])**2)
                                      
                if original_dist <= self.correction_tolerance:
                    marker_data['corrected_x'] = corrected_pos[0]
                    marker_data['corrected_y'] = corrected_pos[1]
                else:
                    # Keep original if correction is too large
                    marker_data['corrected_x'] = marker_data['x']
                    marker_data['corrected_y'] = marker_data['y']
            else:
                # No correction found, keep original
                marker_data['corrected_x'] = marker_data['x']
                marker_data['corrected_y'] = marker_data['y']

    def generate_map_with_boundaries(self):
        """Generate map-like occupancy grid with boundaries and unknown areas"""
        # Start with unknown map (-1 everywhere)
        map_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # If we have a base map, merge it first
        if self.base_map is not None:
            map_grid = self.merge_with_base_map(map_grid)
        else:
            # Define known area around robot and markers as free space (0)
            self.mark_known_areas_as_free(map_grid)
        
        # Add boundary obstacles if we have markers
        if len(self.persistent_markers) >= 1:
            self.add_boundaries_to_map(map_grid)
            
        return map_grid

    def merge_with_base_map(self, map_grid):
        """Merge with existing SLAM map if available"""
        try:
            base_data = np.array(self.base_map.data).reshape(
                self.base_map_info.height, self.base_map_info.width
            )
            
            # Transform base map coordinates to our grid coordinates
            for y in range(self.base_map_info.height):
                for x in range(self.base_map_info.width):
                    # Convert base map cell to world coordinates
                    world_x = self.base_map_info.origin.position.x + x * self.base_map_info.resolution
                    world_y = self.base_map_info.origin.position.y + y * self.base_map_info.resolution
                    
                    # Convert to our grid coordinates
                    our_x = int((world_x - self.origin_x) / self.resolution)
                    our_y = int((world_y - self.origin_y) / self.resolution)
                    
                    # Copy if within bounds
                    if 0 <= our_x < self.grid_size and 0 <= our_y < self.grid_size:
                        map_grid[our_y, our_x] = base_data[y, x]
                        
        except Exception as e:
            self.get_logger().warn(f"Error merging base map: {e}")
            
        return map_grid

    def mark_known_areas_as_free(self, map_grid):
        """Mark areas around robot and markers as known free space"""
        # Mark area around robot as free
        if self.robot_pose_received:
            robot_range = 3.0  # 3 meter radius around robot
            self.mark_area_as_free(map_grid, self.robot_x, self.robot_y, robot_range)
        
        # Mark areas around markers as free
        for marker_data in self.persistent_markers.values():
            mx, my = marker_data['corrected_x'], marker_data['corrected_y']
            marker_range = 2.0  # 2 meter radius around markers
            self.mark_area_as_free(map_grid, mx, my, marker_range)
            
        # Connect areas between robot and markers
        if self.robot_pose_received:
            for marker_data in self.persistent_markers.values():
                mx, my = marker_data['corrected_x'], marker_data['corrected_y']
                self.mark_path_as_free(map_grid, self.robot_x, self.robot_y, mx, my)

    def mark_area_as_free(self, map_grid, center_x, center_y, radius):
        """Mark circular area as free space (0)"""
        center_grid_x = int((center_x - self.origin_x) / self.resolution)
        center_grid_y = int((center_y - self.origin_y) / self.resolution)
        radius_cells = int(radius / self.resolution)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    x = center_grid_x + dx
                    y = center_grid_y + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        # Only mark as free if it was unknown, don't override obstacles
                        if map_grid[y, x] == -1:
                            map_grid[y, x] = 0

    def mark_path_as_free(self, map_grid, x1, y1, x2, y2):
        """Mark path between two points as free"""
        # Convert to grid coordinates
        gx1 = int((x1 - self.origin_x) / self.resolution)
        gy1 = int((y1 - self.origin_y) / self.resolution)
        gx2 = int((x2 - self.origin_x) / self.resolution)
        gy2 = int((y2 - self.origin_y) / self.resolution)
        
        # Use Bresenham's line algorithm
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        x, y = gx1, gy1
        x_inc = 1 if gx1 < gx2 else -1
        y_inc = 1 if gy1 < gy2 else -1
        error = dx - dy
        
        path_width = 3  # cells
        
        while True:
            # Mark area around path point as free
            for i in range(-path_width, path_width + 1):
                for j in range(-path_width, path_width + 1):
                    px, py = x + i, y + j
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                        if map_grid[py, px] == -1:  # Only mark unknown areas
                            map_grid[py, px] = 0
            
            if x == gx2 and y == gy2:
                break
                
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc

    def add_boundaries_to_map(self, map_grid):
        """Add boundary obstacles to the map using persistent markers"""
        if len(self.persistent_markers) < 1:
            return

        # Get corrected marker positions
        safe_markers = self.filter_markers_by_robot_distance()
        
        if len(safe_markers) < 1:
            return
            
        # Draw small obstacles around markers
        for marker_data in safe_markers:
            mx, my = marker_data['corrected_x'], marker_data['corrected_y']
            self.draw_obstacle_circle(map_grid, mx, my, self.marker_buffer_radius)
        
        if len(safe_markers) < 2:
            return
            
        # Create optimal connections between markers
        connections = self.find_smart_connections(safe_markers)
        
        # Draw boundary lines as obstacles
        for connection in connections:
            start_marker, end_marker = connection
            start_x, start_y = start_marker['corrected_x'], start_marker['corrected_y']
            end_x, end_y = end_marker['corrected_x'], end_marker['corrected_y']
            
            if self.is_connection_safe(start_x, start_y, end_x, end_y):
                # Validate boundary with map if correction is enabled
                if (not self.enable_map_correction or 
                    self.validate_boundary_with_map((start_x, start_y), (end_x, end_y))):
                    self.draw_obstacle_line(map_grid, start_x, start_y, end_x, end_y, self.line_thickness)
                else:
                    self.get_logger().debug(f"Skipped boundary line - no corresponding obstacles in map")

    def find_smart_connections(self, markers):
        """Find optimal connections with angle constraints and max 2 connections per marker"""
        if len(markers) < 2:
            return []

        connections = []
        marker_connections = {marker['id']: [] for marker in markers}
        
        # Calculate all possible connections with distances
        possible_connections = []
        for i, marker1 in enumerate(markers):
            for j, marker2 in enumerate(markers):
                if i >= j:  # Avoid duplicates and self-connections
                    continue
                    
                x1, y1 = marker1['corrected_x'], marker1['corrected_y']
                x2, y2 = marker2['corrected_x'], marker2['corrected_y']
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Check distance constraints and minimum line length
                if (self.min_connection_distance <= distance <= self.max_connection_distance and 
                    distance >= self.min_line_length):
                    possible_connections.append({
                        'marker1': marker1,
                        'marker2': marker2,
                        'distance': distance,
                        'priority': 1.0 / distance  # Prefer shorter connections
                    })
        
        # Sort by priority (shorter distances first)
        possible_connections.sort(key=lambda x: x['priority'], reverse=True)
        
        # Greedily add connections while respecting constraints
        for conn in possible_connections:
            marker1 = conn['marker1']
            marker2 = conn['marker2']
            
            # Check if either marker already has 2 connections
            if (len(marker_connections[marker1['id']]) >= 2 or 
                len(marker_connections[marker2['id']]) >= 2):
                continue
            
            # Check angle constraint if marker already has one connection
            if len(marker_connections[marker1['id']]) == 1:
                if not self.check_angle_constraint(marker1, marker2, marker_connections[marker1['id']][0]):
                    continue
                    
            if len(marker_connections[marker2['id']]) == 1:
                if not self.check_angle_constraint(marker2, marker1, marker_connections[marker2['id']][0]):
                    continue
            
            # Add the connection
            connections.append((marker1, marker2))
            marker_connections[marker1['id']].append(marker2)
            marker_connections[marker2['id']].append(marker1)
        
        self.get_logger().debug(f"Created {len(connections)} smart boundary connections")
        return connections

    def check_angle_constraint(self, center_marker, new_marker, existing_marker):
        """Check if adding new_marker connection to center_marker violates angle constraint"""
        # Calculate vectors from center to both markers
        cx, cy = center_marker['corrected_x'], center_marker['corrected_y']
        
        # Vector to existing connection
        ex, ey = existing_marker['corrected_x'], existing_marker['corrected_y']
        vec1 = np.array([ex - cx, ey - cy])
        
        # Vector to potential new connection
        nx, ny = new_marker['corrected_x'], new_marker['corrected_y']
        vec2 = np.array([nx - cx, ny - cy])
        
        # Calculate angle between vectors
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return False
            
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(abs(cos_angle)))
        
        # Check if angle is acute and within limit
        return angle_deg <= self.max_acute_angle

    def filter_markers_by_robot_distance(self):
        """Filter out markers that are too close to the robot"""
        if not self.robot_pose_received:
            self.get_logger().warn("Robot pose not received, using all markers")
            return list(self.persistent_markers.values())

        safe_markers = []
        for marker_data in self.persistent_markers.values():
            if self.is_marker_safe(marker_data):
                safe_markers.append(marker_data)

        return safe_markers

    def is_marker_safe(self, marker_data):
        """Check if a marker is safe (far enough from robot)"""
        if not self.robot_pose_received:
            return True
            
        robot_pos = np.array([self.robot_x, self.robot_y])
        marker_pos = np.array([marker_data['corrected_x'], marker_data['corrected_y']])
        distance = np.linalg.norm(robot_pos - marker_pos)
        
        return distance >= self.robot_safety_distance

    def is_connection_safe(self, x1, y1, x2, y2):
        """Check if connecting two points would interfere with robot position"""
        if not self.robot_pose_received:
            return True

        # Calculate distance from robot to line segment
        robot_pos = np.array([self.robot_x, self.robot_y])
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        
        # Distance from point to line segment
        line_vec = p2 - p1
        point_vec = robot_pos - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return True
            
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        if proj_length < 0:
            closest_point = p1
        elif proj_length > line_len:
            closest_point = p2
        else:
            closest_point = p1 + proj_length * line_unitvec
            
        distance_to_line = np.linalg.norm(robot_pos - closest_point)
        
        return distance_to_line >= self.robot_safety_distance

    def find_nearby_obstacle(self, world_x, world_y):
        """Find nearest obstacle in base map within search radius"""
        if not self.base_map:
            return None
            
        # Convert world coordinates to base map grid coordinates
        map_x = int((world_x - self.base_map_info.origin.position.x) / self.base_map_info.resolution)
        map_y = int((world_y - self.base_map_info.origin.position.y) / self.base_map_info.resolution)
        
        # Check if position is within base map bounds
        if (map_x < 0 or map_x >= self.base_map_info.width or 
            map_y < 0 or map_y >= self.base_map_info.height):
            return None
            
        # Convert base map data to numpy array
        map_data = np.array(self.base_map.data).reshape(
            self.base_map_info.height, self.base_map_info.width
        )
        
        # Search radius in map cells
        search_radius_cells = int(self.obstacle_search_radius / self.base_map_info.resolution)
        
        best_obstacle_pos = None
        min_distance = float('inf')
        
        # Search in expanding squares around the marker position
        for radius in range(1, search_radius_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check cells on the perimeter of current radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                        
                    check_x = map_x + dx
                    check_y = map_y + dy
                    
                    # Check bounds
                    if (check_x < 0 or check_x >= self.base_map_info.width or 
                        check_y < 0 or check_y >= self.base_map_info.height):
                        continue
                    
                    # Check if this cell is an obstacle
                    if map_data[check_y, check_x] >= self.obstacle_threshold:
                        # Convert back to world coordinates
                        obstacle_world_x = (self.base_map_info.origin.position.x + 
                                          check_x * self.base_map_info.resolution)
                        obstacle_world_y = (self.base_map_info.origin.position.y + 
                                          check_y * self.base_map_info.resolution)
                        
                        # Calculate distance to original marker position
                        distance = np.sqrt((world_x - obstacle_world_x)**2 + 
                                         (world_y - obstacle_world_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_obstacle_pos = (obstacle_world_x, obstacle_world_y)
            
            # If we found an obstacle in this radius, use it (closest obstacle wins)
            if best_obstacle_pos is not None:
                break
                
        return best_obstacle_pos

    def validate_boundary_with_map(self, start_pos, end_pos):
        """Validate if boundary line corresponds to actual obstacles in map"""
        if not self.base_map or not self.enable_map_correction:
            return True
            
        # Sample points along the line and check if they correspond to obstacles
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Number of sample points along the line
        num_samples = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) / self.base_map_info.resolution)
        num_samples = max(5, min(50, num_samples))  # Limit sample count
        
        obstacle_matches = 0
        total_samples = 0
        
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y)
            
            # Check if this point corresponds to an obstacle in the map
            if self.is_obstacle_at_position(sample_x, sample_y):
                obstacle_matches += 1
            total_samples += 1
            
        # Consider line valid if at least 60% of samples correspond to obstacles
        match_ratio = obstacle_matches / total_samples if total_samples > 0 else 0
        return match_ratio >= 0.6

    def is_obstacle_at_position(self, world_x, world_y):
        """Check if given world position corresponds to obstacle in base map"""
        if not self.base_map:
            return False
            
        # Convert to map coordinates
        map_x = int((world_x - self.base_map_info.origin.position.x) / self.base_map_info.resolution)
        map_y = int((world_y - self.base_map_info.origin.position.y) / self.base_map_info.resolution)
        
        # Check bounds
        if (map_x < 0 or map_x >= self.base_map_info.width or 
            map_y < 0 or map_y >= self.base_map_info.height):
            return False
            
        # Get map data
        map_data = np.array(self.base_map.data).reshape(
            self.base_map_info.height, self.base_map_info.width
        )
        
        return map_data[map_y, map_x] >= self.obstacle_threshold

    def draw_obstacle_circle(self, map_grid, cx, cy, radius):
        """Draw obstacle circle (value 100)"""
        center_x = int((cx - self.origin_x) / self.resolution)
        center_y = int((cy - self.origin_y) / self.resolution)
        rad_cells = int(radius / self.resolution)

        for dx in range(-rad_cells, rad_cells + 1):
            for dy in range(-rad_cells, rad_cells + 1):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if dx**2 + dy**2 <= rad_cells**2:
                        map_grid[y, x] = 100  # Occupied

    def draw_obstacle_line(self, map_grid, x1, y1, x2, y2, thickness):
        """Draw obstacle line between two points"""
        # Convert to grid coordinates
        gx1 = int((x1 - self.origin_x) / self.resolution)
        gy1 = int((y1 - self.origin_y) / self.resolution)
        gx2 = int((x2 - self.origin_x) / self.resolution)
        gy2 = int((y2 - self.origin_y) / self.resolution)
        
        # Calculate line thickness in grid cells
        thickness_cells = max(1, int(thickness / self.resolution))
        
        # Use Bresenham-like algorithm to draw thick line
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        x, y = gx1, gy1
        x_inc = 1 if gx1 < gx2 else -1
        y_inc = 1 if gy1 < gy2 else -1
        error = dx - dy

        while True:
            # Draw thick point
            for i in range(-thickness_cells, thickness_cells + 1):
                for j in range(-thickness_cells, thickness_cells + 1):
                    px, py = x + i, y + j
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                        map_grid[py, px] = 100  # Occupied

            if x == gx2 and y == gy2:
                break
                
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc

    def publish_map(self):
        """Periodically publish the map"""
        map_grid = self.generate_map_with_boundaries()
        occupancy_grid = self.grid_to_occupancy(map_grid)
        self.map_pub.publish(occupancy_grid)
        
        # Log marker status periodically
        if len(self.persistent_markers) > 0:
            active_markers = sum(1 for m in self.persistent_markers.values() 
                               if time.time() - m['last_seen'] < 10.0)
            self.get_logger().debug(f"Markers: {len(self.persistent_markers)} total, {active_markers} active")

    def grid_to_occupancy(self, grid):
        """Convert grid to OccupancyGrid message"""
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size

        msg.info.origin = Pose()
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        quat = tf_transformations.quaternion_from_euler(0, 0, 0)
        msg.info.origin.orientation.x = quat[0]
        msg.info.origin.orientation.y = quat[1]
        msg.info.origin.orientation.z = quat[2]
        msg.info.origin.orientation.w = quat[3]

        flat_grid = grid.flatten()
        msg.data = list(flat_grid)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = MapFriendlyBoundaryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()