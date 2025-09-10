#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import carla
import numpy as np
import random
import math
import time
from collections import defaultdict, deque

# ROS2 Messages
from geometry_msgs.msg import PoseStamped, Twist, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2, Image, LaserScan
from std_msgs.msg import Header, String, Bool
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus, CarlaEgoVehicleInfo

import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
import tf_transformations

class RoadNetworkGlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_plan_node')
        
        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # === ROS PUBLISHERS === (Output via ROS)
        self.path_pub = self.create_publisher(Path, '/planned_path', path_qos)
        self.occupancy_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', sensor_qos)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', path_qos)
        self.status_pub = self.create_publisher(String, '/planner_status', 10)
        self.vehicle_info_pub = self.create_publisher(CarlaEgoVehicleInfo, '/carla/ego_vehicle/vehicle_info', 10)
        
        # === ROS SUBSCRIBERS === (Input via ROS)
        self.vehicle_status_sub = self.create_subscription(
            CarlaEgoVehicleStatus, '/carla/ego_vehicle/vehicle_status',
            self.vehicle_status_callback, sensor_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, sensor_qos)
        self.goal_reached_sub = self.create_subscription(
            Bool, '/goal_reached', self.goal_reached_callback, 10)
        
        # === TF2 for coordinate transformations ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        self.town_map = None
        self.blueprint_library = None
        
        # ROS Communication State
        self.current_odom = None
        self.vehicle_status = None
        self.goal_reached = False
        
        # ROAD NETWORK planning variables - STRICTLY following roads
        self.validated_start_transform = None
        self.validated_goal_transform = None
        self.road_network_path = None  # Path strictly following CARLA roads
        
        # Vehicle state
        self.current_position = None
        self.vehicle_spawned = False
        
        # Path publishing control
        self.path_published_count = 0
        self.max_path_publications = 5
        
        # Road network parameters (same as original)
        self.min_distance_between_points = 80.0
        self.max_distance_between_points = 250.0
        self.waypoint_spacing = 5.0
        
        # Timer for main loop
        self.timer = self.create_timer(2.0, self.monitor_loop)
        
        # Path publishing timer
        self.path_timer = self.create_timer(1.0, self.publish_path_repeatedly)
        
        # Connect to CARLA
        self.connect_to_carla()
        
        self.get_logger().info("🌐 Road Network Global Planner with ROS Communication initialized")
        self.publish_status("INITIALIZING")
        
    def publish_status(self, status):
        """Publish planner status via ROS"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        
    def vehicle_status_callback(self, msg):
        """Receive vehicle status via ROS"""
        self.vehicle_status = msg
        
    def odom_callback(self, msg):
        """Receive odometry via ROS"""
        self.current_odom = msg
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
    def goal_reached_callback(self, msg):
        """Receive goal reached notification via ROS"""
        if msg.data:
            self.goal_reached = True
            self.publish_status("GOAL_REACHED")
            self.get_logger().info("🎯 Goal reached notification received via ROS")
    
    def connect_to_carla(self):
        """Connect to CARLA and setup"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.town_map = self.world.get_map()
            self.blueprint_library = self.world.get_blueprint_library()
            
            self.get_logger().info("🔗 Connected to CARLA simulator")
            self.publish_status("CONNECTED_TO_CARLA")
            
            # Spawn vehicle with ROAD NETWORK validated points
            self.spawn_vehicle_with_road_network_validation()
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to connect to CARLA: {e}")
            self.publish_status("CARLA_CONNECTION_FAILED")
    
    def is_spawn_point_valid_on_road(self, spawn_point):
        """Validate spawn point using CARLA's road network - same as original"""
        try:
            waypoint = self.town_map.get_waypoint(
                spawn_point.location, 
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            if not waypoint:
                return False
            
            if waypoint.lane_type != carla.LaneType.Driving:
                return False
            
            try:
                next_waypoints = waypoint.next(10.0)
                prev_waypoints = waypoint.previous(10.0)
                
                has_good_connectivity = len(next_waypoints) > 0 and len(prev_waypoints) > 0
                
                distance_to_road = math.sqrt(
                    (spawn_point.location.x - waypoint.transform.location.x)**2 + 
                    (spawn_point.location.y - waypoint.transform.location.y)**2
                )
                
                is_on_road = distance_to_road < 2.0
                
                return has_good_connectivity and is_on_road
                
            except Exception:
                return False
            
        except Exception as e:
            self.get_logger().debug(f"Spawn point validation error: {e}")
            return False
    
    def get_road_network_spawn_points(self):
        """Get validated spawn points - same logic as original"""
        spawn_points = self.world.get_map().get_spawn_points()
        
        if len(spawn_points) < 2:
            self.get_logger().error("Not enough spawn points available in this map")
            return None, None
        
        current_seed = int(time.time()) % 10000
        random.seed(current_seed)
        self.get_logger().info(f"🎲 Using road network seed: {current_seed}")
        
        valid_spawn_points = []
        for spawn_point in spawn_points:
            if self.is_spawn_point_valid_on_road(spawn_point):
                valid_spawn_points.append(spawn_point)
        
        if len(valid_spawn_points) < 2:
            self.get_logger().warn("Not enough valid spawn points, using all available")
            valid_spawn_points = spawn_points
        
        valid_pairs = []
        for i, start_point in enumerate(valid_spawn_points):
            for j, goal_point in enumerate(valid_spawn_points):
                if i == j:
                    continue
                    
                distance = math.sqrt(
                    (goal_point.location.x - start_point.location.x)**2 + 
                    (goal_point.location.y - start_point.location.y)**2
                )
                
                if self.min_distance_between_points <= distance <= self.max_distance_between_points:
                    if self.are_points_well_connected_via_roads(start_point, goal_point):
                        valid_pairs.append((start_point, goal_point, distance))
        
        if not valid_pairs:
            self.get_logger().warn("No well-connected pairs found, using distance-based fallback")
            for i, start_point in enumerate(valid_spawn_points):
                for j, goal_point in enumerate(valid_spawn_points):
                    if i == j:
                        continue
                    distance = math.sqrt(
                        (goal_point.location.x - start_point.location.x)**2 + 
                        (goal_point.location.y - start_point.location.y)**2
                    )
                    if 50.0 <= distance <= 300.0:
                        valid_pairs.append((start_point, goal_point, distance))
        
        if not valid_pairs:
            start_point = random.choice(valid_spawn_points)
            goal_point = random.choice([sp for sp in valid_spawn_points if sp != start_point])
            distance = math.sqrt(
                (goal_point.location.x - start_point.location.x)**2 + 
                (goal_point.location.y - start_point.location.y)**2
            )
            valid_pairs = [(start_point, goal_point, distance)]
        
        chosen_pair = random.choice(valid_pairs)
        start_point, goal_point, distance = chosen_pair
        
        self.get_logger().info(f"✅ ROAD NETWORK validated points:")
        self.get_logger().info(f"   🟢 START: ({start_point.location.x:.1f}, {start_point.location.y:.1f})")
        self.get_logger().info(f"   🎯 GOAL: ({goal_point.location.x:.1f}, {goal_point.location.y:.1f})")
        self.get_logger().info(f"   📏 Distance: {distance:.1f}m")
        
        return start_point, goal_point
    
    def are_points_well_connected_via_roads(self, start_transform, goal_transform):
        """Enhanced connectivity check - same as original"""
        try:
            start_waypoint = self.town_map.get_waypoint(
                start_transform.location, 
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            goal_waypoint = self.town_map.get_waypoint(
                goal_transform.location, 
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            if not start_waypoint or not goal_waypoint:
                return False
            
            if (start_waypoint.lane_type != carla.LaneType.Driving or 
                goal_waypoint.lane_type != carla.LaneType.Driving):
                return False
            
            start_next = start_waypoint.next(15.0)
            start_prev = start_waypoint.previous(15.0)
            goal_next = goal_waypoint.next(15.0)
            goal_prev = goal_waypoint.previous(15.0)
            
            start_well_connected = len(start_next) > 0 and len(start_prev) > 0
            goal_well_connected = len(goal_next) > 0 and len(goal_prev) > 0
            
            if not (start_well_connected and goal_well_connected):
                return False
            
            current_waypoint = start_waypoint
            goal_location = goal_waypoint.transform.location
            
            initial_distance = math.sqrt(
                (current_waypoint.transform.location.x - goal_location.x)**2 + 
                (current_waypoint.transform.location.y - goal_location.y)**2
            )
            
            for _ in range(15):
                next_waypoints = current_waypoint.next(8.0)
                if not next_waypoints:
                    break
                
                best_waypoint = None
                min_distance = float('inf')
                
                for next_wp in next_waypoints:
                    if next_wp.lane_type != carla.LaneType.Driving:
                        continue
                    
                    distance_to_goal = math.sqrt(
                        (next_wp.transform.location.x - goal_location.x)**2 + 
                        (next_wp.transform.location.y - goal_location.y)**2
                    )
                    
                    if distance_to_goal < min_distance:
                        min_distance = distance_to_goal
                        best_waypoint = next_wp
                
                if best_waypoint is None:
                    break
                
                current_waypoint = best_waypoint
                current_distance = min_distance
                progress_made = initial_distance - current_distance
                
                if progress_made > 20.0:
                    return True
                    
                if current_distance < 25.0:
                    return True
            
            final_distance = math.sqrt(
                (current_waypoint.transform.location.x - goal_location.x)**2 + 
                (current_waypoint.transform.location.y - goal_location.y)**2
            )
            
            progress_made = initial_distance - final_distance
            return progress_made > 10.0
            
        except Exception as e:
            self.get_logger().debug(f"Connectivity check error: {e}")
            return False
    
    def spawn_vehicle_with_road_network_validation(self):
        """Spawn vehicle and publish info via ROS"""
        try:
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            vehicle_bp.set_attribute('role_name', 'ego_vehicle')
            
            self.validated_start_transform, self.validated_goal_transform = self.get_road_network_spawn_points()
            
            if not self.validated_start_transform or not self.validated_goal_transform:
                self.get_logger().error("❌ Could not find road network validated spawn points")
                self.publish_status("SPAWN_POINT_VALIDATION_FAILED")
                return
            
            spawn_attempts = 0
            max_spawn_attempts = 5
            
            while spawn_attempts < max_spawn_attempts:
                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, self.validated_start_transform)
                    break
                except Exception as spawn_error:
                    spawn_attempts += 1
                    self.get_logger().warn(f"Spawn attempt {spawn_attempts} failed: {spawn_error}")
                    if spawn_attempts < max_spawn_attempts:
                        time.sleep(1.0)
                    else:
                        self.get_logger().error("❌ Failed to spawn vehicle after multiple attempts")
                        self.publish_status("VEHICLE_SPAWN_FAILED")
                        return
            
            # Publish vehicle info via ROS
            self.publish_vehicle_info()
            
            # Publish goal pose via ROS
            self.publish_goal_pose()
            
            # Setup sensors
            self.setup_basic_sensors()
            
            self.current_position = (self.validated_start_transform.location.x, self.validated_start_transform.location.y)
            
            # Create ROAD NETWORK path
            self.create_road_network_path()
            
            self.vehicle_spawned = True
            self.publish_status("VEHICLE_SPAWNED")
            
            self.get_logger().info(f"✅ Vehicle spawned with ROAD NETWORK validated route")
            
        except Exception as e:
            self.get_logger().error(f"Failed to spawn vehicle: {e}")
            self.publish_status("SPAWN_ERROR")
    
    def publish_vehicle_info(self):
        """Publish vehicle information via ROS"""
        if not self.vehicle:
            return
            
        try:
            vehicle_info = CarlaEgoVehicleInfo()
            vehicle_info.id = self.vehicle.id
            vehicle_info.type = self.vehicle.type_id
            vehicle_info.rolename = 'ego_vehicle'
            
            # Get vehicle attributes
            for attr in self.vehicle.attributes:
                # You can add specific attributes here as needed
                pass
                
            self.vehicle_info_pub.publish(vehicle_info)
            self.get_logger().info("📤 Published vehicle info via ROS")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing vehicle info: {e}")
    
    def publish_goal_pose(self):
        """Publish goal pose via ROS"""
        if not self.validated_goal_transform:
            return
            
        try:
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            
            goal_msg.pose.position.x = self.validated_goal_transform.location.x
            goal_msg.pose.position.y = self.validated_goal_transform.location.y
            goal_msg.pose.position.z = self.validated_goal_transform.location.z
            
            # Convert rotation to quaternion
            yaw = math.radians(self.validated_goal_transform.rotation.yaw)
            pitch = math.radians(self.validated_goal_transform.rotation.pitch)
            roll = math.radians(self.validated_goal_transform.rotation.roll)
            
            quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            goal_msg.pose.orientation.x = quat[0]
            goal_msg.pose.orientation.y = quat[1]
            goal_msg.pose.orientation.z = quat[2]
            goal_msg.pose.orientation.w = quat[3]
            
            self.goal_pub.publish(goal_msg)
            self.get_logger().info("🎯 Published goal pose via ROS")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing goal pose: {e}")
    
    def setup_basic_sensors(self):
        """Setup basic LiDAR sensor - same as original"""
        if not self.vehicle:
            return
            
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')
        
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
    
    def create_road_network_path(self):
        """Create path that STRICTLY follows CARLA's road network - same logic as original"""
        try:
            start_location = self.validated_start_transform.location
            goal_location = self.validated_goal_transform.location
            
            self.get_logger().info(f"Creating ROAD NETWORK path from ({start_location.x:.1f}, {start_location.y:.1f}) to ({goal_location.x:.1f}, {goal_location.y:.1f})")
            
            start_waypoint = self.town_map.get_waypoint(
                start_location, 
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            goal_waypoint = self.town_map.get_waypoint(
                goal_location, 
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            if not start_waypoint or not goal_waypoint:
                self.get_logger().error("Cannot project to road network")
                self.road_network_path = self.create_direct_fallback_path()
                return
            
            path_waypoints = []
            current_waypoint = start_waypoint
            goal_pos = goal_waypoint.transform.location
            
            path_waypoints.append(current_waypoint)
            
            max_waypoints = 200
            visited_road_segments = set()
            stuck_counter = 0
            
            for iteration in range(max_waypoints):
                segment_id = (current_waypoint.road_id, current_waypoint.lane_id, 
                            round(current_waypoint.transform.location.x / 5.0) * 5.0,
                            round(current_waypoint.transform.location.y / 5.0) * 5.0)
                
                if segment_id in visited_road_segments:
                    stuck_counter += 1
                    if stuck_counter > 5:
                        self.get_logger().warn(f"Detected potential loop, breaking at iteration {iteration}")
                        break
                else:
                    stuck_counter = 0
                    visited_road_segments.add(segment_id)
                
                current_pos = current_waypoint.transform.location
                distance_to_goal = math.sqrt(
                    (current_pos.x - goal_pos.x)**2 + 
                    (current_pos.y - goal_pos.y)**2
                )
                
                next_waypoints = current_waypoint.next(self.waypoint_spacing)
                
                if not next_waypoints:
                    self.get_logger().warn(f"No next waypoints at iteration {iteration}")
                    break
                
                if distance_to_goal < 8.0:
                    goal_candidates = []
                    for next_wp in next_waypoints:
                        if next_wp.lane_type != carla.LaneType.Driving:
                            continue
                        wp_pos = next_wp.transform.location
                        dist_to_goal = math.sqrt(
                            (wp_pos.x - goal_pos.x)**2 + 
                            (wp_pos.y - goal_pos.y)**2
                        )
                        if dist_to_goal < distance_to_goal:
                            goal_candidates.append((next_wp, dist_to_goal))
                    
                    if goal_candidates:
                        goal_candidates.sort(key=lambda x: x[1])
                        current_waypoint = goal_candidates[0][0]
                        path_waypoints.append(current_waypoint)
                        
                        if goal_candidates[0][1] < 3.0:
                            self.get_logger().info(f"Reached goal area via road network at iteration {iteration}")
                            break
                        continue
                    else:
                        self.get_logger().info(f"Close to goal but no road connection at iteration {iteration}")
                        break
                
                best_waypoint = None
                min_distance_to_goal = float('inf')
                
                valid_candidates = []
                for next_wp in next_waypoints:
                    if next_wp.lane_type != carla.LaneType.Driving:
                        continue
                    
                    wp_pos = next_wp.transform.location
                    dist_to_goal = math.sqrt(
                        (wp_pos.x - goal_pos.x)**2 + 
                        (wp_pos.y - goal_pos.y)**2
                    )
                    
                    valid_candidates.append((next_wp, dist_to_goal))
                
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[1])
                    best_waypoint = valid_candidates[0][0]
                
                if best_waypoint is None:
                    self.get_logger().warn(f"No valid next waypoint at iteration {iteration}")
                    break
                
                current_waypoint = best_waypoint
                path_waypoints.append(current_waypoint)
                
                if iteration % 20 == 0:
                    self.get_logger().debug(f"Path building progress: {iteration}/{max_waypoints}, distance to goal: {distance_to_goal:.1f}m")
            
            final_waypoint = path_waypoints[-1] if path_waypoints else start_waypoint
            final_pos = final_waypoint.transform.location
            
            final_distance_to_goal = math.sqrt(
                (final_pos.x - goal_pos.x)**2 + 
                (final_pos.y - goal_pos.y)**2
            )
            
            if final_distance_to_goal < 10.0:
                path_waypoints.append(goal_waypoint)
                self.get_logger().info(f"Goal waypoint added via road network (distance: {final_distance_to_goal:.1f}m)")
            else:
                self.get_logger().warn(f"Goal not reachable via road network (distance: {final_distance_to_goal:.1f}m)")
                extended_waypoints = self.extend_path_to_goal(final_waypoint, goal_waypoint, max_extensions=20)
                path_waypoints.extend(extended_waypoints)
                
                if extended_waypoints:
                    final_waypoint = extended_waypoints[-1]
                    final_pos = final_waypoint.transform.location
                    final_distance_to_goal = math.sqrt(
                        (final_pos.x - goal_pos.x)**2 + 
                        (final_pos.y - goal_pos.y)**2
                    )
                    
                    if final_distance_to_goal < 15.0:
                        path_waypoints.append(goal_waypoint)
                        self.get_logger().info(f"Goal waypoint added after extension (distance: {final_distance_to_goal:.1f}m)")
            
            self.road_network_path = []
            for waypoint in path_waypoints:
                point = (waypoint.transform.location.x, waypoint.transform.location.y)
                self.road_network_path.append(point)
            
            if len(self.road_network_path) >= 2:
                total_length = self.calculate_path_length(self.road_network_path)
                self.get_logger().info(f"✅ ROAD NETWORK PATH created: {len(self.road_network_path)} waypoints, {total_length:.1f}m total")
                self.publish_status("PATH_CREATED")
            else:
                self.get_logger().warn("Road network path too short, using fallback")
                self.road_network_path = self.create_direct_fallback_path()
                self.publish_status("FALLBACK_PATH_CREATED")
                
        except Exception as e:
            self.get_logger().error(f"Road network path creation failed: {e}")
            self.road_network_path = self.create_direct_fallback_path()
            self.publish_status("PATH_CREATION_FAILED")
    
    def extend_path_to_goal(self, start_waypoint, goal_waypoint, max_extensions=20):
        """Extend path via road network - same as original"""
        extended_waypoints = []
        current_waypoint = start_waypoint
        goal_pos = goal_waypoint.transform.location
        
        for i in range(max_extensions):
            next_waypoints = current_waypoint.next(self.waypoint_spacing)
            if not next_waypoints:
                break
            
            best_waypoint = None
            min_distance = float('inf')
            current_pos = current_waypoint.transform.location
            current_distance_to_goal = math.sqrt(
                (current_pos.x - goal_pos.x)**2 + 
                (current_pos.y - goal_pos.y)**2
            )
            
            valid_candidates = []
            for next_wp in next_waypoints:
                if next_wp.lane_type != carla.LaneType.Driving:
                    continue
                
                wp_pos = next_wp.transform.location
                distance_to_goal = math.sqrt(
                    (wp_pos.x - goal_pos.x)**2 + 
                    (wp_pos.y - goal_pos.y)**2
                )
                
                if distance_to_goal < current_distance_to_goal:
                    valid_candidates.append((next_wp, distance_to_goal))
            
            if not valid_candidates:
                break
            
            valid_candidates.sort(key=lambda x: x[1])
            best_waypoint = valid_candidates[0][0]
            
            current_waypoint = best_waypoint
            extended_waypoints.append(current_waypoint)
            
            if valid_candidates[0][1] < 5.0:
                break
        
        return extended_waypoints
    
    def create_direct_fallback_path(self):
        """Create direct path as fallback - same as original"""
        start_pos = (self.validated_start_transform.location.x, self.validated_start_transform.location.y)
        goal_pos = (self.validated_goal_transform.location.x, self.validated_goal_transform.location.y)
        
        path = [start_pos]
        
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        num_segments = max(3, int(distance / 12.0))
        for i in range(1, num_segments):
            t = i / num_segments
            intermediate_x = start_pos[0] + t * dx
            intermediate_y = start_pos[1] + t * dy
            path.append((intermediate_x, intermediate_y))
        
        path.append(goal_pos)
        
        self.get_logger().info(f"Fallback direct path created with {len(path)} points")
        return path
    
    def calculate_path_length(self, path):
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def publish_path_repeatedly(self):
        """Publish the ROAD NETWORK path via ROS multiple times for reliability"""
        if not self.road_network_path or self.path_published_count >= self.max_path_publications:
            return
            
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for i, (x, y) in enumerate(self.road_network_path):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            
            # Calculate orientation towards next waypoint
            if i < len(self.road_network_path) - 1:
                dx = self.road_network_path[i + 1][0] - x
                dy = self.road_network_path[i + 1][1] - y
                yaw = math.atan2(dy, dx)
                
                # Convert yaw to quaternion
                pose.pose.orientation.z = math.sin(yaw / 2.0)
                pose.pose.orientation.w = math.cos(yaw / 2.0)
            else:
                pose.pose.orientation.w = 1.0
                
            path_msg.poses.append(pose)
        
        # Publish path via ROS
        self.path_pub.publish(path_msg)
        self.path_published_count += 1
        
        path_length = self.calculate_path_length(self.road_network_path)
        self.get_logger().info(f"📤 Published ROAD NETWORK path via ROS #{self.path_published_count}: {len(self.road_network_path)} waypoints, length: {path_length:.1f}m")
        
        if self.path_published_count >= self.max_path_publications:
            self.path_timer.cancel()
            self.get_logger().info("✅ ROAD NETWORK path publishing complete - local planner will receive via ROS")
            self.publish_status("PATH_PUBLISHED")
    
    def check_goal_reached(self):
        """Check if vehicle reached the validated goal (via ROS communication)"""
        if not self.current_odom or not self.validated_goal_transform or self.goal_reached:
            return
        
        try:
            current_pos = (self.current_odom.pose.pose.position.x, self.current_odom.pose.pose.position.y)
            goal_pos = (self.validated_goal_transform.location.x, self.validated_goal_transform.location.y)
            
            distance_to_goal = math.sqrt(
                (current_pos[0] - goal_pos[0])**2 + 
                (current_pos[1] - goal_pos[1])**2
            )
            
            if distance_to_goal < 10.0:
                self.goal_reached = True
                self.publish_status("GOAL_REACHED")
                # Publish goal reached via ROS
                goal_reached_msg = Bool()
                goal_reached_msg.data = True
                # Note: We'd need to create this publisher if we want to notify other nodes
                self.get_logger().info("🎯 ROAD NETWORK GOAL REACHED! Mission completed successfully!")
                return True
                
            return False
            
        except Exception as e:
            self.get_logger().error(f"Error checking goal: {e}")
            return False
    
    def monitor_loop(self):
        """Monitoring loop with ROS communication"""
        if not self.vehicle_spawned:
            return
        
        try:
            # Check vehicle status via ROS
            if not self.vehicle or not self.vehicle.is_alive:
                self.get_logger().error("Vehicle no longer exists!")
                self.publish_status("VEHICLE_LOST")
                return
            
            # Update current position from ROS odometry
            if self.current_odom:
                self.current_position = (self.current_odom.pose.pose.position.x, self.current_odom.pose.pose.position.y)
            
            # Check if goal reached
            if self.check_goal_reached():
                return
            
            # Log status with ROS data
            if self.current_position and self.validated_goal_transform:
                distance_to_goal = math.sqrt(
                    (self.current_position[0] - self.validated_goal_transform.location.x)**2 + 
                    (self.current_position[1] - self.validated_goal_transform.location.y)**2
                )
                
                # Log with ROS data availability
                ros_data_status = f"odom={'✓' if self.current_odom else '✗'}, status={'✓' if self.vehicle_status else '✗'}"
                self.get_logger().info(f"📍 Status: pos=({self.current_position[0]:.1f}, {self.current_position[1]:.1f}), "
                                     f"distance_to_goal={distance_to_goal:.1f}m, ROS_data=[{ros_data_status}]")
            
        except Exception as e:
            self.get_logger().error(f"Error in monitor loop: {e}")
            self.publish_status("MONITOR_ERROR")

    def destroy_node(self):
        """Cleanup with ROS communication"""
        try:
            self.publish_status("SHUTTING_DOWN")
            
            if self.vehicle and self.vehicle.is_alive:
                # Stop vehicle
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = 1.0
                control.hand_brake = True
                control.reverse = False
                self.vehicle.apply_control(control)
                time.sleep(0.5)
                self.vehicle.destroy()
                
            if hasattr(self, 'lidar_sensor') and self.lidar_sensor and self.lidar_sensor.is_alive:
                self.lidar_sensor.destroy()
                
            self.get_logger().info("🌐 Road Network Global Planner with ROS Communication cleaned up")
            self.publish_status("SHUTDOWN_COMPLETE")
            
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    planner = None
    try:
        planner = RoadNetworkGlobalPlanner()
        rclpy.spin(planner)
    except KeyboardInterrupt:
        print("Shutting down Road Network Global Planner...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if planner:
            planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
