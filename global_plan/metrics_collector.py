#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import json
import csv
import os
import time
import math
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

# CARLA import for direct obstacle detection
import carla

# ROS2 Messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String, Float32
from carla_msgs.msg import CarlaEgoVehicleControl

class EnhancedNav2MetricsCollector(Node):
    def __init__(self):
        super().__init__('enhanced_nav2_metrics_collector')
        
        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # === Subscribers ===
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, nav_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, sensor_qos)
        self.control_sub = self.create_subscription(
            CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd',
            self.control_callback, sensor_qos)
        self.status_sub = self.create_subscription(
            String, '/local_planner_status', self.status_callback, sensor_qos)
        
        # === NEW: Safety Distance and Speed Response Monitoring ===
        self.front_distance_history = deque(maxlen=1000)  # Front obstacle distances
        self.speed_response_history = deque(maxlen=1000)   # Speed changes in response to obstacles
        self.brake_response_history = deque(maxlen=1000)   # Brake applications
        self.safety_zone_history = deque(maxlen=1000)      # Which safety zone vehicle is in
        
        # === CARLA Connection for Direct Obstacle Detection ===
        self.client = None
        self.world = None
        self.vehicle = None
        self.map = None
        
        # Connect to CARLA for direct vehicle access
        self.carla_connection_timer = self.create_timer(1.0, self.connect_to_carla)
        
        # Safety zone definitions (matching your nav_dwb_ros.py)
        self.emergency_distance = 10.40
        self.critical_distance = 15.02
        self.warning_distance = 18.68
        self.safe_distance = 24.99
        
        # Response tracking variables
        self.previous_speed = 0.0
        self.previous_front_distance = float('inf')
        self.current_safety_zone = "CLEAR"
        self.last_safety_status = "Unknown"
        
        # Safety event counters
        self.emergency_activations = 0
        self.critical_activations = 0
        self.warning_activations = 0
        self.safe_zone_activations = 0
        
        # Speed response metrics
        self.speed_reduction_events = 0
        self.speed_increase_events = 0
        self.total_speed_reductions = 0.0
        self.total_speed_increases = 0.0
        self.max_speed_reduction = 0.0
        self.max_brake_application = 0.0
        
        # Time in each safety zone
        self.time_in_emergency = 0.0
        self.time_in_critical = 0.0
        self.time_in_warning = 0.0
        self.time_in_safe = 0.0
        self.time_in_clear = 0.0
        self.zone_start_time = time.time()
        
        # === Mission State Monitoring ===
        self.mission_state = "NOT_STARTED"
        self.start_time = None
        self.end_time = None
        self.start_position = None
        self.goal_position = None
        self.final_position = None
        
        # === Path and Trajectory Data ===
        self.planned_path = []
        self.actual_trajectory = []
        self.position_history = deque(maxlen=1000)
        
        # === Distance and Time Metrics ===
        self.total_distance_traveled = 0.0
        self.last_position = None
        self.mission_duration = 0.0
        self.waypoints_reached = 0
        self.total_waypoints = 0
        
        # === Computational Load Monitoring ===
        self.cpu_usage_history = deque(maxlen=200)
        self.memory_usage_history = deque(maxlen=200)
        self.control_frequency_history = deque(maxlen=100)
        self.last_control_time = None
        
        # === Navigation Performance ===
        self.max_speed_achieved = 0.0
        self.avg_speed = 0.0
        self.speed_history = deque(maxlen=200)
        self.path_following_errors = deque(maxlen=200)
        
        # === Error and Precision Metrics ===
        self.goal_reach_error = 0.0
        self.path_deviation_max = 0.0
        self.path_deviation_avg = 0.0
        self.smoothness_metric = 0.0
        
        # === Safety and Efficiency ===
        self.emergency_stops = 0
        self.sharp_turns = 0
        self.direction_changes = 0
        self.last_heading = None
        
        # === Data Collection Control ===
        self.collecting_data = False
        self.data_collection_timer = self.create_timer(0.1, self.collect_system_metrics)
        
        # === Output Configuration ===
        self.experiment_name = "nav2_safety_analysis"
        self.output_dir = "./nav2_thesis_metrics"
        self.ensure_output_directory()
        
        # === Enhanced real-time data storage ===
        self.realtime_data = {
            'timestamp': [],
            'position_x': [],
            'position_y': [],
            'speed_kmh': [],
            'distance_to_goal': [],
            'path_error': [],
            'cpu_usage': [],
            'memory_usage': [],
            'control_frequency': [],
            # NEW: Safety and response metrics
            'front_obstacle_distance': [],
            'safety_zone': [],
            'speed_change_kmh': [],
            'brake_power': [],
            'throttle_power': [],
            'safety_status': [],
            'response_time_ms': []
        }
        
        self.get_logger().info("🔬 Enhanced Nav2 Safety Metrics Collector initialized")
        self.get_logger().info(f"📊 Output directory: {self.output_dir}")
        self.get_logger().info(f"🛡️ Safety distance monitoring: Emergency={self.emergency_distance}m, Critical={self.critical_distance}m")
    
    def connect_to_carla(self):
        """Connect to CARLA to access vehicle data directly"""
        try:
            if not self.client:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.get_logger().info("🔗 Connected to CARLA for obstacle detection")
            
            # Find ego vehicle
            if not self.vehicle:
                actors = self.world.get_actors()
                for actor in actors:
                    if ('vehicle' in actor.type_id and 
                        hasattr(actor, 'attributes') and 
                        actor.attributes.get('role_name') == 'ego_vehicle'):
                        self.vehicle = actor
                        self.get_logger().info(f"✅ Found ego vehicle for distance monitoring: {self.vehicle.type_id}")
                        # Cancel connection timer once vehicle is found
                        self.carla_connection_timer.cancel()
                        break
                
                # Fallback search
                if not self.vehicle:
                    vehicles = self.world.get_actors().filter('vehicle.*')
                    if vehicles:
                        for vehicle in vehicles:
                            if hasattr(vehicle, 'attributes') and vehicle.attributes.get('role_name') == 'ego_vehicle':
                                self.vehicle = vehicle
                                self.get_logger().info(f"✅ Found ego vehicle (fallback): {self.vehicle.type_id}")
                                self.carla_connection_timer.cancel()
                                break
        
        except Exception as e:
            self.get_logger().warn(f"CARLA connection for metrics: {e}")
    
    def detect_front_obstacle_distance(self):
        """Detect front obstacle distance using CARLA world data (same logic as nav_dwb_ros.py)"""
        if not self.world or not self.vehicle:
            return float('inf')
        
        try:
            # Get ego vehicle state
            ego_transform = self.vehicle.get_transform()
            ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
            ego_heading = math.radians(ego_transform.rotation.yaw)
            
            nearby_vehicles = {}
            nearby_pedestrians = {}
            detection_range = 100.0
            
            # Get all actors
            all_actors = self.world.get_actors()
            
            for actor in all_actors:
                # Skip self
                if actor.id == self.vehicle.id:
                    continue
                
                try:
                    # Check if actor is alive
                    if not actor.is_alive:
                        continue
                        
                    actor_location = actor.get_location()
                    actor_pos = np.array([actor_location.x, actor_location.y])
                    
                    # Calculate distance
                    raw_distance = np.linalg.norm(actor_pos - ego_pos)
                    vehicle_safety_radius = 1.5
                    actual_gap = raw_distance - (2 * vehicle_safety_radius)
                    distance = max(0.1, actual_gap)
                    
                    # Only process obstacles within detection range
                    if distance <= detection_range:
                        # Calculate relative angle
                        relative_pos = actor_pos - ego_pos
                        angle_to_obstacle = math.atan2(relative_pos[1], relative_pos[0])
                        relative_angle = angle_to_obstacle - ego_heading
                        
                        # Normalize angle to [-π, π]
                        while relative_angle > math.pi:
                            relative_angle -= 2 * math.pi
                        while relative_angle < -math.pi:
                            relative_angle += 2 * math.pi
                        
                        # Create obstacle info
                        obstacle_info = {
                            'id': actor.id,
                            'type': actor.type_id,
                            'distance': distance,
                            'position': actor_pos,
                            'relative_angle': relative_angle,
                            'location': actor_location
                        }
                        
                        # Categorize obstacles
                        if 'vehicle' in actor.type_id.lower():
                            nearby_vehicles[actor.id] = obstacle_info
                        elif 'walker' in actor.type_id.lower() or 'pedestrian' in actor.type_id.lower():
                            nearby_pedestrians[actor.id] = obstacle_info
                
                except Exception:
                    continue
            
            # Calculate front obstacle distance (±20 degrees)
            front_distances = []
            front_angle_threshold = math.pi / 9  # 20 degrees
            
            # Check vehicles in front
            for obstacle in nearby_vehicles.values():
                if abs(obstacle['relative_angle']) <= front_angle_threshold:
                    front_distances.append(obstacle['distance'])
            
            # Check pedestrians in front
            for obstacle in nearby_pedestrians.values():
                if abs(obstacle['relative_angle']) <= front_angle_threshold:
                    front_distances.append(obstacle['distance'])
            
            # Return minimum front distance
            if front_distances:
                return min(front_distances)
            else:
                return float('inf')
                
        except Exception as e:
            self.get_logger().debug(f"Front distance detection error: {e}")
            return float('inf')
    
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.get_logger().info(f"📁 Created output directory: {self.output_dir}")
    
    def status_callback(self, msg):
        """Monitor safety status from local planner"""
        status = msg.data
        current_time = time.time()
        
        # Extract safety information from status messages
        if "EMERGENCY" in status:
            new_zone = "EMERGENCY"
            if self.current_safety_zone != "EMERGENCY":
                self.emergency_activations += 1
        elif "CRITICAL" in status:
            new_zone = "CRITICAL"
            if self.current_safety_zone != "CRITICAL":
                self.critical_activations += 1
        elif "WARNING" in status:
            new_zone = "WARNING"
            if self.current_safety_zone != "WARNING":
                self.warning_activations += 1
        elif "SAFE" in status and "BRAKE" in status:
            new_zone = "SAFE"
            if self.current_safety_zone != "SAFE":
                self.safe_zone_activations += 1
        elif "CLEAR" in status:
            new_zone = "CLEAR"
        else:
            new_zone = self.current_safety_zone  # Keep current zone
        
        # Update time in zone
        if self.current_safety_zone != new_zone and self.collecting_data:
            time_in_zone = current_time - self.zone_start_time
            
            if self.current_safety_zone == "EMERGENCY":
                self.time_in_emergency += time_in_zone
            elif self.current_safety_zone == "CRITICAL":
                self.time_in_critical += time_in_zone
            elif self.current_safety_zone == "WARNING":
                self.time_in_warning += time_in_zone
            elif self.current_safety_zone == "SAFE":
                self.time_in_safe += time_in_zone
            elif self.current_safety_zone == "CLEAR":
                self.time_in_clear += time_in_zone
            
            self.zone_start_time = current_time
        
        self.current_safety_zone = new_zone
        self.last_safety_status = status
        
        # Extract front distance from status if available
        if "m" in status:
            try:
                # Look for distance pattern like "10.5m"
                import re
                distance_match = re.search(r'(\d+\.?\d*)m', status)
                if distance_match:
                    front_distance = float(distance_match.group(1))
                    # Only add if we haven't detected it directly from CARLA
                    if not self.front_distance_history or abs(self.front_distance_history[-1] - front_distance) > 1.0:
                        self.front_distance_history.append(front_distance)
                        self.previous_front_distance = front_distance
            except:
                pass
    
    def path_callback(self, msg):
        """Process planned path and start mission tracking"""
        if len(msg.poses) == 0:
            return
        
        # Extract planned path
        self.planned_path = []
        for pose in msg.poses:
            self.planned_path.append([
                pose.pose.position.x,
                pose.pose.position.y
            ])
        
        # Set start and goal positions
        if self.start_position is None:
            self.start_position = self.planned_path[0]
            self.goal_position = self.planned_path[-1]
            self.total_waypoints = len(self.planned_path)
            
            self.get_logger().info(f"🎯 Mission path received:")
            self.get_logger().info(f"   Start: ({self.start_position[0]:.1f}, {self.start_position[1]:.1f})")
            self.get_logger().info(f"   Goal:  ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f})")
            self.get_logger().info(f"   Waypoints: {self.total_waypoints}")
            
            # Calculate planned path length
            planned_distance = self.calculate_path_length(self.planned_path)
            self.get_logger().info(f"   Planned distance: {planned_distance:.1f}m")
    
    def odom_callback(self, msg):
        """Process odometry data for trajectory and safety tracking"""
        current_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        current_time = time.time()
        
        # Start mission tracking on first movement
        if self.mission_state == "NOT_STARTED" and self.start_position is not None:
            self.mission_state = "RUNNING"
            self.start_time = current_time
            self.collecting_data = True
            self.last_position = current_pos
            self.zone_start_time = current_time
            self.get_logger().info("🚀 Mission started - enhanced safety data collection active")
        
        if self.mission_state == "RUNNING":
            # Update trajectory
            self.actual_trajectory.append(current_pos)
            self.position_history.append((current_time, current_pos))
            
            # === NEW: Direct Front Distance Detection ===
            front_distance = self.detect_front_obstacle_distance()
            if front_distance != float('inf'):
                self.front_distance_history.append(front_distance)
                self.previous_front_distance = front_distance
            
            # Calculate distance traveled
            if self.last_position is not None:
                distance_increment = math.sqrt(
                    (current_pos[0] - self.last_position[0])**2 + 
                    (current_pos[1] - self.last_position[1])**2
                )
                self.total_distance_traveled += distance_increment
            
            # Calculate speed
            current_speed = math.sqrt(
                msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2
            ) * 3.6  # Convert to km/h
            
            self.speed_history.append(current_speed)
            self.max_speed_achieved = max(self.max_speed_achieved, current_speed)
            
            # === NEW: Speed Response Analysis ===
            speed_change = current_speed - self.previous_speed
            
            if abs(speed_change) > 1.0:  # Significant speed change (>1 km/h)
                if speed_change < 0:  # Speed reduction
                    self.speed_reduction_events += 1
                    self.total_speed_reductions += abs(speed_change)
                    self.max_speed_reduction = max(self.max_speed_reduction, abs(speed_change))
                else:  # Speed increase
                    self.speed_increase_events += 1
                    self.total_speed_increases += speed_change
            
            self.speed_response_history.append(speed_change)
            
            # Calculate path following error
            path_error = 0.0
            if self.planned_path:
                path_error = self.calculate_min_distance_to_path(current_pos)
                self.path_following_errors.append(path_error)
                self.path_deviation_max = max(self.path_deviation_max, path_error)
            
            # Calculate distance to goal
            distance_to_goal = 0.0
            if self.goal_position is not None:
                distance_to_goal = math.sqrt(
                    (current_pos[0] - self.goal_position[0])**2 + 
                    (current_pos[1] - self.goal_position[1])**2
                )
                
                # Mission completed when very close to goal and nearly stopped
                if distance_to_goal < 8.0 and current_speed < 1.0:
                    if self.mission_state == "RUNNING":
                        self.complete_mission(current_pos, current_time)
            
            # === NEW: Enhanced real-time data storage ===
            timestamp_relative = current_time - self.start_time if self.start_time else 0
            
            self.realtime_data['timestamp'].append(timestamp_relative)
            self.realtime_data['position_x'].append(current_pos[0])
            self.realtime_data['position_y'].append(current_pos[1])
            self.realtime_data['speed_kmh'].append(current_speed)
            self.realtime_data['distance_to_goal'].append(distance_to_goal)
            self.realtime_data['path_error'].append(path_error)
            
            # System metrics
            cpu_usage = self.cpu_usage_history[-1] if self.cpu_usage_history else 0
            memory_usage = self.memory_usage_history[-1] if self.memory_usage_history else 0
            control_freq = self.control_frequency_history[-1] if self.control_frequency_history else 0
            
            self.realtime_data['cpu_usage'].append(cpu_usage)
            self.realtime_data['memory_usage'].append(memory_usage)
            self.realtime_data['control_frequency'].append(control_freq)
            
            # NEW: Safety and response metrics
            front_distance = self.front_distance_history[-1] if self.front_distance_history else float('inf')
            self.realtime_data['front_obstacle_distance'].append(front_distance if front_distance != float('inf') else 999.9)
            self.realtime_data['safety_zone'].append(self.current_safety_zone)
            self.realtime_data['speed_change_kmh'].append(speed_change)
            self.realtime_data['brake_power'].append(0.0)  # Will be updated in control_callback
            self.realtime_data['throttle_power'].append(0.0)  # Will be updated in control_callback
            self.realtime_data['safety_status'].append(self.last_safety_status)
            
            # Calculate response time (simplified - time since last speed change)
            response_time = 100.0  # Default 100ms
            if len(self.speed_response_history) > 1:
                response_time = 100.0  # You can implement more sophisticated response time calculation
            self.realtime_data['response_time_ms'].append(response_time)
            
            # Track heading changes for smoothness
            current_heading = self.extract_heading_from_quaternion(msg.pose.pose.orientation)
            if self.last_heading is not None:
                heading_change = abs(current_heading - self.last_heading)
                if heading_change > math.pi:
                    heading_change = 2 * math.pi - heading_change
                
                if heading_change > math.pi / 6:  # 30 degrees
                    self.sharp_turns += 1
            
            self.last_position = current_pos
            self.last_heading = current_heading
            self.previous_speed = current_speed
    
    def control_callback(self, msg):
        """Monitor control commands for enhanced safety analysis"""
        current_time = time.time()
        
        # Calculate control frequency
        if self.last_control_time is not None:
            control_frequency = 1.0 / (current_time - self.last_control_time)
            self.control_frequency_history.append(control_frequency)
        
        self.last_control_time = current_time
        
        # === NEW: Enhanced brake and throttle monitoring ===
        brake_power = msg.brake
        throttle_power = msg.throttle
        
        # Store brake and throttle history
        self.brake_response_history.append({
            'timestamp': current_time,
            'brake_power': brake_power,
            'throttle_power': throttle_power,
            'safety_zone': self.current_safety_zone
        })
        
        # Update real-time data if it exists
        if self.realtime_data['brake_power']:
            self.realtime_data['brake_power'][-1] = brake_power
            self.realtime_data['throttle_power'][-1] = throttle_power
        
        # Track maximum brake application
        self.max_brake_application = max(self.max_brake_application, brake_power)
        
        # Detect emergency stops (high brake values)
        if brake_power > 0.7:
            self.emergency_stops += 1
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        if not self.collecting_data:
            return
        
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_info.percent)
            
        except Exception as e:
            self.get_logger().warn(f"Error collecting system metrics: {e}")
    
    def complete_mission(self, final_pos, end_time):
        """Complete mission and calculate enhanced metrics"""
        self.mission_state = "COMPLETED"
        self.end_time = end_time
        self.final_position = final_pos
        self.collecting_data = False
        self.mission_duration = end_time - self.start_time
        
        # Update final zone time
        final_zone_time = end_time - self.zone_start_time
        if self.current_safety_zone == "EMERGENCY":
            self.time_in_emergency += final_zone_time
        elif self.current_safety_zone == "CRITICAL":
            self.time_in_critical += final_zone_time
        elif self.current_safety_zone == "WARNING":
            self.time_in_warning += final_zone_time
        elif self.current_safety_zone == "SAFE":
            self.time_in_safe += final_zone_time
        elif self.current_safety_zone == "CLEAR":
            self.time_in_clear += final_zone_time
        
        # Calculate final metrics
        self.calculate_final_metrics()
        
        # Save all data
        self.save_enhanced_metrics()
        self.save_enhanced_pandas_data()
        
        self.get_logger().info("🏁 Enhanced mission completed - safety metrics saved")
        self.log_enhanced_summary_metrics()
    
    def calculate_final_metrics(self):
        """Calculate final derived metrics including safety metrics"""
        # Original metrics
        if self.final_position and self.goal_position:
            self.goal_reach_error = math.sqrt(
                (self.final_position[0] - self.goal_position[0])**2 + 
                (self.final_position[1] - self.goal_position[1])**2
            )
        
        if self.speed_history:
            self.avg_speed = sum(self.speed_history) / len(self.speed_history)
        
        if self.path_following_errors:
            self.path_deviation_avg = sum(self.path_following_errors) / len(self.path_following_errors)
        
        if len(self.speed_history) > 1:
            speed_variations = []
            for i in range(1, len(self.speed_history)):
                speed_variations.append(abs(self.speed_history[i] - self.speed_history[i-1]))
            self.smoothness_metric = sum(speed_variations) / len(speed_variations)
    
    def calculate_min_distance_to_path(self, current_pos):
        """Calculate minimum distance from current position to planned path"""
        if not self.planned_path:
            return 0.0
        
        min_distance = float('inf')
        for path_point in self.planned_path:
            distance = math.sqrt(
                (current_pos[0] - path_point[0])**2 + 
                (current_pos[1] - path_point[1])**2
            )
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def calculate_path_length(self, path):
        """Calculate total length of a path"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            distance = math.sqrt(
                (path[i+1][0] - path[i][0])**2 + 
                (path[i+1][1] - path[i][1])**2
            )
            total_length += distance
        
        return total_length
    
    def extract_heading_from_quaternion(self, orientation):
        """Extract yaw angle from quaternion"""
        try:
            import tf_transformations
            _, _, yaw = tf_transformations.euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            return yaw
        except ImportError:
            # Simple quaternion to yaw conversion if tf_transformations not available
            siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
            cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
            return math.atan2(siny_cosp, cosy_cosp)
    
    def save_enhanced_pandas_data(self):
        """Save enhanced data in pandas format for detailed safety analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Enhanced real-time trajectory data with safety metrics
        df_realtime = pd.DataFrame(self.realtime_data)
        realtime_file = f"{self.output_dir}/{self.experiment_name}_{timestamp}_enhanced_realtime_data.csv"
        df_realtime.to_csv(realtime_file, index=False)
        
        pickle_file = f"{self.output_dir}/{self.experiment_name}_{timestamp}_enhanced_realtime_data.pkl"
        df_realtime.to_pickle(pickle_file)
        
        # 2. Safety Response Analysis DataFrame
        safety_analysis = {
            'emergency_activations': [self.emergency_activations],
            'critical_activations': [self.critical_activations],
            'warning_activations': [self.warning_activations],
            'safe_zone_activations': [self.safe_zone_activations],
            'time_in_emergency_s': [round(self.time_in_emergency, 2)],
            'time_in_critical_s': [round(self.time_in_critical, 2)],
            'time_in_warning_s': [round(self.time_in_warning, 2)],
            'time_in_safe_s': [round(self.time_in_safe, 2)],
            'time_in_clear_s': [round(self.time_in_clear, 2)],
            'speed_reduction_events': [self.speed_reduction_events],
            'speed_increase_events': [self.speed_increase_events],
            'avg_speed_reduction_kmh': [round(self.total_speed_reductions / max(1, self.speed_reduction_events), 2)],
            'max_speed_reduction_kmh': [round(self.max_speed_reduction, 2)],
            'max_brake_application': [round(self.max_brake_application, 3)],
            'avg_front_distance_m': [round(sum(self.front_distance_history) / max(1, len(self.front_distance_history)), 2)],
            'min_front_distance_m': [round(min(self.front_distance_history) if self.front_distance_history else 0, 2)],
            'safety_response_efficiency': [self.calculate_safety_response_efficiency()]
        }
        
        df_safety = pd.DataFrame(safety_analysis)
        safety_file = f"{self.output_dir}/{self.experiment_name}_{timestamp}_safety_analysis.csv"
        df_safety.to_csv(safety_file, index=False)
        
        # 3. Enhanced Summary metrics DataFrame
        summary_metrics = {
            'experiment_name': [self.experiment_name],
            'timestamp': [timestamp],
            'total_distance_traveled_m': [round(self.total_distance_traveled, 2)],
            'mission_duration_s': [round(self.mission_duration, 2)],
            'goal_reach_error_m': [round(self.goal_reach_error, 2)],
            'planned_path_length_m': [round(self.calculate_path_length(self.planned_path), 2)],
            'path_efficiency_ratio': [round(self.calculate_path_length(self.planned_path) / self.total_distance_traveled if self.total_distance_traveled > 0 else 0, 3)],
            'avg_cpu_usage_percent': [round(sum(self.cpu_usage_history) / max(1, len(self.cpu_usage_history)), 2)],
            'max_cpu_usage_percent': [round(max(self.cpu_usage_history) if self.cpu_usage_history else 0, 2)],
            'avg_memory_usage_percent': [round(sum(self.memory_usage_history) / max(1, len(self.memory_usage_history)), 2)],
            'avg_control_frequency_hz': [round(sum(self.control_frequency_history) / max(1, len(self.control_frequency_history)), 2)],
            'max_speed_achieved_kmh': [round(self.max_speed_achieved, 2)],
            'avg_speed_kmh': [round(self.avg_speed, 2)],
            'path_deviation_max_m': [round(self.path_deviation_max, 2)],
            'path_deviation_avg_m': [round(self.path_deviation_avg, 2)],
            'smoothness_metric': [round(self.smoothness_metric, 3)],
            'emergency_stops': [self.emergency_stops],
            'sharp_turns': [self.sharp_turns],
            'total_waypoints': [self.total_waypoints],
            'computational_load_rating': [self.calculate_computational_load_rating()],
            # NEW: Safety metrics in summary
            'safety_system_activations': [self.emergency_activations + self.critical_activations + self.warning_activations],
            'percentage_time_in_safety_zones': [round((self.time_in_emergency + self.time_in_critical + self.time_in_warning + self.time_in_safe) / max(0.1, self.mission_duration) * 100, 2)],
            'average_brake_power': [round(sum([b['brake_power'] for b in self.brake_response_history]) / max(1, len(self.brake_response_history)), 3)],
            'safety_response_score': [self.calculate_safety_response_score()]
        }
        
        df_summary = pd.DataFrame(summary_metrics)
        summary_file = f"{self.output_dir}/{self.experiment_name}_{timestamp}_enhanced_summary_metrics.csv"
        df_summary.to_csv(summary_file, index=False)
        
        # 4. Detailed brake response timeline
        if self.brake_response_history:
            brake_timeline = pd.DataFrame(self.brake_response_history)
            brake_file = f"{self.output_dir}/{self.experiment_name}_{timestamp}_brake_response_timeline.csv"
            brake_timeline.to_csv(brake_file, index=False)
        
        self.get_logger().info(f"📈 Enhanced pandas data saved:")
        self.get_logger().info(f"   Real-time: {realtime_file}")
        self.get_logger().info(f"   Safety:    {safety_file}")
        self.get_logger().info(f"   Summary:   {summary_file}")
        if self.brake_response_history:
            self.get_logger().info(f"   Brakes:    {brake_file}")
    
    def calculate_safety_response_efficiency(self):
        """Calculate how efficiently the system responds to safety threats"""
        if not self.speed_reduction_events or not self.front_distance_history:
            return 0.0
        
        # Efficiency based on speed reduction per safety activation
        total_activations = self.emergency_activations + self.critical_activations + self.warning_activations
        if total_activations == 0:
            return 100.0  # No safety threats = perfect efficiency
        
        # Higher score for more speed reductions relative to safety activations
        efficiency = min(100.0, (self.speed_reduction_events / total_activations) * 50.0)
        return round(efficiency, 2)
    
    def calculate_safety_response_score(self):
        """Calculate overall safety response score (0-100)"""
        score_components = []
        
        # 1. Collision avoidance (40% weight)
        if self.emergency_stops == 0:
            collision_score = 100.0
        elif self.emergency_stops < 3:
            collision_score = 80.0
        elif self.emergency_stops < 10:
            collision_score = 60.0
        else:
            collision_score = 30.0
        score_components.append(collision_score * 0.4)
        
        # 2. Speed response appropriateness (30% weight)
        if self.speed_reduction_events > 0:
            avg_reduction = self.total_speed_reductions / self.speed_reduction_events
            if 2.0 <= avg_reduction <= 15.0:  # Appropriate speed reductions
                speed_score = 100.0
            elif avg_reduction > 15.0:  # Over-reactive
                speed_score = 70.0
            else:  # Under-reactive
                speed_score = 50.0
        else:
            speed_score = 85.0  # No reductions needed
        score_components.append(speed_score * 0.3)
        
        # 3. Safety zone utilization (20% weight)
        total_mission_time = max(0.1, self.mission_duration)
        critical_time_ratio = (self.time_in_emergency + self.time_in_critical) / total_mission_time
        
        if critical_time_ratio < 0.1:  # Less than 10% in critical zones
            zone_score = 100.0
        elif critical_time_ratio < 0.2:
            zone_score = 80.0
        elif critical_time_ratio < 0.3:
            zone_score = 60.0
        else:
            zone_score = 40.0
        score_components.append(zone_score * 0.2)
        
        # 4. Brake efficiency (10% weight)
        if self.max_brake_application > 0:
            if 0.3 <= self.max_brake_application <= 0.8:  # Appropriate braking
                brake_score = 100.0
            elif self.max_brake_application > 0.8:  # Emergency braking used
                brake_score = 75.0
            else:  # Under-braking
                brake_score = 60.0
        else:
            brake_score = 90.0  # No braking needed
        score_components.append(brake_score * 0.1)
        
        total_score = sum(score_components)
        return round(total_score, 2)
    
    def save_enhanced_metrics(self):
        """Save enhanced metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced metrics dictionary with safety analysis
        metrics = {
            "experiment_info": {
                "experiment_name": self.experiment_name,
                "timestamp": timestamp,
                "carla_version": "0.9.13",
                "nav2_version": "ROS2 Foxy",
                "analysis_type": "Enhanced Safety and Response Analysis"
            },
            "mission_metrics": {
                "total_distance_traveled_m": round(self.total_distance_traveled, 2),
                "mission_duration_s": round(self.mission_duration, 2),
                "goal_reach_error_m": round(self.goal_reach_error, 2),
                "planned_path_length_m": round(self.calculate_path_length(self.planned_path), 2),
                "path_efficiency_ratio": round(self.calculate_path_length(self.planned_path) / self.total_distance_traveled if self.total_distance_traveled > 0 else 0, 3),
                "waypoints_total": self.total_waypoints,
                "mission_status": self.mission_state
            },
            "performance_metrics": {
                "avg_cpu_usage_percent": round(sum(self.cpu_usage_history) / max(1, len(self.cpu_usage_history)), 2),
                "max_cpu_usage_percent": round(max(self.cpu_usage_history) if self.cpu_usage_history else 0, 2),
                "avg_memory_usage_percent": round(sum(self.memory_usage_history) / max(1, len(self.memory_usage_history)), 2),
                "avg_control_frequency_hz": round(sum(self.control_frequency_history) / max(1, len(self.control_frequency_history)), 2),
                "computational_load_rating": self.calculate_computational_load_rating()
            },
            "navigation_metrics": {
                "max_speed_achieved_kmh": round(self.max_speed_achieved, 2),
                "avg_speed_kmh": round(self.avg_speed, 2),
                "path_deviation_max_m": round(self.path_deviation_max, 2),
                "path_deviation_avg_m": round(self.path_deviation_avg, 2),
                "smoothness_metric": round(self.smoothness_metric, 3),
                "emergency_stops": self.emergency_stops,
                "sharp_turns": self.sharp_turns
            },
            # NEW: Enhanced safety and response metrics
            "safety_response_metrics": {
                "safety_zone_parameters": {
                    "emergency_distance_m": self.emergency_distance,
                    "critical_distance_m": self.critical_distance,
                    "warning_distance_m": self.warning_distance,
                    "safe_distance_m": self.safe_distance
                },
                "safety_activations": {
                    "emergency_activations": self.emergency_activations,
                    "critical_activations": self.critical_activations,
                    "warning_activations": self.warning_activations,
                    "safe_zone_activations": self.safe_zone_activations,
                    "total_safety_activations": self.emergency_activations + self.critical_activations + self.warning_activations + self.safe_zone_activations
                },
                "time_distribution": {
                    "time_in_emergency_s": round(self.time_in_emergency, 2),
                    "time_in_critical_s": round(self.time_in_critical, 2),
                    "time_in_warning_s": round(self.time_in_warning, 2),
                    "time_in_safe_s": round(self.time_in_safe, 2),
                    "time_in_clear_s": round(self.time_in_clear, 2),
                    "percentage_in_danger_zones": round((self.time_in_emergency + self.time_in_critical) / max(0.1, self.mission_duration) * 100, 2),
                    "percentage_in_caution_zones": round((self.time_in_warning + self.time_in_safe) / max(0.1, self.mission_duration) * 100, 2),
                    "percentage_in_clear": round(self.time_in_clear / max(0.1, self.mission_duration) * 100, 2)
                },
                "speed_response_analysis": {
                    "speed_reduction_events": self.speed_reduction_events,
                    "speed_increase_events": self.speed_increase_events,
                    "total_speed_reductions_kmh": round(self.total_speed_reductions, 2),
                    "total_speed_increases_kmh": round(self.total_speed_increases, 2),
                    "avg_speed_reduction_kmh": round(self.total_speed_reductions / max(1, self.speed_reduction_events), 2),
                    "max_speed_reduction_kmh": round(self.max_speed_reduction, 2),
                    "speed_response_ratio": round(self.speed_reduction_events / max(1, self.speed_increase_events), 2)
                },
                "braking_analysis": {
                    "max_brake_application": round(self.max_brake_application, 3),
                    "total_brake_events": len([b for b in self.brake_response_history if b['brake_power'] > 0.1]),
                    "emergency_brake_events": len([b for b in self.brake_response_history if b['brake_power'] > 0.7]),
                    "avg_brake_power": round(sum([b['brake_power'] for b in self.brake_response_history]) / max(1, len(self.brake_response_history)), 3),
                    "brake_response_distribution": {
                        "light_braking_0_0.3": len([b for b in self.brake_response_history if 0.1 <= b['brake_power'] <= 0.3]),
                        "moderate_braking_0.3_0.6": len([b for b in self.brake_response_history if 0.3 < b['brake_power'] <= 0.6]),
                        "heavy_braking_0.6_0.8": len([b for b in self.brake_response_history if 0.6 < b['brake_power'] <= 0.8]),
                        "emergency_braking_0.8_1.0": len([b for b in self.brake_response_history if b['brake_power'] > 0.8])
                    }
                },
                "obstacle_distance_analysis": {
                    "min_front_distance_m": round(min(self.front_distance_history) if self.front_distance_history else float('inf'), 2),
                    "avg_front_distance_m": round(sum(self.front_distance_history) / max(1, len(self.front_distance_history)), 2),
                    "max_front_distance_m": round(max(self.front_distance_history) if self.front_distance_history else 0, 2),
                    "distance_samples": len(self.front_distance_history),
                    "close_calls_under_5m": len([d for d in self.front_distance_history if d < 5.0]),
                    "danger_zone_under_10m": len([d for d in self.front_distance_history if d < 10.0]),
                    "safe_distance_over_25m": len([d for d in self.front_distance_history if d > 25.0]),
                    "emergency_zone_samples": len([d for d in self.front_distance_history if d <= self.emergency_distance]),
                    "critical_zone_samples": len([d for d in self.front_distance_history if self.emergency_distance < d <= self.critical_distance]),
                    "warning_zone_samples": len([d for d in self.front_distance_history if self.critical_distance < d <= self.warning_distance])
                },
                "safety_performance_scores": {
                    "safety_response_efficiency": self.calculate_safety_response_efficiency(),
                    "overall_safety_score": self.calculate_safety_response_score(),
                    "collision_avoidance_rating": "EXCELLENT" if self.emergency_stops == 0 else "GOOD" if self.emergency_stops < 3 else "NEEDS_IMPROVEMENT",
                    "speed_adaptation_rating": "EXCELLENT" if self.speed_reduction_events > self.emergency_activations else "GOOD" if self.speed_reduction_events > 0 else "POOR",
                    "brake_efficiency_rating": "EXCELLENT" if 0.3 <= self.max_brake_application <= 0.8 else "GOOD" if self.max_brake_application > 0 else "POOR"
                }
            }
        }
        
        # Save enhanced JSON
        json_filename = f"{self.output_dir}/{self.experiment_name}_{timestamp}_enhanced_metrics.json"
        with open(json_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.get_logger().info(f"📊 Enhanced JSON metrics saved: {json_filename}")
        
        return metrics
    
    def calculate_computational_load_rating(self):
        """Calculate overall computational load rating (0-100)"""
        if not self.cpu_usage_history:
            return 0
        
        avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
        avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
        
        # Weighted combination (CPU more important for real-time systems)
        computational_load = (avg_cpu * 0.7) + (avg_memory * 0.3)
        return min(100, max(0, computational_load))
    
    def log_enhanced_summary_metrics(self):
        """Log enhanced summary of key safety metrics"""
        self.get_logger().info("="*80)
        self.get_logger().info("📊 ENHANCED MISSION SAFETY METRICS SUMMARY")
        self.get_logger().info("="*80)
        
        self.get_logger().info(f"🎯 PRIMARY THESIS METRICS:")
        self.get_logger().info(f"   Distance Traveled: {self.total_distance_traveled:.2f} m")
        self.get_logger().info(f"   Mission Duration:  {self.mission_duration:.2f} s")
        self.get_logger().info(f"   Goal Reach Error:  {self.goal_reach_error:.2f} m")
        self.get_logger().info(f"   Computational Load: {self.calculate_computational_load_rating():.1f} %")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"🛡️ SAFETY SYSTEM PERFORMANCE:")
        total_safety_activations = self.emergency_activations + self.critical_activations + self.warning_activations + self.safe_zone_activations
        self.get_logger().info(f"   Total Safety Activations: {total_safety_activations}")
        self.get_logger().info(f"   Emergency Activations: {self.emergency_activations}")
        self.get_logger().info(f"   Critical Activations:  {self.critical_activations}")
        self.get_logger().info(f"   Warning Activations:   {self.warning_activations}")
        self.get_logger().info(f"   Safe Zone Activations: {self.safe_zone_activations}")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"⏱️ TIME DISTRIBUTION IN SAFETY ZONES:")
        self.get_logger().info(f"   Emergency Zone: {self.time_in_emergency:.1f}s ({self.time_in_emergency/max(0.1,self.mission_duration)*100:.1f}%)")
        self.get_logger().info(f"   Critical Zone:  {self.time_in_critical:.1f}s ({self.time_in_critical/max(0.1,self.mission_duration)*100:.1f}%)")
        self.get_logger().info(f"   Warning Zone:   {self.time_in_warning:.1f}s ({self.time_in_warning/max(0.1,self.mission_duration)*100:.1f}%)")
        self.get_logger().info(f"   Safe Zone:      {self.time_in_safe:.1f}s ({self.time_in_safe/max(0.1,self.mission_duration)*100:.1f}%)")
        self.get_logger().info(f"   Clear Path:     {self.time_in_clear:.1f}s ({self.time_in_clear/max(0.1,self.mission_duration)*100:.1f}%)")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"🚗 SPEED RESPONSE ANALYSIS:")
        self.get_logger().info(f"   Speed Reduction Events: {self.speed_reduction_events}")
        self.get_logger().info(f"   Speed Increase Events:  {self.speed_increase_events}")
        if self.speed_reduction_events > 0:
            avg_reduction = self.total_speed_reductions / self.speed_reduction_events
            self.get_logger().info(f"   Avg Speed Reduction:    {avg_reduction:.1f} km/h")
            self.get_logger().info(f"   Max Speed Reduction:    {self.max_speed_reduction:.1f} km/h")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"🛑 BRAKING SYSTEM ANALYSIS:")
        self.get_logger().info(f"   Max Brake Application:  {self.max_brake_application:.3f}")
        if self.brake_response_history:
            avg_brake = sum([b['brake_power'] for b in self.brake_response_history]) / len(self.brake_response_history)
            emergency_brakes = len([b for b in self.brake_response_history if b['brake_power'] > 0.7])
            self.get_logger().info(f"   Average Brake Power:    {avg_brake:.3f}")
            self.get_logger().info(f"   Emergency Brake Events: {emergency_brakes}")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"📏 OBSTACLE DISTANCE ANALYSIS:")
        if self.front_distance_history:
            min_dist = min(self.front_distance_history)
            avg_dist = sum(self.front_distance_history) / len(self.front_distance_history)
            close_calls = len([d for d in self.front_distance_history if d < 5.0])
            self.get_logger().info(f"   Minimum Front Distance: {min_dist:.2f} m")
            self.get_logger().info(f"   Average Front Distance: {avg_dist:.2f} m")
            self.get_logger().info(f"   Close Calls (<5m):      {close_calls}")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"🏆 SAFETY PERFORMANCE SCORES:")
        self.get_logger().info(f"   Safety Response Efficiency: {self.calculate_safety_response_efficiency():.1f}%")
        self.get_logger().info(f"   Overall Safety Score:       {self.calculate_safety_response_score():.1f}%")
        
        # Assessment based on safety metrics
        danger_time_percentage = (self.time_in_emergency + self.time_in_critical) / max(0.1, self.mission_duration) * 100
        
        self.get_logger().info(f"")
        self.get_logger().info(f"📋 SAFETY ASSESSMENT:")
        if self.emergency_stops == 0 and danger_time_percentage < 10:
            self.get_logger().info("   🟢 EXCELLENT - Safe navigation with minimal risk exposure")
        elif self.emergency_stops < 3 and danger_time_percentage < 20:
            self.get_logger().info("   🟡 GOOD - Acceptable safety performance with some close calls")
        elif danger_time_percentage < 30:
            self.get_logger().info("   🟠 ACCEPTABLE - Safety systems working but frequent activations")
        else:
            self.get_logger().info("   🔴 NEEDS IMPROVEMENT - High risk exposure, system tuning required")
        
        self.get_logger().info("="*80)
    
    def force_save_metrics(self):
        """Force save current enhanced metrics (for manual testing)"""
        if self.mission_state == "RUNNING":
            current_time = time.time()
            current_pos = self.final_position or [0, 0]
            self.complete_mission(current_pos, current_time)


def main(args=None):
    rclpy.init(args=args)
    
    metrics_collector = None
    try:
        metrics_collector = EnhancedNav2MetricsCollector()
        
        # Set up signal handler for graceful shutdown
        import signal
        
        def signal_handler(sig, frame):
            if metrics_collector and metrics_collector.mission_state == "RUNNING":
                metrics_collector.get_logger().info("🛑 Manual stop - saving enhanced metrics...")
                metrics_collector.force_save_metrics()
            rclpy.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        rclpy.spin(metrics_collector)
        
    except KeyboardInterrupt:
        if metrics_collector and metrics_collector.mission_state == "RUNNING":
            metrics_collector.get_logger().info("🛑 Interrupted - saving enhanced metrics...")
            metrics_collector.force_save_metrics()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if metrics_collector:
            metrics_collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
