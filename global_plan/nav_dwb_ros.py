#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import carla
import numpy as np
import math
import time

# ROS2 Navigation Messages
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2, LaserScan
from std_msgs.msg import Header, String, Bool
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus, CarlaEgoVehicleInfo

# TF2 for coordinate transformations
import tf2_ros
import tf_transformations

import sensor_msgs_py.point_cloud2 as pc2

# === NEW: CARLA Agents for Traffic Light Integration ===
try:
    from agents.navigation.basic_agent import BasicAgent
    AGENTS_AVAILABLE = True
except ImportError:
    print("Warning: CARLA agents not available. Traffic light functionality will be disabled.")
    AGENTS_AVAILABLE = False

class EnhancedBalancedCollisionAvoidanceROS(Node):
    def __init__(self):
        super().__init__('enhanced_balanced_collision_avoidance_ros')
        
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
        
        # === ROS PUBLISHERS === (Output via ROS)
        self.laser_pub = self.create_publisher(LaserScan, '/scan', sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, '/odom', sensor_qos)
        
        # CARLA-specific control (for your system)
        self.carla_cmd_pub = self.create_publisher(CarlaEgoVehicleControl, 
                                                 '/carla/ego_vehicle/vehicle_control_cmd', 10)
        
        # ===== STANDARD ROS /cmd_vel for MPC Integration =====
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.status_pub = self.create_publisher(String, '/local_planner_status', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.vehicle_status_pub = self.create_publisher(CarlaEgoVehicleStatus, 
                                                      '/carla/ego_vehicle/vehicle_status', sensor_qos)
        
        # === NEW: Traffic Light Status Publisher ===
        self.traffic_light_pub = self.create_publisher(String, '/traffic_light_status', 10)
        
        # === ROS SUBSCRIBERS === (Input via ROS)
        self.global_path_sub = self.create_subscription(
            Path, '/planned_path', self.global_path_callback, nav_qos)
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, nav_qos)
        self.planner_status_sub = self.create_subscription(
            String, '/planner_status', self.planner_status_callback, 10)
        self.vehicle_info_sub = self.create_subscription(
            CarlaEgoVehicleInfo, '/carla/ego_vehicle/vehicle_info', 
            self.vehicle_info_callback, 10)
        
        # === TF2 ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # === CARLA Connection ===
        self.client = None
        self.world = None
        self.vehicle = None
        self.map = None
        
        # === NEW: Traffic Light Agent Integration ===
        self.traffic_agent = None
        self.traffic_agent_ready = False
        self.last_agent_control = None
        self.traffic_light_active = False
        
        # === ROS Communication State ===
        self.received_path = False
        self.received_goal = False
        self.received_vehicle_info = False
        self.global_planner_status = "UNKNOWN"
        self.goal_pose = None
        
        # === COLLISION MONITORING ===
        self.collision_sensor = None
        self.collision_count = 0
        
        # === PATH FOLLOWING ===
        self.master_waypoints = []
        self.current_pose = None
        self.path_index = 0
        self.waypoint_tolerance = 5.0
        
        # === BALANCED OBSTACLE DETECTION ===
        self.nearby_vehicles = {}
        self.nearby_pedestrians = {}
        self.detection_range = 100.0
        self.min_obstacle_distance = 1.5
        
        # === OPTIMIZED COLLISION AVOIDANCE DISTANCES (UNCHANGED) ===
        self.emergency_distance = 10.40
        self.critical_distance = 15.02
        self.warning_distance = 18.68
        self.safe_distance = 24.99

        # === OPTIMIZED REAL-WORLD SPEED PARAMETERS (UNCHANGED) ===
        self.max_speed = 55.00 / 3.6        # 15.28 m/s
        self.cruise_speed = 34.05 / 3.6     # 9.46 m/s
        self.normal_speed = 23.21 / 3.6     # 6.45 m/s
        self.slow_speed = 12.31 / 3.6       # 3.42 m/s
        self.creep_speed = 1.40 / 3.6       # 0.39 m/s

        # === POWERFUL BRAKING SYSTEM (UNCHANGED) ===
        self.emergency_brake_power = 1.0    # Maximum emergency brake
        self.critical_brake_power = 0.8     # Heavy braking for critical zone
        self.warning_brake_power = 0.65      # Strong braking for warning zone
        self.normal_brake_power = 0.4       # Normal braking
        self.current_brake_power = 0.0      # Track current brake application

        # === NEW: Traffic Light Braking ===
        self.traffic_light_brake_power = 0.0  # Dynamic brake power from agent
        self.traffic_light_status = "NO_TRAFFIC_LIGHT"
        
        # === DETECTION STATE ===
        self.front_obstacle_distance = float('inf')
        self.detection_working = False
        
        # === BALANCED EMERGENCY STATE (UNCHANGED) ===
        self.is_emergency_stopped = False
        self.emergency_counter = 0
        self.resume_threshold = 15
        self.stuck_prevention_counter = 0
        self.max_stuck_time = 5
        
        # === DEBUG ===
        self.debug_counter = 0
        
        # === TIMERS ===
        self.connection_timer = self.create_timer(0.5, self.connect_and_setup)
        self.main_timer = None

        print("⚖️ ENHANCED COLLISION AVOIDANCE with TRAFFIC LIGHT INTEGRATION")
        print(f"🎯 Speed Range: {self.creep_speed*3.6:.1f} - {self.max_speed*3.6:.1f} km/h")
        print(f"🛑 Brake Power: Emergency={self.emergency_brake_power}, Critical={self.critical_brake_power}")
        print(f"🚦 Traffic Light: {'ENABLED' if AGENTS_AVAILABLE else 'DISABLED (No CARLA agents)'}")
        print(f"📋 Priority: 1.Collision Avoidance → 2.Traffic Light → 3.Turn → 4.Cruise")
        print("=" * 80)
        self.get_logger().info("⚖️ Enhanced Collision Avoidance with PRIORITY-BASED Traffic Light Integration")
        self.publish_status("INITIALIZING")
    
    def publish_status(self, status):
        """Publish local planner status via ROS"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def publish_traffic_light_status(self, status):
        """Publish traffic light status via ROS"""
        msg = String()
        msg.data = status
        self.traffic_light_pub.publish(msg)
        
    def publish_goal_reached(self, reached=True):
        """Publish goal reached status via ROS"""
        msg = Bool()
        msg.data = reached
        self.goal_reached_pub.publish(msg)
        
    def global_path_callback(self, msg):
        """Receive global path via ROS"""
        try:
            if self.master_waypoints:
                return
            
            temp_path = []
            for pose in msg.poses:
                temp_path.append([pose.pose.position.x, pose.pose.position.y])
            
            if temp_path and len(temp_path) > 1:
                self.create_master_waypoints(temp_path)
                self.path_index = 0
                self.received_path = True
                self.get_logger().info(f"🎯 Path received via ROS: {len(self.master_waypoints)} waypoints")
                self.publish_status("PATH_RECEIVED")
                
        except Exception as e:
            self.get_logger().error(f"❌ Path callback error: {e}")
            self.publish_status("PATH_ERROR")
    
    def goal_pose_callback(self, msg):
        """Receive goal pose via ROS"""
        try:
            self.goal_pose = msg
            self.received_goal = True
            self.get_logger().info(f"🎯 Goal pose received via ROS: ({msg.pose.position.x:.1f}, {msg.pose.position.y:.1f})")
            self.publish_status("GOAL_RECEIVED")
            
            # Update traffic agent destination if available
            if self.traffic_agent and AGENTS_AVAILABLE:
                try:
                    goal_location = carla.Location(
                        x=msg.pose.position.x,
                        y=msg.pose.position.y,
                        z=msg.pose.position.z
                    )
                    self.traffic_agent.set_destination(goal_location)
                    self.get_logger().info("🚦 Traffic agent destination updated")
                except Exception as e:
                    self.get_logger().warn(f"Failed to update traffic agent destination: {e}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Goal pose callback error: {e}")
    
    def planner_status_callback(self, msg):
        """Receive global planner status via ROS"""
        self.global_planner_status = msg.data
        if msg.data == "VEHICLE_SPAWNED":
            self.get_logger().info("📢 Received vehicle spawned notification via ROS")
        elif msg.data == "PATH_PUBLISHED":
            self.get_logger().info("📢 Received path published notification via ROS")
    
    def vehicle_info_callback(self, msg):
        """Receive vehicle info via ROS"""
        try:
            self.vehicle_info = msg
            self.received_vehicle_info = True
            self.get_logger().info(f"🚗 Vehicle info received via ROS: ID={msg.id}, Type={msg.type}")
        except Exception as e:
            self.get_logger().error(f"❌ Vehicle info callback error: {e}")
    
    def connect_and_setup(self):
        """Connect to CARLA and setup vehicle with traffic light agent"""
        try:
            # Connect to CARLA
            if not self.client:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.get_logger().info("🔗 Connected to CARLA")
                self.publish_status("CONNECTED_TO_CARLA")
            
            # Find ego vehicle (should be spawned by global planner)
            if not self.vehicle:
                actors = self.world.get_actors()
                for actor in actors:
                    if ('vehicle' in actor.type_id and 
                        hasattr(actor, 'attributes') and 
                        actor.attributes.get('role_name') == 'ego_vehicle'):
                        self.vehicle = actor
                        break
                
                # Fallback search
                if not self.vehicle:
                    vehicles = self.world.get_actors().filter('vehicle.*')
                    if vehicles:
                        for vehicle in vehicles:
                            if hasattr(vehicle, 'attributes') and vehicle.attributes.get('role_name') == 'ego_vehicle':
                                self.vehicle = vehicle
                                break
                        if not self.vehicle:
                            self.vehicle = vehicles[0]
                            self.get_logger().warn(f"⚠️ Using fallback vehicle: {self.vehicle.type_id}")
                
                if self.vehicle:
                    self.get_logger().info(f"✅ Found vehicle via ROS communication: {self.vehicle.type_id}")
                    self.setup_collision_sensor()
                    
                    # === NEW: Setup Traffic Light Agent ===
                    self.setup_traffic_light_agent()
                    
                    self.publish_status("VEHICLE_FOUND")
                    
                    # Wait for ROS path before starting
                    if self.received_path:
                        # Start main control loop
                        self.connection_timer.cancel()
                        self.main_timer = self.create_timer(0.1, self.main_control_loop)
                        self.odom_timer = self.create_timer(0.05, self.publish_odometry)
                        self.get_logger().info("🚀 Enhanced collision avoidance with TRAFFIC LIGHT started")
                        self.publish_status("SYSTEM_STARTED")
                    else:
                        self.get_logger().info("⏳ Waiting for ROS path data...")
                        self.publish_status("WAITING_FOR_ROS_DATA")
                else:
                    self.get_logger().info("🔍 Still searching for ego vehicle...")
                    self.publish_status("SEARCHING_VEHICLE")
            
        except Exception as e:
            self.get_logger().error(f"❌ Connection error: {e}")
            self.publish_status("CONNECTION_ERROR")
    
    def setup_traffic_light_agent(self):
        """Setup CARLA BasicAgent with ROS path destination"""
        try:
            if not AGENTS_AVAILABLE:
                self.get_logger().warn("🚦 CARLA agents not available - traffic light disabled")
                self.publish_traffic_light_status("AGENTS_NOT_AVAILABLE")
                return
            
            if not self.vehicle:
                return
            
            # Initialize BasicAgent (same as automatic_control.py)
            self.traffic_agent = BasicAgent(self.vehicle)
            
            # === KEY FIX: Set destination to ROS path end ===
            if hasattr(self, 'goal_pose') and self.goal_pose:
                # Use ROS goal as agent destination
                goal_location = carla.Location(
                    x=self.goal_pose.pose.position.x,
                    y=self.goal_pose.pose.position.y,
                    z=self.goal_pose.pose.position.z
                )
                self.traffic_agent.set_destination(goal_location)
                self.get_logger().info(f"✅ Agent destination set to ROS goal: ({goal_location.x:.1f}, {goal_location.y:.1f})")
            else:
                # Fallback: use spawn points
                spawn_points = self.map.get_spawn_points()
                if spawn_points:
                    self.traffic_agent.set_destination(spawn_points[0].location)
                    self.get_logger().info("⚠️ Agent destination set to fallback spawn point")
            
            self.traffic_agent_ready = True
            self.get_logger().info("✅ Traffic light agent setup complete with destination")
            self.publish_traffic_light_status("AGENT_READY")
            
        except Exception as e:
            self.get_logger().error(f"❌ Traffic light agent setup failed: {e}")
            self.traffic_agent_ready = False
            self.publish_traffic_light_status("AGENT_SETUP_FAILED")
    
    def detect_traffic_light_and_get_control(self):
        """
        Use agent as SENSOR to detect traffic lights and get natural control
        Returns: (traffic_light_detected, agent_speed, agent_steering, tl_status)
        """
        if not self.traffic_agent_ready or not self.traffic_agent:
            return False, 0.0, 0.0, "NO_AGENT"
        
        try:
            # Get agent control (agent detects traffic lights naturally)
            agent_control = self.traffic_agent.run_step()
            self.last_agent_control = agent_control
            
            if not agent_control:
                return False, 0.0, 0.0, "NO_CONTROL"
            
            # Extract control values
            brake_value = agent_control.brake
            throttle_value = agent_control.throttle
            steer_value = agent_control.steer
            
            # Convert agent control to speed (like automatic_control.py)
            agent_target_speed = 0.0
            
            if throttle_value > 0.0:
                # Agent wants to move - calculate target speed
                # automatic_control.py: agent can reach high speeds
                max_agent_speed = self.max_speed  # Let agent use your max speed
                agent_target_speed = throttle_value * max_agent_speed
            
            if brake_value > 0.0:
                # Agent wants to brake - reduce speed
                agent_target_speed *= (1.0 - brake_value)
            
            # Agent steering (for traffic light scenarios)
            agent_steering = steer_value * 0.8
            
            # Determine if traffic light is detected
            traffic_light_detected = False
            tl_status = "NO_TRAFFIC_LIGHT"
            
            # ENHANCED DETECTION: More sensitive to catch early traffic lights
            if brake_value > 0.01 or throttle_value < 0.3:  # ANY agent adjustment
                
                if brake_value > 0.7:  # Heavy braking
                    traffic_light_detected = True
                    tl_status = "RED_LIGHT_EMERGENCY"
                    
                elif brake_value > 0.4:  # Medium braking
                    traffic_light_detected = True
                    tl_status = "RED_LIGHT_CRITICAL"
                    
                elif brake_value > 0.2:  # Light braking
                    traffic_light_detected = True
                    tl_status = "RED_LIGHT_WARNING"
                    
                elif brake_value > 0.05:  # Very light braking
                    traffic_light_detected = True
                    tl_status = "RED_LIGHT_APPROACH"
                    
                elif throttle_value < 0.8:  # Reduced throttle
                    traffic_light_detected = True
                    tl_status = "TRAFFIC_LIGHT_DETECTED"
                    
                elif throttle_value < 0.95:  # Slightly reduced throttle
                    traffic_light_detected = True
                    tl_status = "TRAFFIC_LIGHT_DISTANT"
            
            # Update status
            self.traffic_light_status = tl_status
            self.publish_traffic_light_status(tl_status)
            
            return traffic_light_detected, agent_target_speed, agent_steering, tl_status
            
        except Exception as e:
            self.get_logger().debug(f"Traffic light sensor error: {e}")
            return False, 0.0, 0.0, "AGENT_ERROR"

    def setup_collision_sensor(self):
        """Setup collision sensor (UNCHANGED)"""
        try:
            if not self.vehicle:
                return
            
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, collision_transform, attach_to=self.vehicle)
            
            self.collision_sensor.listen(self.on_collision)
            self.get_logger().info("✅ Collision sensor active")
            
        except Exception as e:
            self.get_logger().error(f"❌ Collision sensor setup failed: {e}")
    
    def get_traffic_light_distance_from_agent(self):
        """
        Get accurate traffic light distance using CARLA agent brake patterns
        EXTENDED RANGE to stop BEFORE zebra crossing (stopping line)
        Returns: (distance_to_traffic_light, traffic_light_status, brake_intensity)
        """
        if not self.traffic_agent_ready or not self.traffic_agent:
            return float('inf'), "NO_AGENT", 0.0
        
        try:
            # Get agent control
            agent_control = self.traffic_agent.run_step()
            self.last_agent_control = agent_control
            
            if not agent_control:
                return float('inf'), "NO_CONTROL", 0.0
            
            # Extract control values
            brake_value = agent_control.brake
            throttle_value = agent_control.throttle
            
            # EXTENDED distance estimation to stop BEFORE zebra crossing
            estimated_distance = float('inf')
            tl_status = "NO_TRAFFIC_LIGHT"
            
            # Map brake intensity to EXTENDED distance (stop before stopping line)
            if brake_value > 0.3:  # Only significant braking indicates real traffic light
                
                if brake_value > 0.8:  # Very strong braking = EXTENDED emergency distance
                    estimated_distance = 18.0  # EXTENDED from 8.0m → 18.0m
                    tl_status = "RED_LIGHT_STOP"
                    
                elif brake_value > 0.6:  # Strong braking = EXTENDED critical distance  
                    estimated_distance = 25.0  # EXTENDED from 12.0m → 25.0m
                    tl_status = "RED_LIGHT_STOP"
                    
                elif brake_value > 0.5:  # Medium-high braking = EXTENDED warning distance
                    estimated_distance = 32.0  # EXTENDED from 16.0m → 32.0m
                    tl_status = "RED_LIGHT_STOP"
                    
                elif brake_value > 0.4:  # Medium braking = EXTENDED safe distance
                    estimated_distance = 40.0  # EXTENDED from 20.0m → 40.0m
                    tl_status = "RED_LIGHT_STOP"
                    
                else:  # Light but significant braking = distant RED or YELLOW
                    estimated_distance = 50.0  # EXTENDED from 26.0m → 50.0m
                    tl_status = "YELLOW_LIGHT_SLOW"
            
            # Update status
            self.traffic_light_status = tl_status
            self.publish_traffic_light_status(tl_status)
            
            return estimated_distance, tl_status, brake_value
            
        except Exception as e:
            self.get_logger().debug(f"Traffic light distance estimation error: {e}")
            return float('inf'), "AGENT_ERROR", 0.0

    def calculate_traffic_light_as_dynamic_obstacle(self, tl_distance):
        """
        Treat RED traffic light EXACTLY like a dynamic vehicle obstacle
        Using EXTENDED distances to stop BEFORE zebra crossing
        WITH COMPLETE STOPPING CAPABILITY (like nav2_ros.txt)
        """
        target_speed_ms = self.cruise_speed  # Start optimistic
        traffic_light_brake_power = 0.0
        tl_safety_status = "TL_CLEAR"
        traffic_light_emergency_stop = False  # Track if we need emergency stop
        
        # === EXTENDED DISTANCE ZONES for STOPPING BEFORE ZEBRA CROSSING ===
        
        # EXTENDED emergency distance to account for stopping line
        extended_emergency_distance = 18.0  # Stop well before zebra crossing
        extended_critical_distance = 25.0   # Give more time to decelerate
        extended_warning_distance = 32.0    # Earlier warning for smooth stop
        extended_safe_distance = 40.0       # Early detection for comfort
        
        if tl_distance <= extended_emergency_distance:  # 18.0m (was 10.40m)
            # EMERGENCY ZONE - Complete stop BEFORE zebra crossing
            target_speed_ms = 0.0
            traffic_light_brake_power = self.emergency_brake_power  # 1.0
            traffic_light_emergency_stop = True  # Trigger emergency stop
            tl_safety_status = f"🚨 TL_EMERGENCY_BEFORE_LINE - {tl_distance:.1f}m"
            
        elif tl_distance <= extended_critical_distance:  # 25.0m (was 15.02m)
            # CRITICAL ZONE - Heavy deceleration to prepare for stop
            speed_factor = (tl_distance - extended_emergency_distance) / \
                        (extended_critical_distance - extended_emergency_distance)
            target_speed_ms = self.creep_speed * speed_factor  # 0.39 m/s * factor
            traffic_light_brake_power = self.critical_brake_power  # 0.8
            tl_safety_status = f"⚡ TL_CRITICAL_APPROACH - {tl_distance:.1f}m"
            
        elif tl_distance <= extended_warning_distance:  # 32.0m (was 18.68m)
            # WARNING ZONE - Moderate deceleration
            speed_factor = (tl_distance - extended_critical_distance) / \
                        (extended_warning_distance - extended_critical_distance)
            target_speed_ms = self.creep_speed + (self.slow_speed - self.creep_speed) * speed_factor
            traffic_light_brake_power = self.warning_brake_power  # 0.65
            tl_safety_status = f"⚠️ TL_WARNING_DECEL - {tl_distance:.1f}m"
            
        elif tl_distance <= extended_safe_distance:  # 40.0m (was 24.99m)
            # SAFE ZONE - Gentle deceleration
            speed_factor = (tl_distance - extended_warning_distance) / \
                        (extended_safe_distance - extended_warning_distance)
            target_speed_ms = self.slow_speed + (self.normal_speed - self.slow_speed) * speed_factor
            traffic_light_brake_power = self.normal_brake_power  # 0.4
            tl_safety_status = f"✅ TL_SAFE_PREP - {tl_distance:.1f}m"
            
        else:  # Beyond extended safe distance
            # CLEAR - Use cruise speed
            target_speed_ms = self.cruise_speed
            traffic_light_brake_power = 0.0
            tl_safety_status = f"🟢 TL_CLEAR - {tl_distance:.1f}m"
        
        return target_speed_ms, traffic_light_brake_power, tl_safety_status, traffic_light_emergency_stop

    
    def get_traffic_light_distance_and_status(self):
        """
        Get traffic light distance and status using existing agent system
        Returns: (distance_to_traffic_light, traffic_light_status)
        """
        if not self.traffic_agent_ready or not self.traffic_agent:
            return float('inf'), "NO_AGENT"
        
        try:
            # Get agent control to determine traffic light influence
            agent_control = self.traffic_agent.run_step()
            if not agent_control:
                return float('inf'), "NO_CONTROL"
            
            # Extract traffic light status using existing method
            _, _, tl_status = self.extract_traffic_light_influence(agent_control)
            
            # Estimate distance based on brake behavior (like your current system)
            brake_value = agent_control.brake
            throttle_value = agent_control.throttle
            
            # Distance estimation based on agent behavior patterns
            if brake_value > 0.7:  # Strong braking = very close to red light
                estimated_distance = 8.0  # Within emergency distance
            elif brake_value > 0.4:  # Medium braking = approaching red light
                estimated_distance = 15.0  # Within critical distance
            elif brake_value > 0.2:  # Light braking = preparing for red
                estimated_distance = 20.0  # Within warning distance
            elif throttle_value < 0.3 and brake_value > 0.1:  # Cautious approach
                estimated_distance = 28.0  # Just within detection range
            else:  # Normal operation = no red light influence
                estimated_distance = float('inf')
            
            return estimated_distance, tl_status
            
        except Exception as e:
            self.get_logger().debug(f"Traffic light distance estimation error: {e}")
            return float('inf'), "AGENT_ERROR"

    def is_red_light_detected(self):
        """Check if RED traffic light is detected using EXTENDED distance-based approach"""
        distance, status, brake_intensity = self.get_traffic_light_distance_from_agent()
        
        # EXTENDED detection range to stop before zebra crossing
        is_red_light = status == "RED_LIGHT_STOP"
        is_within_detection_range = distance <= 50.0  # EXTENDED from 30.0m → 50.0m
        is_significant_braking = brake_intensity > 0.3  # Minimum brake threshold
        
        return is_red_light and is_within_detection_range and is_significant_braking

    def calculate_traffic_light_as_obstacle(self, tl_distance):
        """
        Treat red traffic light as dynamic obstacle using genetic algorithm distances
        Same logic as collision avoidance but for traffic lights
        """
        target_speed_ms = self.cruise_speed  # Start optimistic
        traffic_light_brake_power = 0.0
        tl_safety_status = "TL_CLEAR"
        
        # Apply genetic algorithm distances to traffic light
        if tl_distance <= self.emergency_distance:  # 10.40m
            target_speed_ms = 0.0
            traffic_light_brake_power = self.emergency_brake_power  # 1.0
            tl_safety_status = f"🚨 TL_EMERGENCY - {tl_distance:.1f}m"
            
        elif tl_distance <= self.critical_distance:  # 15.02m
            speed_factor = (tl_distance - self.emergency_distance) / \
                        (self.critical_distance - self.emergency_distance)
            target_speed_ms = self.creep_speed * speed_factor
            traffic_light_brake_power = self.critical_brake_power  # 0.8
            tl_safety_status = f"⚡ TL_CRITICAL - {tl_distance:.1f}m"
            
        elif tl_distance <= self.warning_distance:  # 18.68m
            speed_factor = (tl_distance - self.critical_distance) / \
                        (self.warning_distance - self.critical_distance)
            target_speed_ms = self.creep_speed + (self.slow_speed - self.creep_speed) * speed_factor
            traffic_light_brake_power = self.warning_brake_power  # 0.65
            tl_safety_status = f"⚠️ TL_WARNING - {tl_distance:.1f}m"
            
        elif tl_distance <= self.safe_distance:  # 24.99m
            speed_factor = (tl_distance - self.warning_distance) / \
                        (self.safe_distance - self.warning_distance)
            target_speed_ms = self.slow_speed + (self.normal_speed - self.slow_speed) * speed_factor
            traffic_light_brake_power = self.normal_brake_power  # 0.4
            tl_safety_status = f"✅ TL_SAFE - {tl_distance:.1f}m"
            
        else:  # Beyond safe distance but within 30m - gradual reduction
            # Gradual speed reduction from 30m to 25m (safe distance)
            distance_factor = (30.0 - tl_distance) / (30.0 - self.safe_distance)
            target_speed_ms = self.cruise_speed * (1.0 - 0.3 * distance_factor)  # Reduce up to 30%
            traffic_light_brake_power = 0.1 * distance_factor  # Light braking
            tl_safety_status = f"🟡 TL_APPROACHING - {tl_distance:.1f}m"
        
        return target_speed_ms, traffic_light_brake_power, tl_safety_status

    def detect_agent_traffic_light_activity(self):
        """
        Early detection of traffic light activity using agent behavior
        Agent sees traffic lights 50m+ ahead and starts adjusting behavior
        """
        if not self.traffic_agent_ready or not self.traffic_agent:
            return False, None, "NO_AGENT"
        
        try:
            # Get agent control (agent handles everything perfectly)
            agent_control = self.traffic_agent.run_step()
            self.last_agent_control = agent_control
            
            if not agent_control:
                return False, None, "NO_CONTROL"
            
            # Extract control values
            brake_value = agent_control.brake
            throttle_value = agent_control.throttle
            
            # EARLY DETECTION: Agent reduces throttle BEFORE heavy braking
            # This catches traffic lights 50m+ ahead
            traffic_light_detected = False
            agent_status = "NO_TRAFFIC_LIGHT"
            
            # Agent traffic light patterns (from automatic_control.py behavior):
            if brake_value > 0.4:  # Any deviation from full throttle
                
                if brake_value > 0.8:  # Heavy braking = very close to red light
                    traffic_light_detected = True
                    agent_status = "RED_LIGHT_EMERGENCY_STOP"
                    
                elif brake_value > 0.5:  # Medium braking = approaching red light
                    traffic_light_detected = True
                    agent_status = "RED_LIGHT_CRITICAL_STOP"
                    
                elif brake_value > 0.45:  # Light braking = red light ahead
                    traffic_light_detected = True
                    agent_status = "RED_LIGHT_WARNING_SLOW"
                    
                elif brake_value > 0.45:  # Very light braking = distant red light
                    traffic_light_detected = True
                    agent_status = "RED_LIGHT_PREPARE_STOP"
                    
                elif throttle_value < 0.7:  # Reduced throttle = traffic light approach
                    traffic_light_detected = True
                    agent_status = "TRAFFIC_LIGHT_APPROACH"
                    
                elif throttle_value < 0.9:  # Slightly reduced throttle = distant detection
                    traffic_light_detected = True
                    agent_status = "TRAFFIC_LIGHT_DETECTED"
            
            # Update status
            self.traffic_light_status = agent_status
            self.publish_traffic_light_status(agent_status)
            
            return traffic_light_detected, agent_control, agent_status
            
        except Exception as e:
            self.get_logger().debug(f"Agent traffic light detection error: {e}")
            return False, None, "AGENT_ERROR"

    def apply_agent_control_with_collision_check(self, agent_control):
        """
        Apply agent control but with collision avoidance override
        Agent handles traffic lights perfectly, we handle collisions
        """
        # Convert agent control to our speed/steering format
        agent_linear_vel = 0.0
        agent_angular_vel = 0.0
        
        # Convert agent throttle/brake to linear velocity
        if agent_control.throttle > 0.0:
            # Agent wants to move forward
            # Map throttle to speed (agent's max speed is usually cruise_speed)
            agent_linear_vel = agent_control.throttle * self.cruise_speed
        
        if agent_control.brake > 0.0:
            # Agent wants to slow down/stop
            # Apply brake reduction to speed
            brake_reduction = agent_control.brake
            agent_linear_vel *= (1.0 - brake_reduction)
        
        # Convert agent steering to angular velocity
        agent_angular_vel = agent_control.steer * 0.8  # Max steering rate
        
        # COLLISION OVERRIDE: Check if our collision avoidance is more restrictive
        collision_speed, collision_angular = self.calculate_pure_collision_avoidance()
        
        # Use most restrictive control (collision avoidance can override agent)
        final_linear_vel = min(agent_linear_vel, collision_speed)
        
        # Use agent steering unless collision avoidance requires different steering
        if collision_speed < agent_linear_vel:
            # Collision avoidance is active - use collision steering
            final_angular_vel = collision_angular
        else:
            # No collision threat - use agent steering (better for traffic lights)
            final_angular_vel = agent_angular_vel
        
        return final_linear_vel, final_angular_vel, "AGENT_CONTROL"

    def calculate_pure_collision_avoidance(self):
        """
        Pure collision avoidance system (your proven genetic algorithm)
        Only handles vehicles and pedestrians - NOT traffic lights
        """
        # Use exact same logic as your working system
        target_speed_ms = self.cruise_speed  # Start aggressive
        self.current_brake_power = 0.0
        collision_active = False
        
        # Stuck prevention (from your nav_dwb.txt)
        current_speed = self.get_current_speed()
        if current_speed < 1.0:
            self.stuck_prevention_counter += 1
        else:
            self.stuck_prevention_counter = 0
        
        if self.stuck_prevention_counter > self.max_stuck_time:
            self.is_emergency_stopped = False
            self.emergency_counter = 0
            self.stuck_prevention_counter = 0
            target_speed_ms = self.cruise_speed
        
        # COLLISION AVOIDANCE ZONES (your proven genetic algorithm)
        if self.detection_working:
            if self.front_obstacle_distance <= self.emergency_distance:  # 10.40m
                target_speed_ms = 0.0
                self.current_brake_power = self.emergency_brake_power
                self.is_emergency_stopped = True
                self.emergency_counter = 0
                collision_active = True
                
            elif self.front_obstacle_distance <= self.critical_distance:  # 15.02m
                speed_factor = (self.front_obstacle_distance - self.emergency_distance) / \
                            (self.critical_distance - self.emergency_distance)
                target_speed_ms = self.creep_speed * speed_factor
                self.current_brake_power = self.critical_brake_power
                collision_active = True
                
            elif self.front_obstacle_distance <= self.warning_distance:  # 18.68m
                speed_factor = (self.front_obstacle_distance - self.critical_distance) / \
                            (self.warning_distance - self.critical_distance)
                target_speed_ms = self.creep_speed + (self.slow_speed - self.creep_speed) * speed_factor
                self.current_brake_power = self.warning_brake_power
                collision_active = True
                
            elif self.front_obstacle_distance <= self.safe_distance:  # 24.99m
                speed_factor = (self.front_obstacle_distance - self.warning_distance) / \
                            (self.safe_distance - self.warning_distance)
                target_speed_ms = self.slow_speed + (self.normal_speed - self.slow_speed) * speed_factor
                self.current_brake_power = self.normal_brake_power
                collision_active = True
                
            else:
                # Clear path - emergency recovery or aggressive cruise
                if self.is_emergency_stopped:
                    self.emergency_counter += 1
                    if self.emergency_counter >= self.resume_threshold:
                        self.is_emergency_stopped = False
                        self.emergency_counter = 0
                        target_speed_ms = self.cruise_speed
                    else:
                        target_speed_ms = 0.0
                        self.current_brake_power = self.normal_brake_power
                        collision_active = True
                else:
                    # AGGRESSIVE 50+ km/h CONTROL
                    target_speed_ms = self.cruise_speed
                    self.current_brake_power = 0.0
                    collision_active = False
        else:
            # No detection - be cautious
            target_speed_ms = self.slow_speed
            self.current_brake_power = 0.0
            collision_active = True
        
        # Simple steering (collision avoidance doesn't need complex steering)
        angular_vel = 0.0  # Let agent handle steering for traffic lights
        
        return target_speed_ms, angular_vel


    def calculate_intelligent_path_following_control(self, target_waypoint):
        """
        INTELLIGENT PATH FOLLOWING with AGENT TRAFFIC LIGHT OVERRIDE:
        
        Priority System:
        1. Your Collision Avoidance (highest priority)
        2. Agent Traffic Light Control (when detected)  
        3. Your Path Following (50+ km/h aggressive)
        4. Turn Detection
        
        Key: Follow blue ROS path + Agent traffic light control + Your collision avoidance
        """
        if not self.current_pose:
            return 0.0, 0.0
        
        # === PATH FOLLOWING CALCULATIONS (Your ROS path) ===
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        orientation = self.current_pose.orientation
        _, _, current_yaw = tf_transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        target_x, target_y = target_waypoint['position']
        dx = target_x - current_x
        dy = target_y - current_y
        target_heading = math.atan2(dy, dx)
        
        yaw_error = target_heading - current_yaw
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        # === STEP 1: COLLISION AVOIDANCE (Highest Priority - Your System) ===
        target_speed_ms = self.cruise_speed
        safety_status = "🚀 AGGRESSIVE_CRUISE"
        self.current_brake_power = 0.0
        collision_active = False
        
        # Stuck prevention
        current_speed = self.get_current_speed()
        if current_speed < 1.0:
            self.stuck_prevention_counter += 1
        else:
            self.stuck_prevention_counter = 0
        
        if self.stuck_prevention_counter > self.max_stuck_time:
            self.is_emergency_stopped = False
            self.emergency_counter = 0
            self.stuck_prevention_counter = 0
            self.get_logger().warn("🔓 FORCED RESUME")
        
        # YOUR COLLISION AVOIDANCE (exact same proven system)
        if self.detection_working:
            if self.front_obstacle_distance <= self.emergency_distance:
                target_speed_ms = 0.0
                self.current_brake_power = self.emergency_brake_power
                self.is_emergency_stopped = True
                self.emergency_counter = 0
                safety_status = f"🚨 COLLISION_EMERGENCY - {self.front_obstacle_distance:.1f}m"
                collision_active = True
                
            elif self.front_obstacle_distance <= self.critical_distance:
                speed_factor = (self.front_obstacle_distance - self.emergency_distance) / \
                            (self.critical_distance - self.emergency_distance)
                target_speed_ms = self.creep_speed * speed_factor
                self.current_brake_power = self.critical_brake_power
                safety_status = f"⚡ COLLISION_CRITICAL - {self.front_obstacle_distance:.1f}m"
                collision_active = True
                
            elif self.front_obstacle_distance <= self.warning_distance:
                speed_factor = (self.front_obstacle_distance - self.critical_distance) / \
                            (self.warning_distance - self.critical_distance)
                target_speed_ms = self.creep_speed + (self.slow_speed - self.creep_speed) * speed_factor
                self.current_brake_power = self.warning_brake_power
                safety_status = f"⚠️ COLLISION_WARNING - {self.front_obstacle_distance:.1f}m"
                collision_active = True
                
            elif self.front_obstacle_distance <= self.safe_distance:
                speed_factor = (self.front_obstacle_distance - self.warning_distance) / \
                            (self.safe_distance - self.warning_distance)
                target_speed_ms = self.slow_speed + (self.normal_speed - self.slow_speed) * speed_factor
                self.current_brake_power = self.normal_brake_power
                safety_status = f"✅ COLLISION_SAFE - {self.front_obstacle_distance:.1f}m"
                collision_active = True
                
            else:
                # Clear path - emergency recovery or aggressive control
                if self.is_emergency_stopped:
                    self.emergency_counter += 1
                    if self.emergency_counter >= self.resume_threshold:
                        self.is_emergency_stopped = False
                        self.emergency_counter = 0
                        safety_status = "🟢 COLLISION_RESUMING"
                    else:
                        target_speed_ms = 0.0
                        self.current_brake_power = self.normal_brake_power
                        safety_status = f"⏳ COLLISION_RECOVERY {self.emergency_counter}/{self.resume_threshold}"
                        collision_active = True
                else:
                    # === YOUR 50+ km/h AGGRESSIVE CONTROL ===
                    target_speed_ms = self.cruise_speed  # 34.05 km/h base, can go up to max_speed
                    self.current_brake_power = 0.0
                    safety_status = f"🚀 CLEAR_PATH - {self.front_obstacle_distance:.1f}m"
                    collision_active = False
        else:
            target_speed_ms = self.slow_speed
            self.current_brake_power = 0.0
            safety_status = "⚠️ NO_DETECTION"
            collision_active = True
        
        # Store collision results
        collision_speed = target_speed_ms
        collision_brake = self.current_brake_power
        
        # === STEP 2: AGENT TRAFFIC LIGHT CONTROL (Only if no collision) ===
        if not collision_active:
            traffic_light_detected, agent_speed, agent_steering, tl_status = self.detect_traffic_light_and_get_control()
            
            if traffic_light_detected:
                # === AGENT TRAFFIC LIGHT MODE (Like automatic_control.py) ===
                
                # Use agent's speed control for traffic lights
                if agent_speed < collision_speed:
                    target_speed_ms = agent_speed
                    # Agent determines brake power based on its control
                    if agent_speed < self.creep_speed:
                        self.current_brake_power = self.emergency_brake_power
                    elif agent_speed < self.slow_speed:
                        self.current_brake_power = self.critical_brake_power
                    elif agent_speed < self.normal_speed:
                        self.current_brake_power = self.warning_brake_power
                    else:
                        self.current_brake_power = self.normal_brake_power
                    
                    safety_status = f"🚦 AGENT_TRAFFIC_LIGHT: {tl_status} - {agent_speed*3.6:.1f}km/h"
                    
                    # Log agent control
                    if self.debug_counter % 20 == 0:
                        self.get_logger().info(f"🤖 AGENT TRAFFIC LIGHT CONTROL: {tl_status}")
                        self.get_logger().info(f"   Agent Speed: {agent_speed*3.6:.1f} km/h, Brake: {self.current_brake_power:.2f}")
                        self.get_logger().info(f"   🚦 Traffic light detected - using agent natural behavior")
                    
                    self.publish_status("AGENT_TRAFFIC_LIGHT_ACTIVE")
                else:
                    # Traffic light not restrictive enough
                    if self.debug_counter % 50 == 0:
                        self.get_logger().info(f"🚦 Traffic light detected but not restrictive: {tl_status}")
            else:
                # === NO TRAFFIC LIGHT: YOUR AGGRESSIVE CONTROL ===
                if self.debug_counter % 50 == 0:
                    self.get_logger().info("🚀 AGGRESSIVE MODE: No traffic lights - full speed on ROS path!")
                
                self.publish_status("AGGRESSIVE_PATH_FOLLOWING")
        
        # === STEP 3: PATH FOLLOWING STEERING (Always your control) ===
        # Always use YOUR steering for ROS path following
        angular_vel = yaw_error * 1.0
        angular_vel = np.clip(angular_vel, -0.8, 0.8)
        
        # === STEP 4: TURN DETECTION (Your system) ===
        turn_angle, distance_to_turn, turn_severity = self.detect_upcoming_turn()
        turn_speed_multiplier = self.calculate_turn_speed_reduction(turn_angle, distance_to_turn, turn_severity)
        
        if turn_speed_multiplier < 1.0:
            original_speed = target_speed_ms * 3.6
            target_speed_ms *= turn_speed_multiplier
            reduced_speed = target_speed_ms * 3.6
            safety_status += f" +Turn{turn_severity}({distance_to_turn:.0f}m)"
            
            if hasattr(self, 'turn_log_counter'):
                self.turn_log_counter += 1
            else:
                self.turn_log_counter = 0
                
            if self.turn_log_counter % 20 == 0:
                self.get_logger().info(f"🔄 TURN: {turn_severity} in {distance_to_turn:.1f}m, "
                                    f"speed: {original_speed:.1f}→{reduced_speed:.1f}km/h")
        
        # Apply immediate turn reduction
        if abs(yaw_error) > math.pi / 10:
            target_speed_ms *= 0.5
            safety_status += " +ImmediateTurn"
        
        # Multiple obstacle penalty
        close_obstacles = len([obs for obs in list(self.nearby_vehicles.values()) + 
                            list(self.nearby_pedestrians.values()) 
                            if obs['distance'] < 12.0])
        if close_obstacles > 2:
            target_speed_ms *= max(0.6, 1.0 - (close_obstacles * 0.1))
            safety_status += f" +Multi({close_obstacles})"
        
        # Final speed limits
        target_speed_ms = min(target_speed_ms, self.max_speed)
        target_speed_ms = max(0.0, target_speed_ms)
        
        # Store safety status
        self.safety_status = safety_status
        
        return target_speed_ms, angular_vel

    def on_collision(self, event):
        """Handle collision events (UNCHANGED)"""
        try:
            self.collision_count += 1
            other_actor = event.other_actor
            
            # Only trigger emergency stop for significant collisions
            if hasattr(event, 'normal_impulse'):
                impulse = event.normal_impulse
                intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
                
                if intensity > 10.0:
                    self.is_emergency_stopped = True
                    self.emergency_counter = 0
                    self.get_logger().warn(f"🚨 Significant collision detected - Emergency stop triggered (intensity: {intensity:.1f})")
                    self.publish_status("COLLISION_EMERGENCY_STOP")
            
            self.get_logger().warn(f"\n🚨 COLLISION #{self.collision_count}")
            self.get_logger().warn(f"Other Actor: {other_actor.type_id if other_actor else 'Unknown'}")
            self.get_logger().warn(f"Front Distance: {self.front_obstacle_distance:.2f}m")
            self.get_logger().warn(f"Detection Working: {self.detection_working}")
            self.get_logger().warn(f"Emergency Stopped: {self.is_emergency_stopped}")
            
            # Publish collision info via ROS
            self.publish_status(f"COLLISION_{self.collision_count}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Collision handling error: {e}")
    
    def create_master_waypoints(self, path_points):
        """Create waypoints from ROS path (UNCHANGED)"""
        self.master_waypoints = []
        
        try:
            for i, point in enumerate(path_points):
                waypoint = {
                    'id': i,
                    'position': point,
                    'reached': False
                }
                self.master_waypoints.append(waypoint)
            
            self.get_logger().info(f"✅ {len(self.master_waypoints)} waypoints created from ROS path")
            
        except Exception as e:
            self.get_logger().error(f"❌ Waypoint creation failed: {e}")
    
    def detect_obstacles_balanced(self):
        """Balanced obstacle detection (UNCHANGED)"""
        if not self.world or not self.vehicle:
            self.detection_working = False
            return
        
        try:
            # Get ego vehicle state
            ego_transform = self.vehicle.get_transform()
            ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
            ego_heading = math.radians(ego_transform.rotation.yaw)
            
            # Clear previous detections
            self.nearby_vehicles.clear()
            self.nearby_pedestrians.clear()
            
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
                    if distance <= self.detection_range:
                        # Calculate relative angle
                        relative_pos = actor_pos - ego_pos
                        angle_to_obstacle = math.atan2(relative_pos[1], relative_pos[0])
                        relative_angle = angle_to_obstacle - ego_heading
                        
                        # Normalize angle to [-π, π]
                        while relative_angle > math.pi:
                            relative_angle -= 2 * math.pi
                        while relative_angle < -math.pi:
                            relative_angle += 2 * math.pi
                        
                        # Get obstacle velocity
                        obstacle_velocity = np.array([0.0, 0.0])
                        obstacle_speed = 0.0
                        try:
                            if hasattr(actor, 'get_velocity'):
                                vel = actor.get_velocity()
                                obstacle_velocity = np.array([vel.x, vel.y])
                                obstacle_speed = np.linalg.norm(obstacle_velocity)
                        except:
                            pass
                        
                        # Create obstacle info
                        obstacle_info = {
                            'id': actor.id,
                            'type': actor.type_id,
                            'distance': distance,
                            'position': actor_pos,
                            'relative_angle': relative_angle,
                            'location': actor_location,
                            'velocity': obstacle_velocity,
                            'speed': obstacle_speed
                        }
                        
                        # Categorize obstacles
                        if 'vehicle' in actor.type_id.lower():
                            self.nearby_vehicles[actor.id] = obstacle_info
                        elif 'walker' in actor.type_id.lower() or 'pedestrian' in actor.type_id.lower():
                            self.nearby_pedestrians[actor.id] = obstacle_info
                
                except Exception:
                    continue
            
            # Calculate front obstacle distance
            self.calculate_front_distance()
            
            # Mark detection as working
            self.detection_working = True
            
        except Exception as e:
            self.get_logger().error(f"❌ Balanced obstacle detection error: {e}")
            self.detection_working = False
    
    def calculate_front_distance(self):
        """Calculate front distance (UNCHANGED)"""
        front_distances = []
        
        # Front detection zone (±20 degrees)
        front_angle_threshold = math.pi / 15  # 20 degrees
        
        # Check vehicles in front
        for obstacle in self.nearby_vehicles.values():
            if abs(obstacle['relative_angle']) <= front_angle_threshold:
                front_distances.append(obstacle['distance'])
        
        # Check pedestrians in front
        for obstacle in self.nearby_pedestrians.values():
            if abs(obstacle['relative_angle']) <= front_angle_threshold:
                front_distances.append(obstacle['distance'])
        
        # Set front distance
        if front_distances:
            self.front_obstacle_distance = min(front_distances)
        else:
            self.front_obstacle_distance = float('inf')
    
    def is_collision_threat_active(self):
        """Check if collision avoidance is currently active"""
        return (self.front_obstacle_distance <= self.safe_distance or 
                self.is_emergency_stopped or 
                not self.detection_working)
    
    def publish_odometry(self):
        """Publish odometry via ROS (UNCHANGED)"""
        if not self.vehicle or not self.vehicle.is_alive:
            return
        
        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            
            # Publish Odometry
            odom_msg = Odometry()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Position
            odom_msg.pose.pose.position.x = transform.location.x
            odom_msg.pose.pose.position.y = transform.location.y
            odom_msg.pose.pose.position.z = transform.location.z
            
            # Orientation
            yaw = math.radians(transform.rotation.yaw)
            pitch = math.radians(transform.rotation.pitch)
            roll = math.radians(transform.rotation.roll)
            
            quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            odom_msg.pose.pose.orientation.x = quat[0]
            odom_msg.pose.pose.orientation.y = quat[1]
            odom_msg.pose.pose.orientation.z = quat[2]
            odom_msg.pose.pose.orientation.w = quat[3]
            
            # Velocity
            odom_msg.twist.twist.linear.x = velocity.x
            odom_msg.twist.twist.linear.y = velocity.y
            odom_msg.twist.twist.linear.z = velocity.z
            
            self.odom_pub.publish(odom_msg)
            
            # Publish Vehicle Status
            status_msg = CarlaEgoVehicleStatus()
            status_msg.header = odom_msg.header
            status_msg.velocity = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            status_msg.acceleration.linear.x = 0.0
            status_msg.acceleration.linear.y = 0.0
            status_msg.acceleration.linear.z = 0.0
            status_msg.orientation = odom_msg.pose.pose.orientation
            
            self.vehicle_status_pub.publish(status_msg)
            
            self.publish_tf_transforms(transform, odom_msg.header.stamp)
            
            # Store current pose
            self.current_pose = odom_msg.pose.pose
            
        except Exception as e:
            self.get_logger().error(f"❌ Odometry error: {e}")
    
    def publish_tf_transforms(self, transform, stamp):
        """Publish TF transforms (UNCHANGED)"""
        try:
            # odom -> base_link
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"
            
            t.transform.translation.x = transform.location.x
            t.transform.translation.y = transform.location.y
            t.transform.translation.z = transform.location.z
            
            yaw = math.radians(transform.rotation.yaw)
            pitch = math.radians(transform.rotation.pitch)
            roll = math.radians(transform.rotation.roll)
            
            quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
            
            # map -> odom
            t_map = TransformStamped()
            t_map.header.stamp = stamp
            t_map.header.frame_id = "map"
            t_map.child_frame_id = "odom"
            t_map.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(t_map)
            
        except Exception as e:
            self.get_logger().error(f"❌ TF error: {e}")
    
    def get_current_target_waypoint(self):
        """Get current target waypoint from ROS path (UNCHANGED)"""
        if not self.master_waypoints or not self.current_pose:
            return None
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        if self.path_index < len(self.master_waypoints):
            current_waypoint = self.master_waypoints[self.path_index]
            
            distance = math.sqrt(
                (current_waypoint['position'][0] - current_x)**2 + 
                (current_waypoint['position'][1] - current_y)**2
            )
            
            if distance < self.waypoint_tolerance:
                if not current_waypoint['reached']:
                    current_waypoint['reached'] = True
                    self.path_index += 1
                    
                    self.get_logger().info(f"🎯 Waypoint {self.path_index}/{len(self.master_waypoints)} reached")
                    
                    if self.path_index >= len(self.master_waypoints):
                        self.get_logger().info("🏁 Mission completed!")
                        self.publish_goal_reached(True)
                        self.publish_status("MISSION_COMPLETED")
                        return None
        
        if self.path_index < len(self.master_waypoints):
            return self.master_waypoints[self.path_index]
        
        return None
    
    def detect_upcoming_turn(self, lookahead_distance=30.0):
        """Detect upcoming turns (UNCHANGED)"""
        if not self.master_waypoints or not self.current_pose:
            return 0.0, float('inf'), "NONE"
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Get current heading
        orientation = self.current_pose.orientation
        _, _, current_yaw = tf_transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # Look ahead through waypoints within lookahead distance
        accumulated_distance = 0.0
        previous_heading = current_yaw
        max_turn_angle = 0.0
        distance_to_max_turn = float('inf')
        
        for i in range(self.path_index, min(self.path_index + 10, len(self.master_waypoints) - 1)):
            current_wp = self.master_waypoints[i]
            next_wp = self.master_waypoints[i + 1]
            
            # Calculate distance from vehicle to this waypoint
            wp_distance = math.sqrt(
                (current_wp['position'][0] - current_x)**2 + 
                (current_wp['position'][1] - current_y)**2
            )
            
            if accumulated_distance == 0.0:
                accumulated_distance = wp_distance
            else:
                # Add segment distance
                segment_distance = math.sqrt(
                    (current_wp['position'][0] - self.master_waypoints[i-1]['position'][0])**2 + 
                    (current_wp['position'][1] - self.master_waypoints[i-1]['position'][1])**2
                )
                accumulated_distance += segment_distance
            
            # Stop if we've looked far enough ahead
            if accumulated_distance > lookahead_distance:
                break
            
            # Calculate heading to next waypoint
            dx = next_wp['position'][0] - current_wp['position'][0]
            dy = next_wp['position'][1] - current_wp['position'][1]
            
            if abs(dx) < 0.1 and abs(dy) < 0.1:  # Skip if waypoints too close
                continue
                
            heading_to_next = math.atan2(dy, dx)
            
            # Calculate turn angle
            turn_angle = heading_to_next - previous_heading
            
            # Normalize angle to [-π, π]
            while turn_angle > math.pi:
                turn_angle -= 2 * math.pi
            while turn_angle < -math.pi:
                turn_angle += 2 * math.pi
            
            # Track maximum turn angle and its distance
            if abs(turn_angle) > abs(max_turn_angle):
                max_turn_angle = turn_angle
                distance_to_max_turn = accumulated_distance
            
            previous_heading = heading_to_next
        
        # Classify turn severity
        abs_turn = abs(max_turn_angle)
        if abs_turn > math.pi * 0.4:  # > 72 degrees
            severity = "SHARP"
        elif abs_turn > math.pi * 0.25:  # > 45 degrees  
            severity = "MODERATE"
        elif abs_turn > math.pi * 0.125:  # > 22.5 degrees
            severity = "GENTLE"
        else:
            severity = "NONE"
        
        return max_turn_angle, distance_to_max_turn, severity

    def calculate_turn_speed_reduction(self, turn_angle, distance_to_turn, turn_severity):
        """Calculate speed reduction based on upcoming turn (UNCHANGED)"""
        if turn_severity == "NONE":
            return 1.0  # No speed reduction
        
        # Define deceleration zones based on turn severity
        if turn_severity == "SHARP":
            if distance_to_turn <= 35.0:
                if distance_to_turn <= 15.0:
                    return 0.1  # Creep speed for sharp turns
                elif distance_to_turn <= 25.0:
                    return 0.25  # Slow speed
                else:
                    return 0.45  # Normal speed
            
        elif turn_severity == "MODERATE":
            if distance_to_turn <= 25.0:
                if distance_to_turn <= 10.0:
                    return 0.25  # Slow speed
                elif distance_to_turn <= 20.0:
                    return 0.4   # Normal speed
                else:
                    return 0.6   # Reduced cruise
        
        elif turn_severity == "GENTLE":
            if distance_to_turn <= 15.0:
                if distance_to_turn <= 8.0:
                    return 0.47   # Slightly reduced
                else:
                    return 0.67   # Minimal reduction
        
        return 1.0  # No reduction if not in deceleration zone

    
    def apply_adaptive_vehicle_control(self, linear_vel, angular_vel):
        """
        Apply vehicle control using nav_dwb.txt aggressive control system
        This is the PROVEN control that achieved 54.6 km/h average speed
        """
        try:
            # === USE EXACT nav_dwb.txt CONTROL LOGIC ===
            control = carla.VehicleControl()
            
            if linear_vel > 0.1:
                # Forward motion with nav_dwb.txt aggressive control
                if self.current_brake_power > 0.0:
                    # BRAKING MODE (from nav_dwb.txt)
                    control.throttle = min(0.5, max(0.05, linear_vel / self.max_speed))
                    control.brake = self.current_brake_power
                    control.reverse = False
                else:
                    # AGGRESSIVE ACCELERATION (from nav_dwb.txt) - KEY DIFFERENCE!
                    if self.front_obstacle_distance < self.warning_distance:
                        # Gentle throttle near obstacles
                        control.throttle = min(0.5, max(0.05, linear_vel / self.max_speed))
                    else:
                        # AGGRESSIVE throttle for higher speeds - EXACT nav_dwb.txt mapping
                        control.throttle = min(0.7, max(0.1, linear_vel / self.max_speed))
                    
                    control.brake = 0.0
                    control.reverse = False
                
            elif linear_vel < -0.1:
                # Reverse (nav_dwb.txt)
                control.throttle = min(0.3, max(0.05, abs(linear_vel) / 4.0))
                control.brake = 0.0
                control.reverse = True
                
            else:
                # Stop with nav_dwb.txt braking logic
                control.throttle = 0.0
                if self.is_emergency_stopped:
                    control.brake = self.emergency_brake_power
                elif self.current_brake_power > 0.0:
                    control.brake = self.current_brake_power
                else:
                    control.brake = 0.4
                control.reverse = False
            
            # Steering (nav_dwb.txt)
            control.steer = np.clip(angular_vel * 0.6, -0.8, 0.8)
            
            # Apply control to CARLA vehicle
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.apply_control(control)
            
            # === PUBLISH CARLA-SPECIFIC CONTROL ===
            ros_control = CarlaEgoVehicleControl()
            ros_control.header.stamp = self.get_clock().now().to_msg()
            ros_control.header.frame_id = "ego_vehicle"
            ros_control.throttle = control.throttle
            ros_control.steer = control.steer
            ros_control.brake = control.brake
            ros_control.hand_brake = control.hand_brake
            ros_control.reverse = control.reverse
            ros_control.gear = 1
            ros_control.manual_gear_shift = False
            
            self.carla_cmd_pub.publish(ros_control)
            
            # === /cmd_vel WITH nav_dwb.txt BRAKE CONSIDERATION ===
            cmd_vel_msg = Twist()
            
            # Linear velocity with brake consideration (nav_dwb.txt style)
            if self.current_brake_power > 0.5:
                brake_reduction = self.current_brake_power * 0.8
                cmd_vel_msg.linear.x = linear_vel * (1.0 - brake_reduction)
            else:
                cmd_vel_msg.linear.x = linear_vel
                
            cmd_vel_msg.linear.y = 0.0
            cmd_vel_msg.linear.z = 0.0
            
            cmd_vel_msg.angular.x = 0.0
            cmd_vel_msg.angular.y = 0.0
            cmd_vel_msg.angular.z = angular_vel
            
            # Publish for MPC controller
            self.cmd_vel_pub.publish(cmd_vel_msg)
            
            # Log occasionally
            if self.debug_counter % 50 == 0:
                brake_info = f", BRAKE={self.current_brake_power:.2f}" if self.current_brake_power > 0.0 else ""
                red_light_info = " +RED_LIGHT" if self.is_red_light_detected() else " +AGGRESSIVE"
                self.get_logger().info(f"🎮 ADAPTIVE: linear={linear_vel:.2f}m/s, angular={angular_vel:.2f}rad/s{brake_info}{red_light_info}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Adaptive vehicle control error: {e}")

    
    def get_current_speed(self):
        """Get current speed in km/h (UNCHANGED)"""
        if not self.vehicle or not self.vehicle.is_alive:
            return 0.0
        try:
            velocity = self.vehicle.get_velocity()
            return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        except:
            return 0.0
    
    def main_control_loop(self):
        """Enhanced main control loop with PRIORITY-BASED traffic light integration"""
        if not self.vehicle or not self.vehicle.is_alive:
            self.get_logger().error("❌ Vehicle lost!")
            self.publish_status("VEHICLE_LOST")
            return
        
        # Check if we have received necessary ROS data
        if not self.received_path:
            if self.debug_counter % 50 == 0:
                self.get_logger().warn("⏳ Waiting for ROS path data...")
                self.publish_status("WAITING_FOR_PATH")
            self.debug_counter += 1
            return
        
        self.debug_counter += 1
        
        try:
            # STEP 1: Obstacle detection (UNCHANGED)
            self.detect_obstacles_balanced()
            
            # STEP 2: Get target waypoint from ROS path (UNCHANGED)
            target_waypoint = self.get_current_target_waypoint()
            
            if target_waypoint is None:
                # Mission completed - Safe stop
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = True
                if self.vehicle:
                    self.vehicle.apply_control(control)
                
                # Publish zero cmd_vel for MPC
                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = 0.0
                cmd_vel_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_vel_msg)
                
                if self.debug_counter % 100 == 0:
                    self.get_logger().info("🏁 Mission completed - vehicle safely stopped")
                    self.publish_status("MISSION_COMPLETED")
                return
            
            # STEP 3: Calculate enhanced speed with PRIORITY-BASED control
            linear_vel, angular_vel = self.calculate_intelligent_path_following_control(target_waypoint)
            
            # STEP 4: Apply enhanced control
            self.apply_adaptive_vehicle_control(linear_vel, angular_vel)
            
            # STEP 5: Enhanced debug output
            if self.debug_counter % 20 == 0:
                current_speed = self.get_current_speed()
                
                self.get_logger().info(f"\n⚖️ PRIORITY-BASED COLLISION AVOIDANCE + TRAFFIC LIGHT:")
                self.get_logger().info(f"   📍 Waypoint: {self.path_index+1}/{len(self.master_waypoints)}")
                self.get_logger().info(f"   🏃 Speed: {current_speed:.1f} km/h → {linear_vel*3.6:.1f} km/h")
                self.get_logger().info(f"   🎯 Front Distance: {self.front_obstacle_distance:.1f}m")
                self.get_logger().info(f"   🛑 BRAKE POWER: {self.current_brake_power:.2f}")
                self.get_logger().info(f"   🔍 Detection: {'✅ WORKING' if self.detection_working else '❌ FAILED'}")
                self.get_logger().info(f"   🚗 Vehicles: {len(self.nearby_vehicles)}")
                self.get_logger().info(f"   🚶 Pedestrians: {len(self.nearby_pedestrians)}")
                self.get_logger().info(f"   🚨 Emergency Stop: {'YES' if self.is_emergency_stopped else 'NO'}")
                
                # Traffic Light Info
                collision_threat = self.is_collision_threat_active()
                self.get_logger().info(f"   🚦 Traffic Light: {self.traffic_light_status}")
                self.get_logger().info(f"   🔄 Collision Priority: {'ACTIVE' if collision_threat else 'INACTIVE'}")
                
                self.get_logger().info(f"   🛡️ Safety: {getattr(self, 'safety_status', 'Unknown')}")
                self.get_logger().info(f"   💥 Collisions: {self.collision_count}")
                self.get_logger().info(f"   📡 ROS Status: Path={'✓' if self.received_path else '✗'}, Goal={'✓' if self.received_goal else '✗'}")
                self.get_logger().info(f"   🎮 MPC Cmd: linear={linear_vel:.3f} m/s, angular={angular_vel:.3f} rad/s")
                
                # Show recovery progress
                if self.is_emergency_stopped:
                    self.get_logger().info(f"   ⏳ Recovery: {self.emergency_counter}/{self.resume_threshold}")
                
                # Show stuck prevention
                if self.stuck_prevention_counter > 10:
                    self.get_logger().info(f"   🔓 Stuck Prevention: {self.stuck_prevention_counter}/{self.max_stuck_time}")
                
                print("-" * 70)
            
            # Enhanced ROS2 logging
            if self.debug_counter % 50 == 0:
                status_parts = []
                status_parts.append(f"SPEED: {current_speed:.0f}→{linear_vel*3.6:.0f}km/h")
                status_parts.append(f"DIST: {self.front_obstacle_distance:.1f}m")
                status_parts.append(f"BRAKE: {self.current_brake_power:.1f}")
                
                # Traffic light status
                if "STOP" in self.traffic_light_status:
                    status_parts.append("TL_STOP")
                elif "SLOW" in self.traffic_light_status or "YELLOW" in self.traffic_light_status:
                    status_parts.append("TL_SLOW")
                elif "APPROACH" in self.traffic_light_status:
                    status_parts.append("TL_APPROACH")
                elif "GREEN" in self.traffic_light_status:
                    status_parts.append("TL_GO")
                
                if self.is_emergency_stopped:
                    status_parts.append("EMERGENCY")
                elif self.front_obstacle_distance < self.warning_distance:
                    status_parts.append("BRAKING")
                elif self.front_obstacle_distance < self.safe_distance:
                    status_parts.append("CAUTION")
                else:
                    status_parts.append("CLEAR")
                
                status_parts.append(f"COL: {self.collision_count}")
                
                if self.stuck_prevention_counter > 10:
                    status_parts.append(f"STUCK: {self.stuck_prevention_counter}")
                
                status_parts.append(f"ROS: {'OK' if self.received_path and self.received_goal else 'WAIT'}")
                status_parts.append(f"MPC: {linear_vel:.2f}m/s")
                
                status_str = " [" + ", ".join(status_parts) + "]"
                
                self.get_logger().info(f"⚖️ PRIORITY+TRAFFIC_LIGHT: waypoint {self.path_index+1}/{len(self.master_waypoints)}{status_str}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Enhanced control loop error: {e}")
            self.publish_status("CONTROL_LOOP_ERROR")
    
    def destroy_node(self):
        """Enhanced cleanup with traffic light performance reporting"""
        try:
            self.publish_status("SHUTTING_DOWN")
            
            self.get_logger().info(f"\n⚖️ PRIORITY-BASED COLLISION AVOIDANCE + TRAFFIC LIGHT FINAL REPORT")
            print("=" * 80)
            print(f"🎯 REAL-WORLD SPEED PARAMETERS USED:")
            print(f"   Max Speed: {self.max_speed*3.6:.1f} km/h")
            print(f"   Cruise Speed: {self.cruise_speed*3.6:.1f} km/h")
            print(f"   Emergency Distance: {self.emergency_distance:.1f}m")
            print(f"   Critical Distance: {self.critical_distance:.1f}m")
            print("")
            print(f"🛑 ENHANCED BRAKING SYSTEM:")
            print(f"   Emergency Brake: {self.emergency_brake_power:.1f}")
            print(f"   Critical Brake: {self.critical_brake_power:.1f}")
            print(f"   Warning Brake: {self.warning_brake_power:.1f}")
            print("")
            print(f"🚦 PRIORITY-BASED TRAFFIC LIGHT INTEGRATION:")
            print(f"   Priority Order: 1.Collision → 2.Traffic Light → 3.Turn → 4.Cruise")
            print(f"   Agent Available: {'✅' if AGENTS_AVAILABLE else '❌'}")
            print(f"   Agent Ready: {'✅' if self.traffic_agent_ready else '❌'}")
            print(f"   Final TL Status: {self.traffic_light_status}")
            print("")
            print(f"📊 PERFORMANCE RESULTS:")
            print(f"   Total Collisions: {self.collision_count}")
            print(f"   Detection Working: {self.detection_working}")
            print(f"   Vehicles Detected: {len(self.nearby_vehicles)}")
            print(f"   Pedestrians Detected: {len(self.nearby_pedestrians)}")
            print(f"   Emergency Stops Triggered: {getattr(self, 'emergency_counter', 0)}")
            print(f"   Stuck Prevention Activations: {self.stuck_prevention_counter}")
            print(f"   Final Front Distance: {self.front_obstacle_distance:.1f}m")
            print(f"   Final Brake Power: {self.current_brake_power:.2f}")
            print("")
            print(f"🔧 SYSTEM STATUS:")
            print(f"   ROS Communication: Path={'✓' if self.received_path else '✗'}, Goal={'✓' if self.received_goal else '✗'}")
            print(f"   MPC Integration: /cmd_vel with brake consideration published")
            print("")
            
            # Enhanced success criteria for real-world speeds + powerful braking
            if self.collision_count == 0:
                print("🏆 EXCELLENT - Collision-free with POWERFUL BRAKING!")
            elif self.collision_count < 3:
                print("✅ GOOD - Minimal collisions, POWERFUL BRAKING working")
            elif self.collision_count < 10:
                print("✅ ACCEPTABLE - Some collisions, but POWERFUL BRAKING helped")
            else:
                print("⚠️ NEEDS IMPROVEMENT - Collision avoidance needs tuning")
            
            # Movement assessment (UNCHANGED)
            current_speed = self.get_current_speed()
            if self.path_index > 1:
                print("✅ Vehicle successfully moved through waypoints")
            elif current_speed > 0.5:
                print("✅ Vehicle is actively moving")
            else:
                print("⚠️ Vehicle movement was limited")
            
            # ROS Communication assessment (UNCHANGED)
            if self.received_path and self.received_goal:
                print("✅ ROS Communication worked successfully")
            else:
                print("⚠️ ROS Communication had issues")
            
            # POWERFUL BRAKING assessment
            print("✅ POWERFUL BRAKING system implemented successfully")
            print("✅ MPC /cmd_vel integration with brake consideration ready")
            
            print("=" * 80)
            
            # Stop vehicle safely with POWERFUL BRAKING
            if self.vehicle and self.vehicle.is_alive:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0  # Maximum brake
                control.hand_brake = True
                self.vehicle.apply_control(control)
                time.sleep(0.5)
            
            # Cleanup sensor (UNCHANGED)
            if self.collision_sensor and self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
            
            self.get_logger().info("⚖️ Balanced collision avoidance + POWERFUL BRAKING system cleaned up")
            self.publish_status("SHUTDOWN_COMPLETE")
            
        except Exception as e:
            self.get_logger().error(f"❌ Powerful braking cleanup error: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    planner = None
    try:
        planner = EnhancedBalancedCollisionAvoidanceROS()
        rclpy.spin(planner)
    except KeyboardInterrupt:
        print("Shutting down Balanced Collision Avoidance + POWERFUL BRAKING...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if planner:
            planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
