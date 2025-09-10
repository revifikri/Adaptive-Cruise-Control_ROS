#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
import os

def generate_launch_description():
    # Arguments
    use_ros_communication_arg = DeclareLaunchArgument(
        'use_ros_communication',
        default_value='true',
        description='Use improved ROS communication between planners'
    )
    
    spawn_traffic_arg = DeclareLaunchArgument(
        'spawn_traffic',
        default_value='true',
        description='Spawn traffic vehicles for testing'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug nodes and extra logging'
    )
    
    # === ROS COMMUNICATION ENHANCED NODES ===
    
    # Enhanced global planner with full ROS communication
    ros_global_planner = Node(
        package='global_plan',
        executable='global_plan_node_ros',  # New ROS-enabled executable
        name='global_plan_node_ros',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'min_clearance': 8.0,
            'validation_radius': 12.0,
            'min_distance_between_points': 80.0,
            'max_distance_between_points': 250.0,
            'waypoint_spacing': 5.0,
            'publish_rate': 1.0,
            'max_path_publications': 5
        }],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Enhanced local planner with full ROS communication
    ros_local_planner = Node(
        package='global_plan',
        executable='nav_dwb_ros',  # New ROS-enabled executable
        name='nav_dwb_ros',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            # Balanced collision avoidance parameters
            'emergency_distance': 7.0,
            'critical_distance': 15.0,
            'warning_distance': 20.0,
            'safe_distance': 25.0,
            # Speed parameters
            'max_speed': 17.0,
            'cruise_speed': 11.0,
            'normal_speed': 7.0,
            'slow_speed': 5.0,
            'creep_speed': 1.5,
            # Recovery parameters
            'resume_threshold': 15,
            'max_stuck_time': 50,
            # Detection parameters
            'detection_range': 100.0,
            'min_obstacle_distance': 1.5,
            'waypoint_tolerance': 5.0,
            # Control loop rate
            'control_loop_rate': 10.0,  # 10Hz
            'odometry_rate': 20.0       # 20Hz
        }],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Legacy planners (fallback - original system)
    legacy_global_planner = Node(
        package='global_plan',
        executable='global_plan_node',  # Original executable
        name='global_plan_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'min_clearance': 8.0,
            'validation_radius': 12.0
        }],
        condition=UnlessCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    legacy_local_planner = Node(
        package='global_plan',
        executable='nav_dwb',  # Original executable
        name='nav_dwb',
        output='screen',
        parameters=[{'use_sim_time': False}],
        condition=UnlessCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # === ROS COMMUNICATION MONITORING NODES ===
    
    # ROS Topic Monitor (monitors communication between planners)
    ros_topic_monitor = Node(
        package='global_plan',
        executable='ros_topic_monitor',
        name='ros_topic_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'monitor_topics': [
                '/planned_path',
                '/goal_pose', 
                '/odom',
                '/carla/ego_vehicle/vehicle_control_cmd',
                '/planner_status',
                '/local_planner_status',
                '/goal_reached'
            ],
            'monitor_rate': 2.0
        }],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Communication Health Monitor
    communication_health = Node(
        package='global_plan',
        executable='communication_health',
        name='communication_health',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'timeout_threshold': 5.0,  # seconds
            'health_check_rate': 1.0
        }],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Enhanced visualization with ROS communication data
    ros_visualization_node = Node(
        package='global_plan',
        executable='visualize_ros',
        name='carla_path_visualizer_ros',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'show_communication_status': True,
            'show_safety_zones': True,
            'show_obstacle_detection': True
        }],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Legacy visualization
    legacy_visualization_node = Node(
        package='global_plan',
        executable='visualize',
        name='carla_path_visualizer',
        output='screen',
        parameters=[{'use_sim_time': False}],
        condition=UnlessCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Debug and monitoring node with ROS communication awareness
    debug_node = Node(
        package='global_plan',
        executable='debug_topics_ros',
        name='debug_topics_ros',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'show_ros_communication': True,
            'detailed_logging': LaunchConfiguration('debug_mode')
        }],
        condition=IfCondition(LaunchConfiguration('debug_mode'))
    )
    
    # === ENHANCED TRAFFIC GENERATION ===
    
    # Generate diverse traffic for realistic testing
    traffic_generator = ExecuteProcess(
        cmd=['python3', '/home/av/CARLA_0.9.13/PythonAPI/examples/generate_traffic.py', 
             '--number-of-vehicles', '35',  # More vehicles for testing
             '--number-of-walkers', '8',     # More pedestrians
             '--safe',                       # Safe spawning
             '--asynch',                     # Asynchronous mode
             '--hero',                       # Respect hero vehicle
             '--tm-port', '8000',            # Traffic manager port
             '--car-lights-on'],             # Realistic lighting
        output='screen',
        condition=IfCondition(LaunchConfiguration('spawn_traffic'))
    )
    
    # === INTELLIGENT ROS COMMUNICATION STARTUP SEQUENCE ===
    
    # Stage 1: Start global planner first (immediate)
    # This creates the vehicle and initial ROS topics
    
    # Stage 2: Generate traffic after a short delay
    delayed_traffic = TimerAction(
        period=3.0,
        actions=[traffic_generator],
        condition=IfCondition(LaunchConfiguration('spawn_traffic'))
    )
    
    # Stage 3: Start ROS communication monitoring
    delayed_ros_monitor = TimerAction(
        period=5.0,
        actions=[ros_topic_monitor],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Stage 4: Start local planner after global planner has time to publish path
    delayed_ros_local = TimerAction(
        period=8.0,  # Give time for global planner to create path and publish
        actions=[ros_local_planner],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    delayed_legacy_local = TimerAction(
        period=8.0,
        actions=[legacy_local_planner],
        condition=UnlessCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Stage 5: Start communication health monitoring
    delayed_comm_health = TimerAction(
        period=10.0,
        actions=[communication_health],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Stage 6: Start visualization after navigation is running
    delayed_ros_visualization = TimerAction(
        period=12.0,
        actions=[ros_visualization_node],
        condition=IfCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    delayed_legacy_visualization = TimerAction(
        period=12.0,
        actions=[legacy_visualization_node],
        condition=UnlessCondition(LaunchConfiguration('use_ros_communication'))
    )
    
    # Stage 7: Start debugging/monitoring last
    delayed_debug = TimerAction(
        period=15.0,
        actions=[debug_node],
        condition=IfCondition(LaunchConfiguration('debug_mode'))
    )
    
    return LaunchDescription([
        # Arguments
        use_ros_communication_arg,
        spawn_traffic_arg,
        debug_mode_arg,
        
        # Immediate starts - Global Planners
        ros_global_planner,           # ROS-enabled version
        legacy_global_planner,        # Original version
        
        # Delayed starts in intelligent sequence
        delayed_traffic,              # Traffic generation
        delayed_ros_monitor,          # ROS topic monitoring
        delayed_ros_local,            # ROS-enabled local planner
        delayed_legacy_local,         # Original local planner
        delayed_comm_health,          # Communication health
        delayed_ros_visualization,    # ROS-aware visualization
        delayed_legacy_visualization, # Original visualization
        delayed_debug                 # Debug monitoring
    ])
