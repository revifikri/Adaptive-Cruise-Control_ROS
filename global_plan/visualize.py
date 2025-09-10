#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor

import carla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
import time
from collections import deque
import threading

# ROS2 Messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class CarlaPathVisualizer(Node):
    def __init__(self):
        super().__init__('carla_path_visualizer')
        
        # QoS Profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.global_path_sub = self.create_subscription(
            Path, '/planned_path', self.global_path_callback, 10)
        self.local_path_sub = self.create_subscription(
            Path, '/dwa_local_path', self.local_path_callback, 10)
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/carla/ego_vehicle/lidar',
            self.lidar_callback, sensor_qos)
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        self.town_map = None
        
        # Visualization data with thread safety
        self.data_lock = threading.Lock()
        self.start_point = None
        self.goal_point = None
        self.global_path = []
        self.local_path = []
        self.vehicle_position = None
        self.vehicle_heading = 0.0
        self.obstacles = []
        self.vehicle_trail = deque(maxlen=200)  # Vehicle trajectory
        self.traffic_lights = []
        self.other_vehicles = []
        
        # Connect to CARLA
        self.connect_to_carla()
        
        # Timer for updating data
        self.timer = self.create_timer(0.1, self.update_data)  # 10 Hz
        
        self.get_logger().info("CARLA Path Visualizer initialized")
    
    def connect_to_carla(self):
        """Connect to CARLA and find the vehicle"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.town_map = self.world.get_map()
            
            # Find ego vehicle
            actors = self.world.get_actors()
            for actor in actors:
                if ('vehicle' in actor.type_id and 
                    hasattr(actor, 'attributes') and 
                    actor.attributes.get('role_name') == 'ego_vehicle'):
                    self.vehicle = actor
                    break
            
            if not self.vehicle:
                # Fallback: use any vehicle
                vehicles = self.world.get_actors().filter('vehicle.*')
                if vehicles:
                    self.vehicle = vehicles[0]
                    self.get_logger().warn(f"Using fallback vehicle: {self.vehicle.type_id}")
            
            if self.vehicle:
                self.get_logger().info(f"Found vehicle: {self.vehicle.type_id}")
                # Get initial position as start point
                transform = self.vehicle.get_transform()
                with self.data_lock:
                    self.start_point = (transform.location.x, transform.location.y)
            else:
                self.get_logger().warn("No vehicle found")
                
        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA: {e}")
    
    def global_path_callback(self, msg):
        """Receive global path"""
        new_path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            new_path.append((x, y))
        
        with self.data_lock:
            self.global_path = new_path
            if self.global_path:
                # Set goal point as the last waypoint
                self.goal_point = self.global_path[-1]
                
        self.get_logger().info(f"Received global path with {len(new_path)} waypoints")
    
    def local_path_callback(self, msg):
        """Receive local path from DWA"""
        new_path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            new_path.append((x, y))
        
        with self.data_lock:
            self.local_path = new_path
    
    def lidar_callback(self, msg):
        """Process LiDAR for obstacle visualization"""
        if not self.vehicle:
            return
        
        try:
            # Get vehicle position and orientation
            vehicle_transform = self.vehicle.get_transform()
            vehicle_x = vehicle_transform.location.x
            vehicle_y = vehicle_transform.location.y
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            
            # Extract points from point cloud
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            
            obstacles = []
            for point in points:
                x_lidar, y_lidar, z_lidar = point
                
                # Height filtering for ground-level obstacles
                if -0.5 < z_lidar < 3.0:
                    # Transform to world coordinates
                    cos_yaw = math.cos(vehicle_yaw)
                    sin_yaw = math.sin(vehicle_yaw)
                    
                    world_x = vehicle_x + (x_lidar * cos_yaw - y_lidar * sin_yaw)
                    world_y = vehicle_y + (x_lidar * sin_yaw + y_lidar * cos_yaw)
                    
                    # Only consider nearby obstacles for visualization
                    distance = math.sqrt(x_lidar*x_lidar + y_lidar*y_lidar)
                    if 2.0 < distance < 30.0:
                        obstacles.append((world_x, world_y))
            
            # Cluster obstacles for cleaner visualization
            clustered_obstacles = self.cluster_obstacles(obstacles)
            
            with self.data_lock:
                self.obstacles = clustered_obstacles
            
        except Exception as e:
            self.get_logger().warn(f"Error processing LiDAR: {e}")
    
    def cluster_obstacles(self, obstacles, cluster_distance=3.0):
        """Cluster nearby obstacles"""
        if not obstacles:
            return []
        
        clusters = []
        used = [False] * len(obstacles)
        
        for i, obs in enumerate(obstacles):
            if used[i]:
                continue
            
            cluster = [obs]
            used[i] = True
            
            # Find nearby points
            for j, other_obs in enumerate(obstacles):
                if used[j]:
                    continue
                
                distance = math.sqrt((obs[0] - other_obs[0])**2 + (obs[1] - other_obs[1])**2)
                if distance < cluster_distance:
                    cluster.append(other_obs)
                    used[j] = True
            
            # Only keep substantial clusters
            if len(cluster) >= 3:
                center_x = sum(point[0] for point in cluster) / len(cluster)
                center_y = sum(point[1] for point in cluster) / len(cluster)
                clusters.append((center_x, center_y))
        
        return clusters
    
    def update_data(self):
        """Update all visualization data"""
        if not self.vehicle:
            return
        
        try:
            # Update vehicle position
            transform = self.vehicle.get_transform()
            new_position = (transform.location.x, transform.location.y)
            new_heading = math.radians(transform.rotation.yaw)
            
            # Update other vehicles and traffic lights
            other_vehicles = []
            traffic_lights = []
            
            if self.vehicle:
                vehicle_loc = self.vehicle.get_location()
                
                # Get other vehicles
                for vehicle in self.world.get_actors().filter('vehicle.*'):
                    if vehicle.id != self.vehicle.id:
                        loc = vehicle.get_location()
                        distance = math.sqrt((loc.x - vehicle_loc.x)**2 + (loc.y - vehicle_loc.y)**2)
                        if distance < 80:  # Within 80m
                            other_vehicles.append((loc.x, loc.y))
                
                # Get traffic lights
                for tl in self.world.get_actors().filter('traffic.traffic_light*'):
                    loc = tl.get_location()
                    distance = math.sqrt((loc.x - vehicle_loc.x)**2 + (loc.y - vehicle_loc.y)**2)
                    if distance < 80:  # Within 80m
                        state = tl.get_state()
                        color = 'red' if state == carla.TrafficLightState.Red else \
                               'yellow' if state == carla.TrafficLightState.Yellow else 'green'
                        traffic_lights.append((loc.x, loc.y, color))
            
            with self.data_lock:
                self.vehicle_position = new_position
                self.vehicle_heading = new_heading
                self.other_vehicles = other_vehicles
                self.traffic_lights = traffic_lights
                
                # Add to trail (trajectory)
                self.vehicle_trail.append(new_position)
                
                # Set start point if not set
                if self.start_point is None:
                    self.start_point = new_position
            
        except Exception as e:
            self.get_logger().warn(f"Error updating data: {e}")
    
    def get_visualization_data(self):
        """Get thread-safe copy of visualization data"""
        with self.data_lock:
            return {
                'start_point': self.start_point,
                'goal_point': self.goal_point,
                'global_path': self.global_path.copy(),
                'local_path': self.local_path.copy(),
                'vehicle_position': self.vehicle_position,
                'vehicle_heading': self.vehicle_heading,
                'obstacles': self.obstacles.copy(),
                'vehicle_trail': list(self.vehicle_trail),
                'traffic_lights': self.traffic_lights.copy(),
                'other_vehicles': self.other_vehicles.copy()
            }


class VisualizationWindow:
    def __init__(self, visualizer_node):
        self.node = visualizer_node
        
        # Set up matplotlib with proper backend
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
        
        # Matplotlib setup
        plt.style.use('default')
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Set window title using the figure manager (fixed approach)
        try:
            self.fig.canvas.manager.set_window_title('CARLA Path Visualization')
        except AttributeError:
            # Fallback: set title through figure
            self.fig.suptitle('CARLA Path Visualization', fontsize=16)
        
        # Configure plot style
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title('Vehicle Path Visualization', fontsize=14)
        self.ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Animation
        self.animation = FuncAnimation(self.fig, self.animate, interval=100, blit=False, cache_frame_data=False)
    
    def animate(self, frame):
        """Animation function for matplotlib"""
        # Get current data from the node
        data = self.node.get_visualization_data()
        
        self.ax.clear()
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title('Vehicle Path Visualization', fontsize=14)
        self.ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Plot vehicle trajectory (orange line)
        if len(data['vehicle_trail']) > 1:
            trail_x = [point[0] for point in data['vehicle_trail']]
            trail_y = [point[1] for point in data['vehicle_trail']]
            self.ax.plot(trail_x, trail_y, 'orange', linewidth=2, label='Vehicle Trajectory', alpha=0.8)
        
        # Plot global path as waypoints (blue line)
        if len(data['global_path']) > 1:
            global_x = [point[0] for point in data['global_path']]
            global_y = [point[1] for point in data['global_path']]
            self.ax.plot(global_x, global_y, 'blue', linewidth=3, label='Global Path')
            
            # Mark individual waypoints as small dots
            self.ax.plot(global_x, global_y, 'bo', markersize=4, alpha=0.6)
        
        # Plot local path (if available, cyan line)
        if len(data['local_path']) > 1:
            local_x = [point[0] for point in data['local_path']]
            local_y = [point[1] for point in data['local_path']]
            self.ax.plot(local_x, local_y, 'cyan', linewidth=2, alpha=0.8, label='Local Path')
        
        # Plot start point (green circle)
        if data['start_point']:
            self.ax.plot(data['start_point'][0], data['start_point'][1], 'go', 
                        markersize=10, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Plot current vehicle position (red circle with direction arrow)
        if data['vehicle_position']:
            x, y = data['vehicle_position']
            self.ax.plot(x, y, 'ro', markersize=10, label='Current Position', 
                        markeredgecolor='darkred', markeredgewidth=2)
            
            # Draw direction arrow
            if data['vehicle_heading'] is not None:
                arrow_length = 8.0
                dx = arrow_length * math.cos(data['vehicle_heading'])
                dy = arrow_length * math.sin(data['vehicle_heading'])
                self.ax.arrow(x, y, dx, dy, head_width=2.0, head_length=2.0, 
                            fc='red', ec='red', alpha=0.8)
        
        # Plot goal point (red star)
        if data['goal_point'] and data['goal_point'] != data['start_point']:
            self.ax.plot(data['goal_point'][0], data['goal_point'][1], 'r*', 
                        markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=1)
        
        # Plot other vehicles (black squares)
        if data['other_vehicles']:
            vehicle_x = [v[0] for v in data['other_vehicles']]
            vehicle_y = [v[1] for v in data['other_vehicles']]
            self.ax.plot(vehicle_x, vehicle_y, 'ks', markersize=8, label=f'Other Vehicles ({len(data["other_vehicles"])})', 
                        alpha=0.7, markeredgecolor='white', markeredgewidth=1)
        
        # Plot traffic lights (colored stars)
        traffic_light_colors = {}
        if data['traffic_lights']:
            for tl_x, tl_y, color in data['traffic_lights']:
                if color not in traffic_light_colors:
                    traffic_light_colors[color] = {'x': [], 'y': []}
                traffic_light_colors[color]['x'].append(tl_x)
                traffic_light_colors[color]['y'].append(tl_y)
            
            for color, coords in traffic_light_colors.items():
                self.ax.plot(coords['x'], coords['y'], '^', color=color, markersize=12, 
                           markeredgecolor='black', markeredgewidth=1, 
                           label=f'Traffic Light ({color.title()})')
        
        # Plot obstacles (dark gray circles)
        if data['obstacles']:
            obs_x = [obs[0] for obs in data['obstacles']]
            obs_y = [obs[1] for obs in data['obstacles']]
            self.ax.plot(obs_x, obs_y, 'o', color='darkgray', markersize=8, 
                        alpha=0.8, label=f'Obstacles ({len(data["obstacles"])})',
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Set axis limits with some margin
        if data['vehicle_position'] or data['global_path']:
            all_x = []
            all_y = []
            
            if data['vehicle_position']:
                all_x.append(data['vehicle_position'][0])
                all_y.append(data['vehicle_position'][1])
            
            if data['global_path']:
                all_x.extend([p[0] for p in data['global_path']])
                all_y.extend([p[1] for p in data['global_path']])
            
            if data['vehicle_trail']:
                all_x.extend([p[0] for p in data['vehicle_trail']])
                all_y.extend([p[1] for p in data['vehicle_trail']])
            
            if all_x and all_y:
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                margin = max(40.0, max(x_range, y_range) * 0.15)
                
                self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Add legend (remove duplicates)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), 
                      loc='upper left', framealpha=0.9, fontsize=10, 
                      bbox_to_anchor=(1.02, 1))
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal', adjustable='box')
        
        # Tight layout to prevent legend cutoff
        plt.tight_layout()
    
    def show(self):
        """Show the visualization window"""
        plt.show(block=False)
        plt.draw()
        return self.fig


def main(args=None):
    rclpy.init(args=args)
    
    # Create the visualizer node
    visualizer = CarlaPathVisualizer()
    
    # Create executor for ROS
    executor = SingleThreadedExecutor()
    executor.add_node(visualizer)
    
    # Start ROS spinning in a separate thread
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    try:
        # Create and show visualization window in main thread
        viz_window = VisualizationWindow(visualizer)
        fig = viz_window.show()
        
        print("CARLA Path Visualizer running...")
        print("The visualization window should be open.")
        print("Close the matplotlib window or press Ctrl+C to exit.")
        
        # Keep the window open with proper event handling
        import signal
        import sys
        
        def signal_handler(sig, frame):
            print("\nShutting down...")
            plt.close('all')
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Main event loop - keep window responsive
        while True:
            try:
                # Process matplotlib events
                if plt.get_backend() == 'TkAgg':
                    fig.canvas.flush_events()
                plt.pause(0.01)  # Small pause to allow GUI updates
                
                # Check if window was closed
                if not plt.get_fignums():
                    print("Window closed, shutting down...")
                    break
                    
            except KeyboardInterrupt:
                print("\nInterrupt received, shutting down...")
                break
            except Exception as e:
                print(f"Event loop error: {e}")
                # Continue running even if there's a minor error
                time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("Shutting down CARLA Path Visualizer...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        try:
            executor.shutdown()
            visualizer.destroy_node()
            rclpy.shutdown()
        except:
            pass
        plt.close('all')

if __name__ == '__main__':
    main()
