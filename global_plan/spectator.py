#!/usr/bin/env python3

import carla
import time
import math
import numpy as np
import argparse
import sys
from collections import deque
from enum import Enum
import threading
import json

try:
    import pygame
    from pygame.locals import K_q, K_ESCAPE, K_c, K_r, K_s, K_h, K_1, K_2, K_3, K_4, K_5
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: Pygame not available. Some features will be disabled.")

class CameraMode(Enum):
    """Camera modes for spectator"""
    FOLLOW_BEHIND = "Behind Vehicle"
    FOLLOW_ABOVE = "Above Vehicle"
    CHASE_CAM = "Chase Camera"
    SIDE_VIEW = "Side View"
    BIRD_EYE = "Bird's Eye View"

class VehicleMonitor:
    def __init__(self, args):
        self.args = args
        
        # CARLA connection
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.spectator = None
        self.map = None
        
        # Vehicle tracking
        self.vehicle_history = deque(maxlen=100)  # Position history
        self.speed_history = deque(maxlen=50)     # Speed history
        self.collision_history = []
        self.collision_sensor = None
        
        # Navigation monitoring
        self.global_waypoints = []
        self.current_waypoint_index = 0
        self.navigation_target = None
        self.path_deviation = 0.0
        
        # Obstacle tracking
        self.nearby_vehicles = {}
        self.nearby_pedestrians = {}
        self.static_obstacles = {}
        self.obstacle_history = deque(maxlen=30)
        
        # Camera control
        self.camera_mode = CameraMode.FOLLOW_BEHIND
        self.camera_distance = 8.0
        self.camera_height = 3.5
        self.camera_pitch = -15.0
        self.auto_camera = True
        
        # Display and UI
        self.display_info = True
        self.console_output = True
        self.save_data = args.save_data if hasattr(args, 'save_data') else False
        
        # Performance metrics
        self.start_time = time.time()
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.collision_count = 0
        self.stuck_time = 0.0
        self.last_stuck_check = time.time()
        
        # Update frequencies
        self.update_interval = 0.05  # 20 FPS for smooth camera
        self.status_interval = 1.0   # 1 Hz for status logging
        self.last_status_time = 0
        
        # Threading
        self.running = True
        self.status_thread = None
        
        # Initialize pygame if available
        if PYGAME_AVAILABLE and not args.no_pygame:
            self.init_pygame()
        else:
            self.pygame_enabled = False
    
    def init_pygame(self):
        """Initialize pygame for keyboard input"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 200))
            pygame.display.set_caption("Vehicle Monitor - Press H for help")
            self.pygame_enabled = True
            self.font = pygame.font.Font(None, 24)
        except Exception as e:
            print(f"Failed to initialize pygame: {e}")
            self.pygame_enabled = False
    
    def connect_to_carla(self):
        """Connect to CARLA and find ego vehicle"""
        try:
            print("🔌 Connecting to CARLA...")
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self.map = self.world.get_map()
            
            print("✅ Connected to CARLA successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to CARLA: {e}")
            return False
    
    def find_ego_vehicle(self):
        """Find the ego vehicle in the world"""
        print("🔍 Searching for ego vehicle...")
        
        search_attempts = 0
        max_attempts = 50
        
        while search_attempts < max_attempts and self.running:
            try:
                # First, try to find vehicle with ego role
                actors = self.world.get_actors()
                for actor in actors:
                    if ('vehicle' in actor.type_id and 
                        hasattr(actor, 'attributes') and 
                        actor.attributes.get('role_name') == 'ego_vehicle'):
                        self.ego_vehicle = actor
                        print(f"✅ Found ego vehicle: {actor.type_id} (ID: {actor.id})")
                        self.setup_collision_sensor()
                        return True
                
                # Fallback: look for Tesla Model 3
                tesla_vehicles = self.world.get_actors().filter('*model3*')
                if tesla_vehicles:
                    self.ego_vehicle = tesla_vehicles[0]
                    print(f"✅ Found Tesla Model 3: {self.ego_vehicle.type_id} (ID: {self.ego_vehicle.id})")
                    self.setup_collision_sensor()
                    return True
                
                # Last resort: any vehicle
                if search_attempts > 30:
                    all_vehicles = self.world.get_actors().filter('vehicle.*')
                    if all_vehicles:
                        self.ego_vehicle = all_vehicles[0]
                        print(f"⚠️ Using fallback vehicle: {self.ego_vehicle.type_id} (ID: {self.ego_vehicle.id})")
                        self.setup_collision_sensor()
                        return True
                
                search_attempts += 1
                print(f"🔍 Searching... (attempt {search_attempts}/{max_attempts})")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error during vehicle search: {e}")
                time.sleep(1.0)
        
        print("❌ Could not find ego vehicle")
        return False
    
    def setup_collision_sensor(self):
        """Setup collision sensor for the ego vehicle"""
        try:
            if not self.ego_vehicle:
                return
            
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, collision_transform, attach_to=self.ego_vehicle)
            
            # Setup collision callback
            self.collision_sensor.listen(self.on_collision)
            print("🛡️ Collision sensor setup complete")
            
        except Exception as e:
            print(f"Failed to setup collision sensor: {e}")
    
    def on_collision(self, event):
        """Handle collision events"""
        try:
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            
            if intensity > 5.0:  # Filter minor collisions
                collision_info = {
                    'time': time.time(),
                    'intensity': intensity,
                    'other_actor': event.other_actor.type_id if event.other_actor else 'unknown',
                    'location': (event.transform.location.x, event.transform.location.y),
                    'impulse': (impulse.x, impulse.y, impulse.z)
                }
                
                self.collision_history.append(collision_info)
                self.collision_count += 1
                
                # Console output for collision
                print(f"\n💥 COLLISION #{self.collision_count}!")
                print(f"   🎯 Target: {collision_info['other_actor']}")
                print(f"   💪 Intensity: {intensity:.2f}")
                print(f"   📍 Location: ({collision_info['location'][0]:.1f}, {collision_info['location'][1]:.1f})")
        
        except Exception as e:
            print(f"Error handling collision: {e}")
    
    def update_vehicle_status(self):
        """Update comprehensive vehicle status with CORRECTED average speed calculation"""
        if not self.ego_vehicle or not self.ego_vehicle.is_alive:
            return False
        
        try:
            # Get vehicle state
            transform = self.ego_vehicle.get_transform()
            velocity = self.ego_vehicle.get_velocity()
            control = self.ego_vehicle.get_control()
            
            # Calculate metrics
            position = (transform.location.x, transform.location.y, transform.location.z)
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
            
            # Store history
            current_time = time.time()
            self.vehicle_history.append({
                'time': current_time,
                'position': position,
                'speed': speed_ms,
                'heading': transform.rotation.yaw
            })
            
            self.speed_history.append(speed_ms)
            
            # Update max speed
            if speed_kmh > self.max_speed:
                self.max_speed = speed_kmh
            
            # === CORRECTED: Calculate total distance traveled ===
            if len(self.vehicle_history) >= 2:
                prev_pos = self.vehicle_history[-2]['position']
                curr_pos = self.vehicle_history[-1]['position']
                distance_increment = math.sqrt(
                    (curr_pos[0] - prev_pos[0])**2 + 
                    (curr_pos[1] - prev_pos[1])**2
                )
                self.total_distance += distance_increment
            
            # === CORRECTED: Calculate TRUE average speed (distance-based) ===
            session_duration = current_time - self.start_time
            if session_duration > 0.1 and self.total_distance > 0:
                # TRUE average speed = Total Distance / Total Time
                self.avg_speed = (self.total_distance / session_duration) * 3.6  # Convert m/s to km/h
            else:
                self.avg_speed = 0.0
            
            # Check if stuck
            if speed_ms < 0.5:  # Very slow or stopped
                if not hasattr(self, 'stuck_start_time'):
                    self.stuck_start_time = current_time
                else:
                    self.stuck_time = current_time - self.stuck_start_time
            else:
                if hasattr(self, 'stuck_start_time'):
                    delattr(self, 'stuck_start_time')
                self.stuck_time = 0.0
            
            return True
            
        except Exception as e:
            print(f"Error updating vehicle status: {e}")
            return False
    
    def detect_nearby_obstacles(self):
        """Detect and track nearby obstacles"""
        if not self.ego_vehicle:
            return
        
        try:
            ego_transform = self.ego_vehicle.get_transform()
            ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
            
            # Clear previous detections
            self.nearby_vehicles.clear()
            self.nearby_pedestrians.clear()
            
            # Get all actors
            all_actors = self.world.get_actors()
            
            for actor in all_actors:
                if actor.id == self.ego_vehicle.id:
                    continue
                
                try:
                    actor_location = actor.get_location()
                    actor_pos = np.array([actor_location.x, actor_location.y])
                    distance = np.linalg.norm(actor_pos - ego_pos)
                    
                    # Only track nearby obstacles (within 50m)
                    if distance <= 50.0:
                        obstacle_info = {
                            'id': actor.id,
                            'type': actor.type_id,
                            'distance': distance,
                            'position': actor_pos,
                            'location': actor_location
                        }
                        
                        # Get velocity if available
                        if hasattr(actor, 'get_velocity'):
                            velocity = actor.get_velocity()
                            obstacle_info['velocity'] = np.array([velocity.x, velocity.y])
                            obstacle_info['speed'] = np.linalg.norm(obstacle_info['velocity'])
                        else:
                            obstacle_info['velocity'] = np.array([0.0, 0.0])
                            obstacle_info['speed'] = 0.0
                        
                        # Calculate relative angle
                        relative_pos = actor_pos - ego_pos
                        ego_heading = math.radians(ego_transform.rotation.yaw)
                        relative_angle = math.atan2(relative_pos[1], relative_pos[0]) - ego_heading
                        
                        # Normalize angle
                        while relative_angle > math.pi:
                            relative_angle -= 2 * math.pi
                        while relative_angle < -math.pi:
                            relative_angle += 2 * math.pi
                        
                        obstacle_info['relative_angle'] = relative_angle
                        
                        # Categorize obstacle
                        if 'vehicle' in actor.type_id:
                            self.nearby_vehicles[actor.id] = obstacle_info
                        elif 'walker' in actor.type_id:
                            self.nearby_pedestrians[actor.id] = obstacle_info
                        elif 'static' in actor.type_id:
                            self.static_obstacles[actor.id] = obstacle_info
                
                except Exception:
                    continue
            
            # Store obstacle snapshot for history
            obstacle_snapshot = {
                'time': time.time(),
                'vehicles': len(self.nearby_vehicles),
                'pedestrians': len(self.nearby_pedestrians),
                'static': len(self.static_obstacles)
            }
            self.obstacle_history.append(obstacle_snapshot)
            
        except Exception as e:
            print(f"Error detecting obstacles: {e}")
    
    def update_camera(self):
        """Update spectator camera position"""
        if not self.ego_vehicle or not self.spectator:
            return
        
        try:
            transform = self.ego_vehicle.get_transform()
            yaw_rad = math.radians(transform.rotation.yaw)
            
            if self.camera_mode == CameraMode.FOLLOW_BEHIND:
                # Behind and above the vehicle
                offset = carla.Location(
                    x=-self.camera_distance * math.cos(yaw_rad),
                    y=-self.camera_distance * math.sin(yaw_rad),
                    z=self.camera_height
                )
                spectator_transform = carla.Transform(
                    transform.location + offset,
                    carla.Rotation(pitch=self.camera_pitch, yaw=transform.rotation.yaw)
                )
            
            elif self.camera_mode == CameraMode.FOLLOW_ABOVE:
                # Directly above the vehicle
                offset = carla.Location(x=0, y=0, z=15.0)
                spectator_transform = carla.Transform(
                    transform.location + offset,
                    carla.Rotation(pitch=-90, yaw=transform.rotation.yaw)
                )
            
            elif self.camera_mode == CameraMode.CHASE_CAM:
                # Dynamic chase camera with smooth movement
                velocity = self.ego_vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                dynamic_distance = max(6.0, min(15.0, speed * 0.5))
                
                offset = carla.Location(
                    x=-dynamic_distance * math.cos(yaw_rad),
                    y=-dynamic_distance * math.sin(yaw_rad),
                    z=self.camera_height + speed * 0.1
                )
                spectator_transform = carla.Transform(
                    transform.location + offset,
                    carla.Rotation(pitch=self.camera_pitch - speed * 2, yaw=transform.rotation.yaw)
                )
            
            elif self.camera_mode == CameraMode.SIDE_VIEW:
                # Side view camera
                offset = carla.Location(
                    x=-3.0 * math.sin(yaw_rad),
                    y=3.0 * math.cos(yaw_rad),
                    z=2.0
                )
                spectator_transform = carla.Transform(
                    transform.location + offset,
                    carla.Rotation(pitch=-10, yaw=transform.rotation.yaw + 90)
                )
            
            elif self.camera_mode == CameraMode.BIRD_EYE:
                # High bird's eye view
                offset = carla.Location(x=0, y=0, z=50.0)
                spectator_transform = carla.Transform(
                    transform.location + offset,
                    carla.Rotation(pitch=-90, yaw=0)
                )
            
            self.spectator.set_transform(spectator_transform)
            
        except Exception as e:
            print(f"Error updating camera: {e}")
    
    def handle_pygame_events(self):
        """Handle pygame keyboard events"""
        if not self.pygame_enabled:
            return True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_q or event.key == K_ESCAPE:
                    return False
                elif event.key == K_h:
                    self.print_help()
                elif event.key == K_1:
                    self.camera_mode = CameraMode.FOLLOW_BEHIND
                    print(f"📷 Camera mode: {self.camera_mode.value}")
                elif event.key == K_2:
                    self.camera_mode = CameraMode.FOLLOW_ABOVE
                    print(f"📷 Camera mode: {self.camera_mode.value}")
                elif event.key == K_3:
                    self.camera_mode = CameraMode.CHASE_CAM
                    print(f"📷 Camera mode: {self.camera_mode.value}")
                elif event.key == K_4:
                    self.camera_mode = CameraMode.SIDE_VIEW
                    print(f"📷 Camera mode: {self.camera_mode.value}")
                elif event.key == K_5:
                    self.camera_mode = CameraMode.BIRD_EYE
                    print(f"📷 Camera mode: {self.camera_mode.value}")
                elif event.key == K_c:
                    self.console_output = not self.console_output
                    print(f"📺 Console output: {'ON' if self.console_output else 'OFF'}")
                elif event.key == K_r:
                    self.reset_statistics()
                elif event.key == K_s:
                    self.save_monitoring_data()
        
        return True
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*60)
        print("🚗 VEHICLE MONITOR - KEYBOARD CONTROLS")
        print("="*60)
        print("Camera Controls:")
        print("  1 - Follow Behind Camera")
        print("  2 - Follow Above Camera") 
        print("  3 - Dynamic Chase Camera")
        print("  4 - Side View Camera")
        print("  5 - Bird's Eye View")
        print("\nMonitoring Controls:")
        print("  C - Toggle console output")
        print("  R - Reset statistics")
        print("  S - Save monitoring data")
        print("  H - Show this help")
        print("  Q/ESC - Quit")
        print("="*60 + "\n")
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.start_time = time.time()
        self.total_distance = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.collision_count = 0
        self.stuck_time = 0.0
        self.collision_history.clear()
        self.vehicle_history.clear()
        self.speed_history.clear()
        print("📊 Statistics reset")
    
    def save_monitoring_data(self):
        """Save monitoring data to file with CORRECTED metrics"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"vehicle_monitor_{timestamp}.json"
            
            # Calculate final corrected average speed
            session_duration = time.time() - self.start_time
            if session_duration > 0.1 and self.total_distance > 0:
                corrected_avg_speed = (self.total_distance / session_duration) * 3.6
            else:
                corrected_avg_speed = 0.0
            
            data = {
                'session_info': {
                    'start_time': self.start_time,
                    'duration': session_duration,
                    'total_distance': self.total_distance,
                    'max_speed': self.max_speed,
                    'corrected_avg_speed': corrected_avg_speed,  # CORRECTED
                    'collision_count': self.collision_count,
                    'calculation_method': 'distance_based_average'  # Documentation
                },
                'collision_history': self.collision_history,
                'vehicle_history': list(self.vehicle_history)[-50:],  # Last 50 points
                'obstacle_history': list(self.obstacle_history)[-30:] # Last 30 points
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Data saved to {filename}")
            print(f"📊 Corrected Average Speed: {corrected_avg_speed:.1f} km/h")
            
        except Exception as e:
            print(f"❌ Failed to save data: {e}")
    
    def log_status(self):
        """Log comprehensive vehicle status with CORRECTED metrics"""
        if not self.console_output:
            return
        
        try:
            current_time = time.time()
            session_duration = current_time - self.start_time
            
            # Get current vehicle state
            if not self.ego_vehicle or not self.ego_vehicle.is_alive:
                print("❌ Vehicle not available")
                return
            
            transform = self.ego_vehicle.get_transform()
            velocity = self.ego_vehicle.get_velocity()
            control = self.ego_vehicle.get_control()
            
            speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_ms * 3.6
            
            # Clear screen and print header
            print("\n" + "="*80)
            print(f"🚗 VEHICLE MONITOR - Session Time: {session_duration:.1f}s")
            print("="*80)
            
            # Vehicle Status
            print(f"📍 Position: ({transform.location.x:.1f}, {transform.location.y:.1f}, {transform.location.z:.1f})")
            print(f"🧭 Heading: {transform.rotation.yaw:.1f}°")
            print(f"🏃 Speed: {speed_kmh:.1f} km/h ({speed_ms:.1f} m/s)")
            
            # Control Status
            if hasattr(control, 'throttle'):
                print(f"🎮 Control: Throttle={control.throttle:.2f}, Brake={control.brake:.2f}, Steer={control.steer:.2f}")
                if control.reverse:
                    print("   ⏪ REVERSE")
                if control.hand_brake:
                    print("   🤚 HANDBRAKE")
            
            # === CORRECTED: Performance Metrics with TRUE average speed ===
            print(f"📊 Statistics:")
            print(f"   📏 Total Distance: {self.total_distance:.1f}m")
            print(f"   ⚡ Max Speed: {self.max_speed:.1f} km/h")
            print(f"   📈 TRUE Avg Speed: {self.avg_speed:.1f} km/h")  # Now matches metrics_collector.py
            
            # === NEW: Additional useful metrics ===
            if session_duration > 0:
                print(f"   🕒 Session Duration: {session_duration:.1f}s")
                print(f"   📊 Distance Rate: {(self.total_distance/session_duration)*3.6:.1f} km/h")
            
            print(f"   💥 Collisions: {self.collision_count}")
            
            # Stuck Detection
            if self.stuck_time > 3.0:
                print(f"   ⚠️ STUCK for {self.stuck_time:.1f}s")
            
            # Obstacle Status
            total_obstacles = len(self.nearby_vehicles) + len(self.nearby_pedestrians)
            if total_obstacles > 0:
                print(f"🚧 Nearby Obstacles: {total_obstacles} total")
                print(f"   🚗 Vehicles: {len(self.nearby_vehicles)}")
                print(f"   🚶 Pedestrians: {len(self.nearby_pedestrians)}")
                
                # Show closest obstacles
                all_obstacles = list(self.nearby_vehicles.values()) + list(self.nearby_pedestrians.values())
                if all_obstacles:
                    closest = min(all_obstacles, key=lambda x: x['distance'])
                    angle_deg = math.degrees(closest['relative_angle'])
                    print(f"   📏 Closest: {closest['type']} at {closest['distance']:.1f}m, {angle_deg:.1f}°")
            
            # Recent Collisions
            if self.collision_history:
                recent_collisions = [c for c in self.collision_history if current_time - c['time'] < 30.0]
                if recent_collisions:
                    print(f"💥 Recent Collisions (last 30s): {len(recent_collisions)}")
                    for collision in recent_collisions[-3:]:  # Show last 3
                        time_ago = current_time - collision['time']
                        print(f"   {collision['other_actor']} ({time_ago:.1f}s ago, intensity: {collision['intensity']:.1f})")
            
            # Camera Info
            print(f"📷 Camera: {self.camera_mode.value}")
            
            print("="*80)
            print("Press H for help, Q to quit")
            
        except Exception as e:
            print(f"Error logging status: {e}")
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        print("🚀 Starting vehicle monitoring...")
        
        last_update = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Handle pygame events
                if not self.handle_pygame_events():
                    break
                
                # Update vehicle status
                if not self.update_vehicle_status():
                    print("❌ Lost connection to vehicle")
                    break
                
                # Detect obstacles
                self.detect_nearby_obstacles()
                
                # Update camera
                if self.auto_camera:
                    self.update_camera()
                
                # Log status periodically
                if current_time - self.last_status_time > self.status_interval:
                    self.log_status()
                    self.last_status_time = current_time
                
                # Sleep to maintain update rate
                elapsed = time.time() - current_time
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def cleanup(self):
        """Cleanup resources with CORRECTED final report"""
        try:
            self.running = False
            
            # Destroy collision sensor
            if self.collision_sensor and self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
            
            # Save final data if requested
            if self.save_data:
                self.save_monitoring_data()
            
            # === CORRECTED: Final statistics calculation ===
            session_duration = time.time() - self.start_time
            
            # Recalculate final average speed to ensure accuracy
            if session_duration > 0.1 and self.total_distance > 0:
                final_avg_speed = (self.total_distance / session_duration) * 3.6
            else:
                final_avg_speed = 0.0
            
            print("\n" + "="*60)
            print("📊 FINAL MONITORING REPORT (CORRECTED METRICS)")
            print("="*60)
            print(f"⏱️ Session Duration: {session_duration:.1f}s")
            print(f"📏 Total Distance: {self.total_distance:.1f}m")
            '''print(f"⚡ Max Speed: {self.max_speed:.1f} km/h")
            print(f"📈 TRUE Average Speed: {final_avg_speed:.1f} km/h") ''' # CORRECTED
            print(f"💥 Total Collisions: {self.collision_count}")
            
            # === NEW: Performance assessment ===
            print(f"")
            print(f"📋 PERFORMANCE ASSESSMENT:")
            print(f"   Distance Rate: {self.total_distance/session_duration:.1f} m/s")
            if session_duration > 60:  # Only show efficiency for longer sessions
                expected_distance = final_avg_speed * (session_duration/3600) * 1000  # Expected distance in meters
                if expected_distance > 0:
                    efficiency = (self.total_distance / expected_distance) * 100
                    print(f"   Movement Efficiency: {efficiency:.1f}%")
            
            # Collision assessment
            if self.collision_count == 0:
                print("✅ COLLISION-FREE SESSION!")
            elif self.collision_count < 5:
                print("⚠️ Session completed with minor collisions")
            else:
                print("❌ Session had significant collision issues")
            
            # Speed assessment  
            if final_avg_speed >= 30.0:
                print("✅ EXCELLENT SPEED PERFORMANCE - Achieved highway-like speeds")
            elif final_avg_speed >= 20.0:
                print("✅ GOOD SPEED PERFORMANCE - Achieved urban driving speeds")
            elif final_avg_speed >= 10.0:
                print("⚠️ MODERATE SPEED PERFORMANCE - Conservative driving")
            else:
                print("❌ LOW SPEED PERFORMANCE - Very conservative or frequent stops")
            
            print("="*60)
            
            if self.pygame_enabled:
                pygame.quit()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def run(self):
        """Main entry point"""
        try:
            # Connect to CARLA
            if not self.connect_to_carla():
                return False
            
            # Find ego vehicle
            if not self.find_ego_vehicle():
                return False
            
            # Print initial help
            if self.pygame_enabled:
                self.print_help()
            
            # Start monitoring
            self.run_monitoring_loop()
            
            return True
            
        except Exception as e:
            print(f"Error in main run: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Real-time Vehicle Monitor for CARLA')
    parser.add_argument('--host', default='127.0.0.1', help='IP of CARLA server (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=2000, help='TCP port of CARLA server (default: 2000)')
    parser.add_argument('--no-pygame', action='store_true', help='Disable pygame interface')
    parser.add_argument('--save-data', action='store_true', help='Automatically save monitoring data')
    parser.add_argument('--camera-mode', choices=['behind', 'above', 'chase', 'side', 'bird'], 
                       default='behind', help='Initial camera mode')
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = VehicleMonitor(args)
    
    try:
        success = monitor.run()
        if success:
            print("✅ Monitoring completed successfully")
        else:
            print("❌ Monitoring failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
