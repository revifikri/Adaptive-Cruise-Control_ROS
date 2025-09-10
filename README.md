# Adaptive-Cruise-Control_ROS




### Command to run the program
ros2 launch global_plan ros.launch.py use_ros_communication:=true spawn_traffic:=true

### parameter_optimizer.py
This file use genetic algorithm method to get speed parameter

### spectator.py
This file used to monitor ego vehicle during simulation

### visualize.py
This file use to visualize every event during during simulation (global planning, actor vehicle, pedestrian, traffic light, etc)
