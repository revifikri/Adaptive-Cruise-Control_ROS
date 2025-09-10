import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'global_plan'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'nav2-common',
        'nav2-core',
        'nav2-controller',
        'nav2-planner', 
        'nav2-behaviors',
        'nav2-lifecycle-manager',
        'dwb-core',
        'tf2-ros',
        'tf2-geometry-msgs',
    ],
    zip_safe=True,
    maintainer='av',
    maintainer_email='av@todo.todo',
    description='CARLA Path Planner with ROS Communication and Balanced Collision Avoidance',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # === ORIGINAL SYSTEM (Legacy) ===
            # Global planner (unchanged - your original)
            'global_plan_node = global_plan.global_plan:main',
            
            # Local planner (unchanged - your original) 
            'nav_dwb = global_plan.nav_dwb:main',
            
            # Nav2 DWB local planner (original)
            'dwa_local_planner = global_plan.nav2_dwb_local_planner:main',
            
            # Custom local planner (fallback - original)
            'custom_local_planner = global_plan.local_plan:main',

            # Debug tools (original)
            'debug_topics = global_plan.debug_topics:main',
            
            # Visualization (original)
            'visualize = global_plan.visualization:main',
            
            # === NEW ROS COMMUNICATION SYSTEM ===
            # Enhanced global planner with ROS communication
            'global_plan_node_ros = global_plan.global_plan_ros:main',
            
            # Enhanced local planner with ROS communication
            'nav_dwb_ros = global_plan.nav_dwb_ros:main',
            
            # ROS Communication monitoring
            'ros_topic_monitor = global_plan.ros_topic_monitor:main',
            
            # Communication health monitoring
            'communication_health = global_plan.communication_health:main',
            
            # Enhanced visualization with ROS data
            'visualize_ros = global_plan.visualize_ros:main',
            
            # Enhanced debug with ROS communication
            'debug_topics_ros = global_plan.debug_topics_ros:main',
        ],
    },
)
