import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import cv2

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
	# TODO
        # Load and process the generated map
        self.previous_map = cv2.imread('path_to_generated_map.ppm', cv2.IMREAD_GRAYSCALE)
        self.previous_map = cv2.threshold(self.previous_map, 128, 255, cv2.THRESH_BINARY)[1]
        
        self.map_resolution = 0.05  # meters per pixel, not sure what to set here
        self.map_origin = (-10, -10)  # same here needs to be changed
        
        self.robot_pose = None

    def scan_callback(self, msg):
        if self.robot_pose is None:
            return

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        for range, angle in zip(ranges, angles):
            if np.isinf(range) or np.isnan(range):
                continue

            # Convert laser scan to world coordinates
            x = self.robot_pose.x + range * np.cos(angle)
            y = self.robot_pose.y + range * np.sin(angle)

            # Convert world coordinates to map coordinates
            map_x, map_y = self.world_to_map(x, y)

            # Check if the point is within map bounds
            if 0 <= map_x < self.previous_map.shape[1] and 0 <= map_y < self.previous_map.shape[0]:
                previous_map_value = self.previous_map[map_y, map_x]
                
                # TODO I'm not sure about this condition as a check for changes, it might be too sensitive to minor changes, requires testing
                if previous_map_value == 255 and range < msg.range_max:
                    self.get_logger().info(f"New obstacle detected at ({x:.2f}, {y:.2f})")

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position

    def world_to_map(self, x, y):
        map_x = int((x - self.map_origin[0]) / self.map_resolution)
        map_y = int((y - self.map_origin[1]) / self.map_resolution)
        return map_x, map_y

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = ObstacleDetector()
    rclpy.spin(obstacle_detector)
    obstacle_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
