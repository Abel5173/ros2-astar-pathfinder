#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot Controller started!')

        self.path          = []
        self.current_idx   = 0
        self.robot_x       = 0.0
        self.robot_y       = 0.0
        self.robot_yaw     = 0.0
        self.moving        = False
        self.path_received = False
        self.stuck_counter = 0
        self.last_x        = 0.0
        self.last_y        = 0.0

        self.create_subscription(
            Path, '/astar_path', self.path_callback, 10)
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control_loop)
        # Check if stuck every 3 seconds
        self.create_timer(3.0, self.stuck_check)

    def path_callback(self, msg):
        if self.path_received:
            return
        if len(msg.poses) > 0:
            # Take every 6th waypoint — less zigzagging, smoother path
            self.path = msg.poses[::6]
            # Always include the final goal
            if msg.poses[-1] not in self.path:
                self.path.append(msg.poses[-1])
            self.current_idx   = 0
            self.moving        = True
            self.path_received = True
            self.get_logger().info(
                f'Path received! Navigating {len(self.path)} waypoints...')

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def stuck_check(self):
        if not self.moving:
            return
        dx = self.robot_x - self.last_x
        dy = self.robot_y - self.last_y
        moved = math.sqrt(dx*dx + dy*dy)
        if moved < 0.03:
            self.stuck_counter += 1
            self.get_logger().warn(
                f'Robot may be stuck! (count={self.stuck_counter})')
            if self.stuck_counter >= 2:
                # Skip to next waypoint
                self.get_logger().warn('Skipping to next waypoint...')
                self.current_idx  += 1
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0
        self.last_x = self.robot_x
        self.last_y = self.robot_y

    def control_loop(self):
        if not self.moving or self.current_idx >= len(self.path):
            return

        target = self.path[self.current_idx].pose.position
        dx     = target.x - self.robot_x
        dy     = target.y - self.robot_y
        dist   = math.sqrt(dx*dx + dy*dy)
        angle  = math.atan2(dy, dx)

        angle_err = angle - self.robot_yaw
        while angle_err >  math.pi: angle_err -= 2*math.pi
        while angle_err < -math.pi: angle_err += 2*math.pi

        cmd = Twist()

        if dist < 0.35:
            # Waypoint reached — move to next
            self.current_idx += 1
            self.stuck_counter = 0
            if self.current_idx >= len(self.path):
                self.get_logger().info('Goal reached! Robot stopped.')
                self.moving       = False
                cmd.linear.x      = 0.0
                cmd.angular.z     = 0.0
        else:
            if abs(angle_err) > 0.5:
                # Rotate in place
                cmd.linear.x  = 0.0
                cmd.angular.z = max(-1.2, min(1.2, 1.2 * angle_err))
            else:
                # Drive forward with gentle correction
                cmd.linear.x  = min(0.12, dist * 0.4)
                cmd.angular.z = max(-0.8, min(0.8, 0.8 * angle_err))

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
