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

        self.declare_parameter('path_topic', '/astar_path')
        path_topic = self.get_parameter('path_topic').get_parameter_value().string_value

        self.path          = []
        self.current_idx   = 0
        self.robot_x       = 0.0
        self.robot_y       = 0.0
        self.robot_yaw     = 0.0
        self.moving        = False
        self.stuck_counter = 0
        self.last_x        = 0.0
        self.last_y        = 0.0
        self.reached_goal  = False

        # Track how many consecutive times we've tried to skip a waypoint
        self.skip_attempts = 0

        self.create_subscription(Path,     path_topic, self.path_callback, 10)
        self.create_subscription(Odometry, '/odom',    self.odom_callback,  10)

        self.get_logger().info(f'Listening for paths on: {path_topic}')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control_loop)
        # Check if stuck every 2 seconds (faster recovery than before)
        self.create_timer(2.0, self.stuck_check)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def path_callback(self, msg):
        if len(msg.poses) == 0:
            return
        if self.reached_goal:
            return  # already done, ignore further replans

        new_path = list(msg.poses)

        # Find the closest waypoint on the new path to the robot's position.
        # Skip any waypoints already behind us (within 0.15 m).
        best_idx  = 0
        best_dist = float('inf')
        for i, pose in enumerate(new_path):
            dx = pose.pose.position.x - self.robot_x
            dy = pose.pose.position.y - self.robot_y
            d  = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        # Only resume ahead of current position if close enough to a waypoint.
        start_idx = best_idx if (best_dist < 0.30 and best_idx > 0) else 0

        self.path          = new_path
        self.current_idx   = start_idx
        self.moving        = True
        self.reached_goal  = False
        self.skip_attempts = 0
        self.get_logger().info(
            f'Updated path received: {len(self.path)} waypoints, '
            f'resuming from waypoint {self.current_idx}.')

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    def stuck_check(self):
        if not self.moving:
            self.last_x = self.robot_x
            self.last_y = self.robot_y
            return

        dx    = self.robot_x - self.last_x
        dy    = self.robot_y - self.last_y
        moved = math.sqrt(dx * dx + dy * dy)

        if moved < 0.03:
            self.stuck_counter += 1
            self.get_logger().warn(
                f'Robot may be stuck! (count={self.stuck_counter})')

            # Skip waypoints progressively to escape a stuck situation.
            # After 2 consecutive stuck checks (4 s), skip one waypoint.
            # After 4 more (8 s), skip two at once. Cap at 5 skips total.
            if self.stuck_counter >= 2:
                skip = min(1 + (self.skip_attempts // 2), 3)
                new_idx = self.current_idx + skip
                if new_idx < len(self.path):
                    self.get_logger().warn(
                        f'Skipping {skip} waypoint(s) '
                        f'({self.current_idx} -> {new_idx}) to get unstuck.')
                    self.current_idx   = new_idx
                    self.skip_attempts += 1
                    self.stuck_counter  = 0
                else:
                    # No more waypoints to skip — stop and wait for replan
                    self.get_logger().warn('No waypoints left to skip.')
                    self.stuck_counter = 0
        else:
            self.stuck_counter  = 0
            self.skip_attempts  = 0

        self.last_x = self.robot_x
        self.last_y = self.robot_y

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def control_loop(self):
        if not self.moving or self.current_idx >= len(self.path):
            return

        target = self.path[self.current_idx].pose.position
        dx     = target.x - self.robot_x
        dy     = target.y - self.robot_y
        dist   = math.sqrt(dx * dx + dy * dy)
        angle  = math.atan2(dy, dx)

        angle_err = angle - self.robot_yaw
        while angle_err >  math.pi: angle_err -= 2 * math.pi
        while angle_err < -math.pi: angle_err += 2 * math.pi

        cmd = Twist()

        # Slightly looser tolerance for the final goal waypoint
        is_last      = (self.current_idx == len(self.path) - 1)
        waypoint_tol = 0.30 if is_last else 0.20

        if dist < waypoint_tol:
            # Waypoint reached — advance
            self.current_idx  += 1
            self.stuck_counter  = 0
            self.skip_attempts  = 0
            if self.current_idx >= len(self.path):
                self.get_logger().info('Goal reached! Robot stopped.')
                self.moving      = True   # keep accepting new paths
                self.reached_goal = True
                cmd.linear.x     = 0.0
                cmd.angular.z    = 0.0
        else:
            if abs(angle_err) > 0.4:
                # Rotate in place first
                cmd.linear.x  = 0.0
                cmd.angular.z = max(-1.2, min(1.2, 1.5 * angle_err))
            else:
                # Drive forward with gentle angular correction
                cmd.linear.x  = min(0.15, dist * 0.5)
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