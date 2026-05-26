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
        self.skip_attempts = 0

        # FIX B1/B2: replace the confusing reached_goal flag with a clean
        # at_goal flag; path_callback always accepts new paths regardless.
        self.at_goal = False

        # FIX B3: velocity ramping — track previous commands for smooth
        # acceleration / deceleration
        self._prev_linear  = 0.0
        self._prev_angular = 0.0
        self._MAX_LIN_ACCEL  = 0.08   # m/s per control tick (0.1 s) — faster ramp
        self._MAX_ANG_ACCEL  = 0.20   # rad/s per control tick

        self.create_subscription(Path,     path_topic, self.path_callback, 10)
        self.create_subscription(Odometry, '/odom',    self.odom_callback,  10)

        self.get_logger().info(f'Listening for paths on: {path_topic}')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control_loop)
        self.create_timer(1.5, self.stuck_check)   # check every 1.5s instead of 2s

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def path_callback(self, msg):
        if len(msg.poses) == 0:
            return

        # FIX B1: always accept new paths — no early return on at_goal.
        # When a new goal is published the planner will send a new path
        # and the controller must start following it.
        self.at_goal = False

        new_path = list(msg.poses)
        # RESUME_LOOKAHEAD must be > grid resolution (0.25m) so we skip
        # waypoints the robot has already passed or is sitting on top of.
        RESUME_LOOKAHEAD = 0.40

        best_idx  = 0
        best_dist = float('inf')
        for i, pose in enumerate(new_path):
            dx = pose.pose.position.x - self.robot_x
            dy = pose.pose.position.y - self.robot_y
            d  = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        # Walk forward from the closest waypoint until we find one that
        # is at least RESUME_LOOKAHEAD away — that is our next target.
        start_idx = len(new_path) - 1  # default: last waypoint
        for i in range(best_idx, len(new_path)):
            dx = new_path[i].pose.position.x - self.robot_x
            dy = new_path[i].pose.position.y - self.robot_y
            if math.sqrt(dx * dx + dy * dy) >= RESUME_LOOKAHEAD:
                start_idx = i
                break

        # If we already had a valid index into a previous path, try to
        # preserve progress: only reset if the new start_idx is further
        # ahead than where we were (prevents replans from rolling us back).
        if self.moving and self.path and self.current_idx < len(self.path):
            prev_target = self.path[self.current_idx].pose.position
            # Find where that old target sits in the new path
            best_carry = start_idx
            best_carry_dist = float('inf')
            for i in range(len(new_path)):
                dx = new_path[i].pose.position.x - prev_target.x
                dy = new_path[i].pose.position.y - prev_target.y
                d  = math.sqrt(dx * dx + dy * dy)
                if d < best_carry_dist:
                    best_carry_dist = d
                    best_carry = i
            # Use the carried index only if it's ahead of start_idx
            if best_carry > start_idx and best_carry_dist < 0.30:
                start_idx = best_carry

        self.path          = new_path
        self.current_idx   = start_idx
        self.moving        = True
        self.skip_attempts = 0
        self.get_logger().info(
            f'Updated path: {len(self.path)} waypoints, '
            f'resuming from {self.current_idx} (closest={best_idx}, d={best_dist:.2f}m).')

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
        if not self.moving or self.at_goal:
            self.last_x = self.robot_x
            self.last_y = self.robot_y
            return

        dx    = self.robot_x - self.last_x
        dy    = self.robot_y - self.last_y
        moved = math.sqrt(dx * dx + dy * dy)

        if moved < 0.03:
            self.stuck_counter += 1
            self.get_logger().warn(f'Robot may be stuck (count={self.stuck_counter})')

            if self.stuck_counter >= 2:
                skip = min(2 + (self.skip_attempts // 2), 5)  # start at 2, escalate faster
                new_idx = self.current_idx + skip
                if new_idx < len(self.path):
                    self.get_logger().warn(
                        f'Skipping {skip} waypoint(s) ({self.current_idx}→{new_idx})')
                    self.current_idx   = new_idx
                    self.skip_attempts += 1
                    # FIX B5: reset stuck_counter to 0 after skip, but
                    # keep skip_attempts so escalation still works.
                    self.stuck_counter = 0
                    # Also reset velocity ramp so we don't lurch
                    self._prev_linear  = 0.0
                    self._prev_angular = 0.0
                else:
                    self.get_logger().warn('No waypoints left to skip.')
                    self.stuck_counter = 0
        else:
            self.stuck_counter  = 0
            self.skip_attempts  = 0

        self.last_x = self.robot_x
        self.last_y = self.robot_y

    # ------------------------------------------------------------------
    # FIX B3: ramp a velocity command toward the target without
    # exceeding the per-tick acceleration limit.
    # ------------------------------------------------------------------
    def _ramp(self, current, target, max_delta):
        delta = target - current
        delta = max(-max_delta, min(max_delta, delta))
        return current + delta

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def control_loop(self):
        cmd = Twist()

        if not self.moving or self.current_idx >= len(self.path) or self.at_goal:
            # FIX B3: ramp velocity to zero smoothly on stop
            cmd.linear.x  = self._ramp(self._prev_linear,  0.0, self._MAX_LIN_ACCEL)
            cmd.angular.z = self._ramp(self._prev_angular, 0.0, self._MAX_ANG_ACCEL)
            self._prev_linear  = cmd.linear.x
            self._prev_angular = cmd.angular.z
            self.cmd_pub.publish(cmd)
            return

        target = self.path[self.current_idx].pose.position
        dx     = target.x - self.robot_x
        dy     = target.y - self.robot_y
        dist   = math.sqrt(dx * dx + dy * dy)
        angle  = math.atan2(dy, dx)

        angle_err = angle - self.robot_yaw
        while angle_err >  math.pi: angle_err -= 2 * math.pi
        while angle_err < -math.pi: angle_err += 2 * math.pi

        is_last      = (self.current_idx == len(self.path) - 1)
        waypoint_tol = 0.30 if is_last else 0.28

        if dist < waypoint_tol:
            self.current_idx  += 1
            self.stuck_counter  = 0
            self.skip_attempts  = 0
            if self.current_idx >= len(self.path):
                self.get_logger().info('Goal reached! Stopping.')
                # FIX B2: set at_goal cleanly; moving stays True so the
                # robot will respond to the next path immediately.
                self.at_goal = True
            # FIX B3: ramp to zero at waypoint arrival
            target_lin  = 0.0
            target_ang  = 0.0
        else:
            if abs(angle_err) > 0.4:
                target_lin  = 0.0
                target_ang  = max(-1.0, min(1.0, 1.5 * angle_err))
            else:
                target_lin  = min(0.20, dist * 0.6)
                target_ang  = max(-0.7, min(0.7, 0.8 * angle_err))

        # FIX B3: apply acceleration ramp
        cmd.linear.x  = self._ramp(self._prev_linear,  target_lin,  self._MAX_LIN_ACCEL)
        cmd.angular.z = self._ramp(self._prev_angular, target_ang,  self._MAX_ANG_ACCEL)
        self._prev_linear  = cmd.linear.x
        self._prev_angular = cmd.angular.z

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()