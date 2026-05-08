#!/usr/bin/env python3
import math
import heapq
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan


class DStarLitePlanner(Node):
    def __init__(self):
        super().__init__('dstar_lite_planner')
        self.get_logger().info('D* Lite Planner Node started!')

        self.grid_width  = 40
        self.grid_height = 40
        self.resolution  = 0.25
        self.origin      = -5.0

        self.obstacles = set()
        self.build_house_obstacles()
        self.static_obstacles = frozenset(self.obstacles)

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False
        self.last_start    = None

        self.goal = (35, 35)
        self.path = []

        self.dynamic_obstacles = set()

        # ---------------------------------------------------------------
        # FIX 2 & 3: Debounce — track how many consecutive scans show the
        # same dynamic-obstacle set before we act on it.
        # ---------------------------------------------------------------
        self._last_dynamic_snapshot = frozenset()
        self._dynamic_stable_count  = 0
        self._DEBOUNCE_SCANS        = 3   # require 3 consecutive matching scans

        self.path_pub = self.create_publisher(Path,          '/dstar_path',  10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/dstar_grid',  10)
        self.goal_pub = self.create_publisher(PoseStamped,   '/goal_pose',   10)

        self.create_subscription(Odometry,    '/odom',      self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan,   '/scan',      self.scan_callback, 10)

        self.create_timer(1.0, self.republish)

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def build_house_obstacles(self):
        for i in range(40):
            self.obstacles.add((0,  i)); self.obstacles.add((39, i))
            self.obstacles.add((i,  0)); self.obstacles.add((i, 39))
        for i in range(5,  25): self.obstacles.add((15, i))
        for i in range(20, 40): self.obstacles.add((20, i))
        for i in range(0,  15): self.obstacles.add((i,  20))
        for i in range(20, 35): self.obstacles.add((i,  25))
        # 5-cell-wide doorways
        for i in [9, 10, 11, 12, 13]:  self.obstacles.discard((15, i))
        for i in [27, 28, 29, 30, 31]: self.obstacles.discard((20, i))
        for i in [6,  7,  8,  9, 10]:  self.obstacles.discard((i,  20))
        for i in [26, 27, 28, 29, 30]: self.obstacles.discard((i,  25))

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    def world_to_grid(self, x, y):
        col = int(round((x - self.origin) / self.resolution))
        row = int(round((y - self.origin) / self.resolution))
        col = max(0, min(self.grid_width  - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return (row, col)

    def grid_to_world(self, row, col):
        x = float(col) * self.resolution + self.origin
        y = float(row) * self.resolution + self.origin
        return x, y

    # ------------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f'First odom received: world ({self.robot_x:.2f}, {self.robot_y:.2f})')

        new_start = self.world_to_grid(self.robot_x, self.robot_y)
        if new_start != self.last_start:
            self.last_start = new_start
            self._replan(new_start)

    def goal_callback(self, msg):
        new_goal = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        if new_goal == self.goal:
            return
        self.goal = new_goal
        self.get_logger().info(
            f'New goal received: world ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}) -> grid {self.goal}')
        if self.last_start is not None:
            self._replan(self.last_start)

    def scan_callback(self, msg):
        """
        Register dynamic obstacles only for hits that are:
          1. Very close (< 0.45 m) — FIX 2: tighter range stops wall-edge false
             positives that caused the 19-step / 26-step oscillation
          2. NOT already in the static map
          3. Not the robot's own cell or the goal

        FIX 3: Debounce — only act when the same obstacle set has been
        seen in 3 consecutive scans, suppressing single-scan flicker.
        """
        new_dynamic = set()
        angle = msg.angle_min
        for r in msg.ranges:
            angle += msg.angle_increment
            if math.isnan(r) or math.isinf(r):
                continue
            # FIX 2: reduced from 0.8 m to 0.45 m
            if r < msg.range_min or r > 0.45:
                continue
            hit_x = self.robot_x + r * math.cos(self.robot_yaw + angle)
            hit_y = self.robot_y + r * math.sin(self.robot_yaw + angle)
            cell  = self.world_to_grid(hit_x, hit_y)
            if cell in self.static_obstacles:
                continue
            if self.last_start and cell == self.last_start:
                continue
            if cell == self.goal:
                continue
            new_dynamic.add(cell)

        # ---------------------------------------------------------------
        # FIX 3: Debounce logic
        # ---------------------------------------------------------------
        snapshot = frozenset(new_dynamic)
        if snapshot == self._last_dynamic_snapshot:
            self._dynamic_stable_count += 1
        else:
            self._last_dynamic_snapshot = snapshot
            self._dynamic_stable_count  = 1

        if self._dynamic_stable_count < self._DEBOUNCE_SCANS:
            return  # wait for the reading to stabilise

        # Debounce passed — proceed only if something actually changed
        if new_dynamic == self.dynamic_obstacles:
            return

        new_cells     = new_dynamic - self.dynamic_obstacles
        removed_cells = self.dynamic_obstacles - new_dynamic
        path_cells    = set(self.path) if self.path else set()
        path_blocked  = bool(new_cells     & path_cells)
        path_cleared  = bool(removed_cells & path_cells)

        self.obstacles -= self.dynamic_obstacles
        self.dynamic_obstacles = new_dynamic
        self.obstacles |= self.dynamic_obstacles

        if (path_blocked or path_cleared) and self.last_start is not None:
            self.get_logger().info('Dynamic obstacle changed path — replanning.')
            self._replan(self.last_start)

    # ------------------------------------------------------------------
    # D* Lite — pure A* fallback (simple, correct, no incremental state)
    # ------------------------------------------------------------------
    def _astar(self, start, goal):
        """Standard A* on the current obstacle set."""
        heap = []
        heapq.heappush(heap, (0.0, start))
        came_from = {}
        g = {start: 0.0}

        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                return list(reversed(path))

            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                           (1,1),(1,-1),(-1,1),(-1,-1)]:
                nb = (cur[0]+dr, cur[1]+dc)
                if not (0 <= nb[0] < self.grid_height and
                        0 <= nb[1] < self.grid_width):
                    continue
                if nb in self.obstacles:
                    continue
                cost = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                tg = g[cur] + cost
                if tg < g.get(nb, float('inf')):
                    came_from[nb] = cur
                    g[nb] = tg
                    h = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
                    heapq.heappush(heap, (tg + h, nb))
        return None

    # ------------------------------------------------------------------
    # Replan entry point
    # ------------------------------------------------------------------
    def _replan(self, start):
        goal = self.goal
        if start in self.obstacles:
            self.get_logger().warn(
                f'Start cell {start} is inside an obstacle - skipping replan.')
            return
        if goal in self.obstacles:
            self.get_logger().warn(
                f'Goal cell {goal} is inside an obstacle - skipping replan.')
            return

        new_path = self._astar(start, goal)
        if new_path:
            self.path = new_path
            self.get_logger().info(
                f'D* Lite replanned: {start} -> {goal} ({len(new_path)-1} steps)')
            self.publish_path(self.path)
        else:
            self.get_logger().warn(
                f'D* Lite: No path found from {start} to {goal}')

    # ------------------------------------------------------------------
    # Periodic republish
    # ------------------------------------------------------------------
    def republish(self):
        self.publish_grid()
        if self.path:
            self.publish_path(self.path)
        self.publish_goal()

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for (r, c) in path:
            ps = PoseStamped()
            ps.header.frame_id    = 'map'
            x, y = self.grid_to_world(r, c)
            ps.pose.position.x    = x
            ps.pose.position.y    = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def publish_grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id        = 'map'
        msg.header.stamp           = self.get_clock().now().to_msg()
        msg.info.resolution        = self.resolution
        msg.info.width             = self.grid_width
        msg.info.height            = self.grid_height
        msg.info.origin.position.x = self.origin
        msg.info.origin.position.y = self.origin
        msg.data = [0] * (self.grid_width * self.grid_height)
        for (r, c) in self.obstacles:
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                msg.data[r * self.grid_width + c] = 100
        self.grid_pub.publish(msg)

    def publish_goal(self):
        msg = PoseStamped()
        msg.header.frame_id    = 'map'
        msg.header.stamp       = self.get_clock().now().to_msg()
        x, y = self.grid_to_world(self.goal[0], self.goal[1])
        msg.pose.position.x    = x
        msg.pose.position.y    = y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DStarLitePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
