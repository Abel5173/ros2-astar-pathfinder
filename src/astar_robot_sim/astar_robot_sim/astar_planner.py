#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import heapq, math
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info('A* Planner Node started!')

        self.grid_width  = 40
        self.grid_height = 40
        self.resolution  = 0.25
        # World origin offset: grid cell (r,c) → world (x,y) = (c*res + origin, r*res + origin)
        self.origin      = -5.0

        self.obstacles = set()
        self.build_house_obstacles()

        # Real robot position in world coords (updated from /odom)
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.odom_received = False

        # Goal in grid coords (updated from /goal_pose, default = far corner)
        self.goal = (35, 35)

        # Current planned path — replanned whenever start or goal changes
        self.path = []
        self.last_start = None   # track to avoid replanning on every odom tick

        # Dynamic obstacles detected by LiDAR — separate from static map walls
        # so we can clear them between scans without erasing the house layout
        self.dynamic_obstacles = set()
        self.dynamic_inflation_cells = 1

        self.path_pub = self.create_publisher(Path,          '/astar_path',     10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid',  10)
        self.goal_pub = self.create_publisher(PoseStamped,   '/goal_pose',       10)

        # Subscribe to real odometry for live start position
        self.create_subscription(Odometry,    '/odom',      self.odom_callback,  10)
        # Accept external goal commands (e.g. from RViz "2D Nav Goal" or another node)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback,  10)
        # Convert LiDAR hits into dynamic obstacle cells
        self.create_subscription(LaserScan,   '/scan',      self.scan_callback,  10)

        # Republish grid + path at 2 Hz so RViz stays current
        self.create_timer(2.0, self.republish)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def world_to_grid(self, x, y):
        """Convert world (x, y) metres → grid (row, col), clamped to bounds."""
        col = int(round((x - self.origin) / self.resolution))
        row = int(round((y - self.origin) / self.resolution))
        col = max(0, min(self.grid_width  - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return (row, col)

    def grid_to_world(self, row, col):
        """Convert grid (row, col) → world (x, y) metres."""
        x = float(col) * self.resolution + self.origin
        y = float(row) * self.resolution + self.origin
        return x, y

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract Yaw from Quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f'First odom received: world ({self.robot_x:.2f}, {self.robot_y:.2f})')

        new_start = self.world_to_grid(self.robot_x, self.robot_y)

        # Only replan when the robot has moved to a different grid cell
        if new_start != self.last_start:
            self.last_start = new_start
            self._replan(new_start, self.goal)

    def goal_callback(self, msg):
        new_goal = self.world_to_grid(
            msg.pose.position.x, msg.pose.position.y)

        if new_goal == self.goal:
            return

        self.goal = new_goal
        self.get_logger().info(
            f'New goal received: world ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}) → grid {self.goal}')

        if self.last_start is not None:
            self._replan(self.last_start, self.goal)

    def scan_callback(self, msg):
        """
        Project each valid laser ray into a world hit point, convert to a
        grid cell, and add it to dynamic_obstacles.  Replans only when the
        set of dynamic obstacles actually changes so we don't thrash A*.
        """
        new_dynamic = set()

        angle = msg.angle_min
        for r in msg.ranges:
            # Skip invalid / out-of-range readings
            if math.isnan(r) or math.isinf(r):
                angle += msg.angle_increment
                continue
            if r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            # Hit point in world frame (laser frame ≈ robot frame for a
            # fixed lidar_joint, robot yaw comes from odom)
            hit_x = self.robot_x + r * math.cos(self.robot_yaw + angle)
            hit_y = self.robot_y + r * math.sin(self.robot_yaw + angle)

            cell_r, cell_c = self.world_to_grid(hit_x, hit_y)

            # Inflate each scan hit by one cell so the path keeps safer
            # clearance from walls and newly detected obstacles.
            for dr in range(-self.dynamic_inflation_cells, self.dynamic_inflation_cells + 1):
                for dc in range(-self.dynamic_inflation_cells, self.dynamic_inflation_cells + 1):
                    rr = cell_r + dr
                    cc = cell_c + dc
                    if not (0 <= rr < self.grid_height and 0 <= cc < self.grid_width):
                        continue
                    cell = (rr, cc)
                    # Never mark the robot's own cell or the goal as an obstacle
                    if self.last_start and cell == self.last_start:
                        continue
                    if cell == self.goal:
                        continue
                    new_dynamic.add(cell)

            angle += msg.angle_increment

        if new_dynamic == self.dynamic_obstacles:
            return  # nothing changed — skip replan

        # Merge: remove old dynamic cells from obstacles, add new ones
        self.obstacles -= self.dynamic_obstacles
        self.dynamic_obstacles = new_dynamic
        self.obstacles |= self.dynamic_obstacles

        # Replan from current position if we have one
        if self.last_start is not None:
            self._replan(self.last_start, self.goal)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _replan(self, start, goal):
        """Run A* from current grid start to goal and publish the result."""
        if start in self.obstacles:
            self.get_logger().warn(
                f'Start cell {start} is inside an obstacle — skipping replan.')
            return
        if goal in self.obstacles:
            self.get_logger().warn(
                f'Goal cell {goal} is inside an obstacle — skipping replan.')
            return

        path = self.astar(start, goal)
        if path:
            self.path = path
            self.get_logger().info(
                f'Replanned: {start} → {goal}  ({len(path)-1} steps)')
            self.publish_path(self.path)
        else:
            self.get_logger().warn(
                f'No path found from {start} to {goal}')

    def build_house_obstacles(self):
        # Outer walls
        for i in range(40):
            self.obstacles.add((0,  i))
            self.obstacles.add((39, i))
            self.obstacles.add((i,  0))
            self.obstacles.add((i, 39))
        # Interior walls
        for i in range(5,  25): self.obstacles.add((15, i))
        for i in range(20, 40): self.obstacles.add((20, i))
        for i in range(0,  15): self.obstacles.add((i,  20))
        for i in range(20, 35): self.obstacles.add((i,  25))
        # Doorways (5 cells wide)
        for i in [9, 10, 11, 12, 13]:  self.obstacles.discard((15, i))
        for i in [27, 28, 29, 30, 31]: self.obstacles.discard((20, i))
        for i in [6,  7,  8,  9, 10]:  self.obstacles.discard((i,  20))
        for i in [26, 27, 28, 29, 30]: self.obstacles.discard((i,  25))

    def republish(self):
        self.publish_grid()
        if self.path:
            self.publish_path(self.path)
        self.publish_goal()

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        heap = []
        heapq.heappush(heap, (0, start))
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
                cost = math.sqrt(2) if dr != 0 and dc != 0 else 1.0
                tg = g[cur] + cost
                if tg < g.get(nb, float('inf')):
                    came_from[nb] = cur
                    g[nb] = tg
                    heapq.heappush(heap,
                        (tg + self.heuristic(nb, goal), nb))
        return None

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
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
