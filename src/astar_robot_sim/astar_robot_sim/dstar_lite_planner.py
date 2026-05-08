#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan


class DStarLitePlanner(Node):
    def __init__(self):
        super().__init__('dstar_lite_planner')
        self.get_logger().info('D* Lite Planner Node started!')

        self.grid_width = 40
        self.grid_height = 40
        self.resolution = 0.25
        self.origin = -5.0

        self.obstacles = set()
        self.build_house_obstacles()
        self.dynamic_obstacles = set()
        self.dynamic_inflation_cells = 1

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False
        self.last_start = None

        self.start = (2, 2)
        self.goal = (35, 35)
        self.path = []

        self.g = {}
        self.rhs = {}
        self.U = {}
        self.km = 0.0

        self.path_pub = self.create_publisher(Path, '/dstar_path', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/dstar_grid', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.create_timer(1.0, self.republish)

    def build_house_obstacles(self):
        for i in range(40):
            self.obstacles.add((0, i))
            self.obstacles.add((39, i))
            self.obstacles.add((i, 0))
            self.obstacles.add((i, 39))
        for i in range(5, 25): self.obstacles.add((15, i))
        for i in range(20, 40): self.obstacles.add((20, i))
        for i in range(0, 15): self.obstacles.add((i, 20))
        for i in range(20, 35): self.obstacles.add((i, 25))
        for i in [10, 11, 12]: self.obstacles.discard((15, i))
        for i in [28, 29, 30]: self.obstacles.discard((20, i))
        for i in [7,  8,  9 ]: self.obstacles.discard((i, 20))
        for i in [27, 28, 29]: self.obstacles.discard((i, 25))

    def world_to_grid(self, x, y):
        col = int(round((x - self.origin) / self.resolution))
        row = int(round((y - self.origin) / self.resolution))
        col = max(0, min(self.grid_width - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return (row, col)

    def grid_to_world(self, row, col):
        x = float(col) * self.resolution + self.origin
        y = float(row) * self.resolution + self.origin
        return x, y

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f'First odom received: world ({self.robot_x:.2f}, {self.robot_y:.2f})')

        new_start = self.world_to_grid(self.robot_x, self.robot_y)
        if new_start != self.last_start:
            self.last_start = new_start
            self._replan(new_start, self.goal)

    def goal_callback(self, msg):
        new_goal = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        if new_goal == self.goal:
            return
        self.goal = new_goal
        self.get_logger().info(
            f'New goal received: world ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f}) -> grid {self.goal}')
        if self.last_start is not None:
            self._replan(self.last_start, self.goal)

    def scan_callback(self, msg):
        new_dynamic = set()
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r):
                angle += msg.angle_increment
                continue
            if r < msg.range_min or r > msg.range_max:
                angle += msg.angle_increment
                continue

            hit_x = self.robot_x + r * math.cos(self.robot_yaw + angle)
            hit_y = self.robot_y + r * math.sin(self.robot_yaw + angle)
            cell_r, cell_c = self.world_to_grid(hit_x, hit_y)

            for dr in range(-self.dynamic_inflation_cells, self.dynamic_inflation_cells + 1):
                for dc in range(-self.dynamic_inflation_cells, self.dynamic_inflation_cells + 1):
                    rr = cell_r + dr
                    cc = cell_c + dc
                    if not (0 <= rr < self.grid_height and 0 <= cc < self.grid_width):
                        continue
                    cell = (rr, cc)
                    if self.last_start and cell == self.last_start:
                        continue
                    if cell == self.goal:
                        continue
                    new_dynamic.add(cell)

            angle += msg.angle_increment

        if new_dynamic == self.dynamic_obstacles:
            return

        self.obstacles -= self.dynamic_obstacles
        self.dynamic_obstacles = new_dynamic
        self.obstacles |= self.dynamic_obstacles
        if self.last_start is not None:
            self._replan(self.last_start, self.goal)

    def h(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def g_val(self, s):
        return self.g.get(s, float('inf'))

    def rhs_val(self, s):
        return self.rhs.get(s, float('inf'))

    def calc_key(self, s):
        g_rhs = min(self.g_val(s), self.rhs_val(s))
        return (g_rhs + self.h(self.start, s) + self.km, g_rhs)

    def neighbors(self, s):
        result = []
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                       (1,1),(1,-1),(-1,1),(-1,-1)]:
            nb = (s[0]+dr, s[1]+dc)
            if 0 <= nb[0] < self.grid_height and \
               0 <= nb[1] < self.grid_width and \
               nb not in self.obstacles:
                cost = math.sqrt(2) if dr!=0 and dc!=0 else 1.0
                result.append((nb, cost))
        return result

    def update_vertex(self, u):
        if u != self.goal:
            best = float('inf')
            for nb, cost in self.neighbors(u):
                val = cost + self.g_val(nb)
                if val < best:
                    best = val
            self.rhs[u] = best
        if u in self.U:
            del self.U[u]
        if self.g_val(u) != self.rhs_val(u):
            self.U[u] = self.calc_key(u)

    def initialize(self):
        self.g   = {}
        self.rhs = {}
        self.U   = {}
        self.km  = 0.0
        self.rhs[self.goal] = 0.0
        self.U[self.goal]   = self.calc_key(self.goal)

    def compute_shortest_path(self):
        iterations = 0
        max_iter = self.grid_width * self.grid_height * 2
        while self.U and iterations < max_iter:
            iterations += 1
            u = min(self.U, key=lambda x: self.U[x])
            k_old = self.U[u]
            k_new = self.calc_key(u)
            if k_old < k_new:
                self.U[u] = k_new
            elif self.g_val(u) > self.rhs_val(u):
                self.g[u] = self.rhs_val(u)
                del self.U[u]
                for nb, _ in self.neighbors(u):
                    self.update_vertex(nb)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for nb, _ in self.neighbors(u):
                    self.update_vertex(nb)
            start_key = self.calc_key(self.start)
            if self.U.get(self.start) == start_key and \
               self.g_val(self.start) == self.rhs_val(self.start):
                break

    def extract_path(self):
        path = [self.start]
        cur  = self.start
        visited = set()
        for _ in range(self.grid_width * self.grid_height):
            if cur == self.goal:
                break
            if cur in visited:
                return None
            visited.add(cur)
            best_nb, best_cost = None, float('inf')
            for nb, cost in self.neighbors(cur):
                val = cost + self.g_val(nb)
                if val < best_cost:
                    best_cost = val
                    best_nb   = nb
            if best_nb is None:
                return None
            path.append(best_nb)
            cur = best_nb
        return path if cur == self.goal else None

    def _replan(self, start, goal):
        if start in self.obstacles:
            self.get_logger().warn(
                f'Start cell {start} is inside an obstacle - skipping replan.')
            return
        if goal in self.obstacles:
            self.get_logger().warn(
                f'Goal cell {goal} is inside an obstacle - skipping replan.')
            return

        self.start = start
        self.goal = goal
        self.initialize()
        self.compute_shortest_path()
        new_path = self.extract_path()
        if new_path:
            self.path = new_path
            self.get_logger().info(
                f'D* Lite replanned: {start} -> {goal} ({len(new_path)-1} steps)')
            self.publish_path(self.path)
        else:
            self.get_logger().warn(
                f'D* Lite: No path found from {start} to {goal}')

    def republish(self):
        self.publish_grid()
        if self.path:
            self.publish_path(self.path)
        self.publish_goal()

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
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        x, y = self.grid_to_world(self.goal[0], self.goal[1])
        msg.pose.position.x = x
        msg.pose.position.y = y
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
