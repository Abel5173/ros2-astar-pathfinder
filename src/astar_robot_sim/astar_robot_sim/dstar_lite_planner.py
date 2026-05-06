#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class DStarLitePlanner(Node):
    def __init__(self):
        super().__init__('dstar_lite_planner')
        self.get_logger().info('D* Lite Planner Node started!')

        self.grid_width  = 40
        self.grid_height = 40
        self.resolution  = 0.25

        self.obstacles = set()
        self.build_house_obstacles()

        self.start = (2, 2)
        self.goal  = (37, 37)

        self.g   = {}
        self.rhs = {}
        self.U   = {}
        self.km  = 0.0

        self.path_pub = self.create_publisher(Path, '/dstar_path', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/dstar_grid', 10)

        self.initialize()
        self.compute_shortest_path()
        self.path = self.extract_path()

        if self.path:
            self.get_logger().info(
                f'D* Lite initial path: {len(self.path)-1} steps.')
        else:
            self.get_logger().warn('D* Lite: No initial path found!')
            self.path = []

        self.obstacle_added = False  # flag — only add once

        self.create_timer(2.0, self.republish)
        self.create_timer(8.0, self.add_dynamic_obstacle)

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

    def h(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

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

    def add_dynamic_obstacle(self):
        if self.obstacle_added:
            return                          # only run once
        self.obstacle_added = True
        self.get_logger().info(
            'D* Lite: New obstacle detected! Replanning...')
        new_obs = [(10,10),(11,10),(12,10),(13,10),(14,10)]
        for obs in new_obs:
            self.obstacles.add(obs)
            self.g[obs]   = float('inf')
            self.rhs[obs] = float('inf')
            for nb, _ in self.neighbors(obs):
                self.update_vertex(nb)
        self.km += self.h(self.start, self.start)
        self.compute_shortest_path()
        new_path = self.extract_path()
        if new_path:
            self.path = new_path
            self.get_logger().info(
                f'D* Lite: Replanned! New path: {len(new_path)-1} steps.')
        else:
            self.get_logger().warn('D* Lite: Replan failed!')

    def republish(self):
        self.publish_grid()
        if self.path:
            self.publish_path(self.path)

    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        for (r, c) in path:
            ps = PoseStamped()
            ps.header.frame_id    = 'map'
            ps.pose.position.x    = float(c) * self.resolution - 5.0
            ps.pose.position.y    = float(r) * self.resolution - 5.0
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
        msg.info.origin.position.x = -5.0
        msg.info.origin.position.y = -5.0
        msg.data = [0] * (self.grid_width * self.grid_height)
        for (r, c) in self.obstacles:
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                msg.data[r * self.grid_width + c] = 100
        self.grid_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DStarLitePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
