#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import heapq, math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info('A* Planner Node started!')

        self.grid_width  = 40
        self.grid_height = 40
        self.resolution  = 0.25

        self.obstacles = set()
        self.build_house_obstacles()

        # Start and goal well away from walls
        self.start = (4, 4)
        self.goal  = (35, 35)

        self.path_pub = self.create_publisher(Path, '/astar_path', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.path = self.astar(self.start, self.goal)
        if self.path:
            self.get_logger().info(f'A* Path found! {len(self.path)-1} steps.')
        else:
            self.get_logger().warn('No path found!')
            self.path = []

        self.create_timer(2.0, self.republish)

    def build_house_obstacles(self):
        # Outer walls
        for i in range(40):
            self.obstacles.add((0, i))
            self.obstacles.add((39, i))
            self.obstacles.add((i, 0))
            self.obstacles.add((i, 39))
        # Add 1-cell padding around all obstacles so robot stays clear of walls
        for i in range(5, 25): self.obstacles.add((15, i))
        for i in range(20, 40): self.obstacles.add((20, i))
        for i in range(0, 15): self.obstacles.add((i, 20))
        for i in range(20, 35): self.obstacles.add((i, 25))
        # Wider doorways (5 cells wide) so robot can fit through
        for i in [9, 10, 11, 12, 13]:  self.obstacles.discard((15, i))
        for i in [27, 28, 29, 30, 31]: self.obstacles.discard((20, i))
        for i in [6, 7, 8, 9, 10]:     self.obstacles.discard((i, 20))
        for i in [26, 27, 28, 29, 30]: self.obstacles.discard((i, 25))

    def republish(self):
        self.publish_grid()
        if self.path:
            self.publish_path(self.path)
            self.publish_goal()

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

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

    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
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

    def publish_goal(self):
        msg = PoseStamped()
        msg.header.frame_id    = 'map'
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.position.x    = float(self.goal[1]) * self.resolution - 5.0
        msg.pose.position.y    = float(self.goal[0]) * self.resolution - 5.0
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
