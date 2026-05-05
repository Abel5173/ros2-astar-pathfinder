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

        self.grid_width  = 20
        self.grid_height = 20
        self.resolution  = 0.5  # meters per cell

        # Obstacle cells (row, col)
        self.obstacles = {(4,2),(4,3),(4,4),(8,5),(8,6),(10,10),(10,11)}
        self.start = (1, 1)
        self.goal  = (18, 18)

        self.path_pub = self.create_publisher(Path, '/astar_path', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        # Run once after 1 second
        self.timer = self.create_timer(1.0, self.run_once)
        self.done  = False

    def run_once(self):
        if self.done:
            return
        self.done = True
        self.publish_grid()
        path = self.astar(self.start, self.goal)
        if path:
            self.get_logger().info(f'Path found! {len(path)-1} steps.')
            self.publish_path(path)
        else:
            self.get_logger().warn('No path found! Check obstacles.')

    def heuristic(self, a, b):
        # Manhattan distance
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
                    f = tg + self.heuristic(nb, goal)
                    heapq.heappush(heap, (f, nb))
        return None

    def publish_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for (r, c) in path:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(c) * self.resolution
            ps.pose.position.y = float(r) * self.resolution
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.get_logger().info('Path published to /astar_path')

    def publish_grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width  = self.grid_width
        msg.info.height = self.grid_height
        msg.data = [0] * (self.grid_width * self.grid_height)
        for (r, c) in self.obstacles:
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                msg.data[r * self.grid_width + c] = 100
        self.grid_pub.publish(msg)
        self.get_logger().info('Occupancy grid published to /occupancy_grid')

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
