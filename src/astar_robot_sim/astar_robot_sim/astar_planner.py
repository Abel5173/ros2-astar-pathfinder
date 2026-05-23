import rclpy
from rclpy.node import Node
import heapq, math, time
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SpawnEntity, DeleteEntity


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info('A* Planner Node started!')

        self.grid_width  = 60
        self.grid_height = 48
        self.resolution  = 0.25
        self.origin_x    = -7.75   # world X of grid column 0
        self.origin_y    = -6.25   # world Y of grid row 0
        # Keep a single 'origin' alias for backward-compat with coord helpers
        self.origin      = self.origin_x

        self.obstacles = set()
        self.build_house_obstacles()
        # No inflation needed — the rasterizer already adds res*0.6 margin
        # around each wall. Inflating further causes the spawn cell to be
        # marked as obstacle when the robot starts near an interior wall.
        self.static_obstacles = frozenset(self.obstacles)

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False

        self.goal       = (31, 7)    # world (-6.0, 1.5) — left room, opposite side of house
        self.path       = []
        self.last_start = None

        # ------------------------------------------------------------------
        # Patrol task — robot visits each waypoint in sequence, then loops.
        # Waypoints chosen as clear open-floor positions in each room,
        # verified free of walls AND furniture.
        #   (31,  7) = world (-6.00,  1.50) — left room
        #   (15, 21) = world (-2.50, -2.50) — centre lower corridor
        #   (41, 43) = world ( 3.00,  4.00) — right upper room
        #   ( 9, 35) = world ( 1.00, -4.00) — bottom corridor
        # ------------------------------------------------------------------
        self._patrol_waypoints = [
            (31,  7),   # left room
            (15, 21),   # centre lower corridor
            (41, 43),   # right upper room
            ( 9, 35),   # bottom corridor
        ]
        self._patrol_idx      = 0
        self._patrol_arrived  = False   # True once robot is within arrival radius
        self._ARRIVAL_RADIUS  = 0.50    # metres — how close counts as "arrived"
        self.goal = self._patrol_waypoints[self._patrol_idx]
        self.get_logger().info(
            f'Patrol task started. {len(self._patrol_waypoints)} waypoints. '
            f'First goal: {self.goal}'
        )

        self.dynamic_obstacles = set()

        self._last_dynamic_snapshot = frozenset()
        self._dynamic_stable_count  = 0
        self._DEBOUNCE_SCANS        = 6   # more debounce — prevents replan storm

        # Metrics
        self.replan_count   = 0
        self.last_path_cost = 0.0
        self.last_replan_ms = 0.0

        # FIX 6: track the last published path content so the 1 Hz
        # republish does NOT send the same path again and reset the
        # controller's waypoint index.
        self._last_published_path = []

        self.path_pub    = self.create_publisher(Path,          '/astar_path',    10)
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.goal_pub    = self.create_publisher(PoseStamped,   '/goal_pose',      10)
        self.metrics_pub = self.create_publisher(String,        '/astar_metrics',  10)
        self.marker_pub  = self.create_publisher(Marker,        '/goal_marker',    10)
        self.markers_pub = self.create_publisher(MarkerArray,   '/patrol_markers', 10)

        self.create_subscription(Odometry,    '/odom',       self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose',  self.goal_callback, 10)
        self.create_subscription(LaserScan,   '/scan',       self.scan_callback, 10)

        self.create_timer(1.0, self.republish)

        # Spawn the goal cylinder in Gazebo once after a short delay
        self._gazebo_goal_spawned = False
        self._spawn_client  = self.create_client(SpawnEntity,  '/spawn_entity')
        self._delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.create_timer(2.0, self._spawn_goal_in_gazebo)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def world_to_grid(self, x, y):
        col = int(round((x - self.origin_x) / self.resolution))
        row = int(round((y - self.origin_y) / self.resolution))
        col = max(0, min(self.grid_width  - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return (row, col)

    def grid_to_world(self, row, col):
        x = col * self.resolution + self.origin_x
        y = row * self.resolution + self.origin_y
        return float(x), float(y)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f'First odom: world ({self.robot_x:.2f},{self.robot_y:.2f})')

        new_start = self.world_to_grid(self.robot_x, self.robot_y)
        if new_start != self.last_start:
            self.last_start = new_start
            self._check_patrol_arrival()
            self._replan(new_start, self.goal)

    def goal_callback(self, msg):
        new_goal = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        if new_goal == self.goal:
            return
        self.goal = new_goal
        self.get_logger().info(f'New goal: grid {self.goal}')
        if self.last_start is not None:
            self._replan(self.last_start, self.goal)

    def scan_callback(self, msg):
        new_dynamic = set()
        angle = msg.angle_min
        for r in msg.ranges:
            angle += msg.angle_increment
            if math.isnan(r) or math.isinf(r):
                continue
            if r < msg.range_min or r > 0.25:   # tighter filter — ignore static walls robot is near
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

        snapshot = frozenset(new_dynamic)
        if snapshot == self._last_dynamic_snapshot:
            self._dynamic_stable_count += 1
        else:
            self._last_dynamic_snapshot = snapshot
            self._dynamic_stable_count  = 1
        if self._dynamic_stable_count < self._DEBOUNCE_SCANS:
            return
        if new_dynamic == self.dynamic_obstacles:
            return

        path_cells    = set(self.path) if self.path else set()
        new_cells     = new_dynamic - self.dynamic_obstacles
        removed_cells = self.dynamic_obstacles - new_dynamic
        path_blocked  = bool(new_cells     & path_cells)
        path_cleared  = bool(removed_cells & path_cells)

        self.obstacles -= self.dynamic_obstacles
        self.dynamic_obstacles = new_dynamic
        self.obstacles |= self.dynamic_obstacles

        if (path_blocked or path_cleared) and self.last_start is not None:
            self.get_logger().info(
                f'Dynamic obstacle change: +{len(new_cells)} -{len(removed_cells)}. Replanning.')
            self._replan(self.last_start, self.goal)

    # ------------------------------------------------------------------
    # Map — rasterised from turtlebot3_house model.sdf
    # Each wall is described as (cx, cy, yaw, length, thickness).
    # We mark every grid cell whose centre falls within the wall box.
    # ------------------------------------------------------------------
    def build_house_obstacles(self):
        # (world_cx, world_cy, yaw_rad, half_length, half_thickness)
        # Derived from model.sdf link poses + collision offsets.
        # Walls with yaw≈0 or π are horizontal (along X); yaw≈±π/2 are vertical (along Y).
        walls = [
            # --- outer walls ---
            # Wall_84: x=-6.2, y=5.275, yaw=π  → horizontal, length=2.75
            (-6.200,  5.275, 0.0,   1.375, 0.10),
            # Wall_82: x=1.25, y=5.269, yaw=π  → horizontal, length=2.75
            ( 1.250,  5.269, 0.0,   1.375, 0.10),
            # Wall_81: x=5.078, y=5.269, yaw=π → horizontal, length=5.0
            ( 5.078,  5.269, 0.0,   2.500, 0.10),
            # Wall_83: x=-2.475, y=5.275, yaw=π → horizontal, length≈4.84 (bottom sill)
            (-2.475,  5.275, 0.0,   2.419, 0.10),
            # Wall_87: x=-6.325, y=-3.925, yaw=0 → horizontal, length=2.5
            (-6.325, -3.925, 0.0,   1.250, 0.10),
            # Wall_94: x=6.2, y=-5.275, yaw=0 → horizontal, length=2.75
            ( 6.200, -5.275, 0.0,   1.375, 0.10),
            # Wall_85: x=-7.5, y=3.1, yaw=-π/2 → vertical, length≈2.21 (solid part)
            (-7.500,  3.1 - 1.147, 1.5708, 1.103, 0.10),
            # Wall_85 upper solid: x=-7.5, y=3.1+1.769
            (-7.500,  3.1 + 1.769, 1.5708, 0.481, 0.10),
            # Wall_86: x=-7.5, y=-1.5, yaw=-π/2 → vertical, multiple segments
            (-7.500, -1.5 - 2.215, 1.5708, 0.285, 0.10),
            (-7.500, -1.5 + 0.052, 1.5708, 0.571, 0.10),
            (-7.500, -1.5 + 2.161, 1.5708, 0.339, 0.10),
            # Wall_88: x=-5.152, y=-1.496, yaw=π/2 → vertical, length=5.0
            (-5.152, -1.496, 1.5708, 2.500, 0.10),
            # Wall_95: x=7.5, y=-2.725, yaw=π/2 → vertical, multiple segments
            ( 7.500, -2.725 - 2.329, 1.5708, 0.296, 0.10),
            ( 7.500, -2.725 - 0.037, 1.5708, 0.638, 0.10),
            ( 7.500, -2.725 + 2.438, 1.5708, 0.187, 0.10),
            # Wall_96: x=7.5, y=2.5, yaw=π/2 → vertical, multiple segments
            ( 7.500,  2.5 - 1.998, 1.5708, 0.752, 0.10),
            ( 7.500,  2.5 + 2.069, 1.5708, 0.681, 0.10),
            # Wall_93: x=4.9, y=-2.725, yaw=-π/2 → vertical, length=5.25
            ( 4.900, -2.725, 1.5708, 2.625, 0.10),
            # Wall_101: x=-0.05, y=3.1, yaw=-π/2 → vertical, length=4.5
            (-0.050,  3.100, 1.5708, 2.250, 0.10),
            # Wall_99: x=-5.15, y=3.1, yaw=π/2 → vertical, two solid segments
            (-5.150,  3.1 - 1.621, 1.5708, 0.629, 0.10),
            (-5.150,  3.1 + 1.837, 1.5708, 0.413, 0.10),
            # --- interior walls ---
            # Wall_104: x=2.3, y=3.1, yaw=π/2 → vertical, two solid segments
            ( 2.300,  3.1 - 0.584, 1.5708, 1.666, 0.10),
            ( 2.300,  3.1 + 2.116, 1.5708, 0.134, 0.10),
            # Wall_106: x=6.2, y=-0.175, yaw=0 → horizontal, two solid segments
            ( 6.2 - 0.903, -0.175, 0.0,   0.472, 0.10),
            ( 6.2 + 0.922, -0.175, 0.0,   0.453, 0.10),
            # Wall_108: x=-1.375, y=-0.175, yaw=0 → horizontal, two solid segments
            (-1.375 - 0.856, -0.175, 0.0,  2.894, 0.10),
            (-1.375 + 3.344, -0.175, 0.0,  0.406, 0.10),
            # Wall_90: x=1.125, y=0.925, yaw=0 → horizontal, length=2.5
            ( 1.125,  0.925, 0.0,   1.250, 0.10),
            # Wall_91: x=2.3, y=0.375, yaw=-π/2 → vertical, two solid segments
            ( 2.300,  0.375 - 0.552, 1.5708, 0.074, 0.10),
            ( 2.300,  0.375 + 0.523, 1.5708, 0.102, 0.10),
            # Wall_92: x=3.6, y=-0.175, yaw=0 → horizontal, length=2.75
            ( 3.600, -0.175, 0.0,   1.375, 0.10),
            # Wall_98: x=-6.325, y=0.925, yaw=0 → horizontal, two solid segments
            (-6.325 - 0.842,  0.925, 0.0,  0.408, 0.10),
            (-6.325 + 0.858,  0.925, 0.0,  0.392, 0.10),
        ]

        res = self.resolution
        for (cx, cy, yaw, hl, ht) in walls:
            # Axis-aligned bounding box of the rotated wall rectangle
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            # Four corners relative to centre
            corners = [
                ( hl * cos_y - ht * sin_y,  hl * sin_y + ht * cos_y),
                (-hl * cos_y - ht * sin_y, -hl * sin_y + ht * cos_y),
                ( hl * cos_y + ht * sin_y,  hl * sin_y - ht * cos_y),
                (-hl * cos_y + ht * sin_y, -hl * sin_y - ht * cos_y),
            ]
            xs = [cx + dx for dx, dy in corners]
            ys = [cy + dy for dx, dy in corners]
            x_min, x_max = min(xs) - res, max(xs) + res
            y_min, y_max = min(ys) - res, max(ys) + res

            c_min = max(0, int(math.floor((x_min - self.origin_x) / res)))
            c_max = min(self.grid_width  - 1, int(math.ceil((x_max - self.origin_x) / res)))
            r_min = max(0, int(math.floor((y_min - self.origin_y) / res)))
            r_max = min(self.grid_height - 1, int(math.ceil((y_max - self.origin_y) / res)))

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    wx = c * res + self.origin_x
                    wy = r * res + self.origin_y
                    # Transform to wall-local frame
                    dx = wx - cx
                    dy = wy - cy
                    local_x =  dx * cos_y + dy * sin_y
                    local_y = -dx * sin_y + dy * cos_y
                    if abs(local_x) <= hl + res * 0.6 and abs(local_y) <= ht + res * 0.6:
                        self.obstacles.add((r, c))

    # ------------------------------------------------------------------
    def inflate_obstacles(self, radius=1):
        inflated = set(self.obstacles)
        for (r, c) in self.obstacles:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                        inflated.add((nr, nc))
        self.obstacles = inflated

    # ------------------------------------------------------------------
    def heuristic(self, a, b):
        # Octile distance — consistent for 8-connected grid
        dr = abs(a[0]-b[0]); dc = abs(a[1]-b[1])
        return max(dr, dc) + (math.sqrt(2)-1)*min(dr, dc)

    def astar(self, start, goal):
        heap = [(0.0, start)]
        came_from = {}
        g = {start: 0.0}
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur in came_from:
                    path.append(cur); cur = came_from[cur]
                path.append(start)
                return list(reversed(path))
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nb = (cur[0]+dr, cur[1]+dc)
                if not (0<=nb[0]<self.grid_height and 0<=nb[1]<self.grid_width):
                    continue
                if nb in self.obstacles:
                    continue
                # FIX A4: block diagonal corner-cutting
                if dr != 0 and dc != 0:
                    if (cur[0]+dr, cur[1]) in self.obstacles or \
                       (cur[0], cur[1]+dc) in self.obstacles:
                        continue
                cost = math.sqrt(2) if (dr and dc) else 1.0
                tg = g[cur] + cost
                if tg < g.get(nb, float('inf')):
                    came_from[nb] = cur; g[nb] = tg
                    heapq.heappush(heap, (tg + self.heuristic(nb, goal), nb))
        return None

    def _check_patrol_arrival(self):
        """Advance to the next patrol waypoint when the robot is close enough."""
        gx, gy = self.grid_to_world(*self.goal)
        dist = math.sqrt((self.robot_x - gx)**2 + (self.robot_y - gy)**2)
        if dist < self._ARRIVAL_RADIUS:
            if not self._patrol_arrived:
                self._patrol_arrived = True
                room_names = ['Left Room', 'Centre Corridor (lower)',
                              'Right Upper Room', 'Bottom Corridor']
                name = room_names[self._patrol_idx % len(room_names)]
                self.get_logger().info(
                    f'[PATROL] Arrived at waypoint {self._patrol_idx + 1}/'
                    f'{len(self._patrol_waypoints)}: {name} '
                    f'world=({gx:.1f},{gy:.1f})'
                )
                # Advance to next waypoint
                self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_waypoints)
                self.goal = self._patrol_waypoints[self._patrol_idx]
                self._patrol_arrived = False
                self._last_published_path = []   # force path republish for new goal
                next_gx, next_gy = self.grid_to_world(*self.goal)
                self.get_logger().info(
                    f'[PATROL] Next goal: waypoint {self._patrol_idx + 1}/'
                    f'{len(self._patrol_waypoints)} '
                    f'grid={self.goal} world=({next_gx:.1f},{next_gy:.1f})'
                )
        else:
            self._patrol_arrived = False

    def _replan(self, start, goal):
        if start in self.obstacles:
            self.get_logger().warn(f'Start {start} in obstacle — skipping.')
            return
        if goal in self.obstacles:
            self.get_logger().warn(f'Goal {goal} in obstacle — skipping.')
            return

        t0 = time.monotonic()
        path = self.astar(start, goal)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if path:
            self.path = path
            self.replan_count  += 1
            self.last_replan_ms = elapsed_ms
            self.last_path_cost = sum(
                math.sqrt(2) if (abs(path[i][0]-path[i-1][0])==1 and
                                 abs(path[i][1]-path[i-1][1])==1) else 1.0
                for i in range(1, len(path)))
            self.get_logger().info(
                f'A* {start}→{goal} steps={len(path)-1} '
                f'cost={self.last_path_cost:.2f} {elapsed_ms:.1f}ms replans={self.replan_count}')
            self._publish_path_if_changed()
            self._publish_metrics()
        else:
            self.get_logger().warn(f'A*: No path from {start} to {goal}')

    # FIX 6: only push path to controller when it actually changed
    def _publish_path_if_changed(self):
        if self.path != self._last_published_path:
            self._last_published_path = list(self.path)
            self.publish_path(self.path)

    def _publish_metrics(self):
        msg = String()
        msg.data = (f'planner=astar replans={self.replan_count} '
                    f'path_steps={len(self.path)-1} '
                    f'path_cost={self.last_path_cost:.3f} '
                    f'replan_ms={self.last_replan_ms:.2f}')
        self.metrics_pub.publish(msg)

    # ------------------------------------------------------------------
    # Republish timer — grid and goal marker always; path only if changed
    # ------------------------------------------------------------------
    def republish(self):
        self.publish_grid()
        self._publish_path_if_changed()
        self.publish_goal()
        self.publish_goal_marker()
        self.publish_patrol_markers()

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
            ps.pose.position.x, ps.pose.position.y = self.grid_to_world(r, c)
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
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = [0] * (self.grid_width * self.grid_height)
        for (r, c) in self.obstacles:
            if 0<=r<self.grid_height and 0<=c<self.grid_width:
                msg.data[r * self.grid_width + c] = 100
        self.grid_pub.publish(msg)

    def publish_goal(self):
        msg = PoseStamped()
        msg.header.frame_id    = 'map'
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = self.grid_to_world(*self.goal)
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    # FIX 4: large red cylinder marker visible in RViz + Gazebo overhead view
    def publish_goal_marker(self):
        """Publish the active goal as a bright green cylinder in RViz."""
        msg = Marker()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.ns              = 'goal'
        msg.id              = 0
        msg.type            = Marker.CYLINDER
        msg.action          = Marker.ADD
        gx, gy = self.grid_to_world(*self.goal)
        msg.pose.position.x  = gx
        msg.pose.position.y  = gy
        msg.pose.position.z  = 0.5
        msg.pose.orientation.w = 1.0
        msg.scale.x = 0.4
        msg.scale.y = 0.4
        msg.scale.z = 1.0
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.lifetime.sec = 0
        self.marker_pub.publish(msg)

    def publish_patrol_markers(self):
        """Publish all patrol waypoints as cylinders in RViz.
        Current target  → bright green,  full opacity,  tall (1.0 m)
        Upcoming        → white/grey,     semi-transparent, short (0.5 m)
        Already visited → dim green,      low opacity,  short (0.5 m)
        """
        room_names = ['Left Room', 'Centre Corridor (lower)',
                      'Right Upper Room', 'Bottom Corridor']
        array = MarkerArray()
        n = len(self._patrol_waypoints)
        now = self.get_clock().now().to_msg()

        for i, wp in enumerate(self._patrol_waypoints):
            wx, wy = self.grid_to_world(*wp)
            is_current = (i == self._patrol_idx)
            # A waypoint is "visited" if the robot has already passed it
            # this lap — i.e. its index is behind the current patrol index
            is_visited = (i < self._patrol_idx)

            m = Marker()
            m.header.frame_id  = 'map'
            m.header.stamp     = now
            m.ns               = 'patrol'
            m.id               = i
            m.type             = Marker.CYLINDER
            m.action           = Marker.ADD
            m.pose.position.x  = wx
            m.pose.position.y  = wy
            m.pose.orientation.w = 1.0

            if is_current:
                # Bright green, tall — the active target
                m.pose.position.z = 0.5
                m.scale.x = 0.40
                m.scale.y = 0.40
                m.scale.z = 1.00
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 1.0
            elif is_visited:
                # Dim green, short — already visited this lap
                m.pose.position.z = 0.25
                m.scale.x = 0.30
                m.scale.y = 0.30
                m.scale.z = 0.50
                m.color.r = 0.0
                m.color.g = 0.5
                m.color.b = 0.0
                m.color.a = 0.4
            else:
                # White/grey, short — upcoming waypoint
                m.pose.position.z = 0.25
                m.scale.x = 0.30
                m.scale.y = 0.30
                m.scale.z = 0.50
                m.color.r = 0.9
                m.color.g = 0.9
                m.color.b = 0.9
                m.color.a = 0.6

            m.lifetime.sec = 0   # permanent
            array.markers.append(m)

            # Label — waypoint number above the cylinder
            label = Marker()
            label.header.frame_id  = 'map'
            label.header.stamp     = now
            label.ns               = 'patrol_labels'
            label.id               = i
            label.type             = Marker.TEXT_VIEW_FACING
            label.action           = Marker.ADD
            label.pose.position.x  = wx
            label.pose.position.y  = wy
            label.pose.position.z  = 1.3 if is_current else 0.9
            label.pose.orientation.w = 1.0
            label.scale.z          = 0.25
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text             = f'{i+1}: {room_names[i]}'
            label.lifetime.sec     = 0
            array.markers.append(label)

        self.markers_pub.publish(array)

    def _spawn_goal_in_gazebo(self):
        """Spawn all patrol waypoints as cylinders in Gazebo at startup."""
        if self._gazebo_goal_spawned:
            return
        if not self._spawn_client.service_is_ready():
            return

        room_names = ['Left_Room', 'Corridor_Upper',
                      'Right_Upper_Room', 'Corridor_Lower']

        for i, wp in enumerate(self._patrol_waypoints):
            gx, gy = self.grid_to_world(*wp)
            # Current target = bright green, others = grey
            if i == self._patrol_idx:
                ambient = '0 1 0 1'
                diffuse = '0 1 0 1'
                emissive = '0 0.6 0 1'
            else:
                ambient = '0.7 0.7 0.7 1'
                diffuse = '0.7 0.7 0.7 1'
                emissive = '0 0 0 1'

            sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="patrol_wp_{i}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <pose>0 0 0.5 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>1.0</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{ambient}</ambient>
          <diffuse>{diffuse}</diffuse>
          <emissive>{emissive}</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
            req = SpawnEntity.Request()
            req.name            = f'patrol_wp_{i}'
            req.xml             = sdf
            req.initial_pose.position.x = gx
            req.initial_pose.position.y = gy
            req.initial_pose.position.z = 0.0
            req.reference_frame = 'world'
            self._spawn_client.call_async(req)

        self._gazebo_goal_spawned = True
        self.get_logger().info(
            f'Spawned {len(self._patrol_waypoints)} patrol waypoint markers in Gazebo.'
        )


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
