#!/usr/bin/env python3
"""
D* Lite planner — true incremental replanning.

Key differences from A*:
  - Maintains rhs[] (one-step lookahead) and g[] (current cost-to-goal) tables.
  - On obstacle change, only the affected cells are re-queued; the rest of the
    cost table is reused — O(changed cells * log N) instead of O(N log N).
  - km accumulates a key modifier as the robot moves, keeping the priority
    queue admissible without a full rebuild.
  - _compute_shortest_path() propagates changes lazily: only inconsistent
    cells (where g != rhs) are processed.

When an obstacle appears on the current path the planner typically updates
in < 1 ms on a 40×40 grid; A* would re-search the whole grid.
"""
import math
import heapq
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

INF = float('inf')


class DStarLitePlanner(Node):
    def __init__(self):
        super().__init__('dstar_lite_planner')
        self.get_logger().info('D* Lite Planner Node started!')

        self.grid_width  = 60
        self.grid_height = 48
        self.resolution  = 0.25
        self.origin_x    = -7.75
        self.origin_y    = -6.25
        self.origin      = self.origin_x

        self.obstacles = set()
        self.build_house_obstacles()
        # No inflation needed — rasterizer margin already provides clearance
        self.static_obstacles = frozenset(self.obstacles)

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False
        self.last_start    = None

        self.goal = (31, 7)    # world (-6.0, 1.5) — left room, opposite side of house
        self.path = []

        # ------------------------------------------------------------------
        # Patrol task — same verified waypoints as A* planner
        # ------------------------------------------------------------------
        self._patrol_waypoints = [
            (31,  7),   # left room
            (15, 21),   # centre lower corridor
            (41, 43),   # right upper room
            ( 9, 35),   # bottom corridor
        ]
        self._patrol_idx     = 0
        self._patrol_arrived = False
        self._ARRIVAL_RADIUS = 0.50
        self.goal = self._patrol_waypoints[self._patrol_idx]
        self.get_logger().info(
            f'Patrol task started. {len(self._patrol_waypoints)} waypoints. '
            f'First goal: {self.goal}'
        )

        self.dynamic_obstacles = set()

        self._last_dynamic_snapshot = frozenset()
        self._dynamic_stable_count  = 0
        self._DEBOUNCE_SCANS        = 6   # more debounce — prevents replan storm

        # FIX A2/A3: metrics
        self.replan_count   = 0
        self.last_path_cost = 0.0
        self.last_replan_ms = 0.0

        # ---------------------------------------------------------------
        # FIX A1: D* Lite internal state
        #   g[s]   — current best cost from s to goal
        #   rhs[s] — one-step lookahead cost from s to goal
        #   A cell is *consistent*  when g[s] == rhs[s].
        #   A cell is *overconsistent* when g[s]  > rhs[s] (can be improved).
        #   A cell is *underconsistent* when g[s] < rhs[s] (path through it
        #            became longer — neighbours must be updated).
        #   km — accumulated key modifier; increases each time start moves,
        #        so we never have to re-insert the whole queue.
        # ---------------------------------------------------------------
        self._g   = {}   # g[cell]   = INF by default
        self._rhs = {}   # rhs[cell] = INF by default
        self._U   = []   # priority queue: (key, cell)
        self._U_set = {} # cell -> best key in queue (for lazy deletion)
        self._km  = 0.0
        self._s_last = None  # last known start position

        self.path_pub    = self.create_publisher(Path,          '/dstar_path',    10)
        self.grid_pub    = self.create_publisher(OccupancyGrid, '/dstar_grid',    10)
        self.goal_pub    = self.create_publisher(PoseStamped,   '/goal_pose',     10)
        self.metrics_pub = self.create_publisher(String,        '/dstar_metrics', 10)
        self._Marker     = Marker
        self.marker_pub  = self.create_publisher(Marker,        '/goal_marker',   10)
        self.markers_pub = self.create_publisher(MarkerArray,   '/patrol_markers', 10)

        self.create_subscription(Odometry,    '/odom',      self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(LaserScan,   '/scan',      self.scan_callback, 10)

        self.create_timer(1.0, self.republish)

        # Spawn goal cylinder in Gazebo once Gazebo is ready
        self._gazebo_goal_spawned = False
        self._spawn_client  = self.create_client(SpawnEntity,  '/spawn_entity')
        self._delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.create_timer(2.0, self._spawn_goal_in_gazebo)

        # Initialise D* Lite tables for the current goal
        self._dstar_init(self.goal)

    # ==================================================================
    # D* Lite core  (Koenig & Likhachev 2002)
    # ==================================================================

    def _g_val(self, s):
        return self._g.get(s, INF)

    def _rhs_val(self, s):
        return self._rhs.get(s, INF)

    def _heuristic(self, a, b):
        """Octile distance — consistent heuristic for 8-connected grid."""
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    def _key(self, s, start):
        g_s   = self._g_val(s)
        rhs_s = self._rhs_val(s)
        k2    = min(g_s, rhs_s)
        k1    = k2 + self._heuristic(start, s) + self._km
        return (k1, k2)

    def _neighbors(self, s):
        r, c = s
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                       (1,1),(1,-1),(-1,1),(-1,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                # FIX A4: block corner-cutting for diagonals
                if dr != 0 and dc != 0:
                    if (r + dr, c) in self.obstacles or \
                       (r, c + dc) in self.obstacles:
                        continue
                yield (nr, nc), (math.sqrt(2) if (dr and dc) else 1.0)

    def _U_push(self, s, key):
        """Push or update s in the lazy-deletion priority queue."""
        if s not in self._U_set or key < self._U_set[s]:
            self._U_set[s] = key
            heapq.heappush(self._U, (key, s))

    def _U_top(self, start):
        """Return (key, cell) for the smallest valid entry."""
        while self._U:
            key, s = self._U[0]
            if self._U_set.get(s) == key:   # still valid
                return key, s
            heapq.heappop(self._U)           # stale — discard
        return None, None

    def _U_pop(self):
        while self._U:
            key, s = heapq.heappop(self._U)
            if self._U_set.get(s) == key:
                del self._U_set[s]
                return key, s
        return None, None

    def _update_vertex(self, s, start):
        """Recompute rhs for s and (re-)insert into U if inconsistent."""
        goal = self.goal
        if s != goal:
            best = INF
            for nb, cost in self._neighbors(s):
                if nb not in self.obstacles:
                    val = cost + self._g_val(nb)
                    if val < best:
                        best = val
            self._rhs[s] = best
        # Remove stale entry (lazy deletion: just mark as invalid)
        if s in self._U_set:
            del self._U_set[s]
        if self._g_val(s) != self._rhs_val(s):
            self._U_push(s, self._key(s, start))

    def _compute_shortest_path(self, start):
        """
        Process the priority queue until start is consistent.
        This is the heart of D* Lite: only inconsistent cells are touched.
        """
        iterations = 0
        while True:
            top_key, _ = self._U_top(start)
            start_key   = self._key(start, start)
            if top_key is None:
                break
            if top_key >= start_key and self._rhs_val(start) == self._g_val(start):
                break
            k_old, u = self._U_pop()
            if k_old is None:
                break
            k_new = self._key(u, start)
            if k_old < k_new:
                # Key has changed — re-insert with updated key
                self._U_push(u, k_new)
            elif self._g_val(u) > self._rhs_val(u):
                # Overconsistent: improve g
                self._g[u] = self._rhs_val(u)
                for nb, _ in self._neighbors(u):
                    if nb not in self.obstacles:
                        self._update_vertex(nb, start)
            else:
                # Underconsistent: raise g, propagate to neighbours
                self._g[u] = INF
                self._update_vertex(u, start)
                for nb, _ in self._neighbors(u):
                    if nb not in self.obstacles:
                        self._update_vertex(nb, start)
            iterations += 1
            if iterations > self.grid_width * self.grid_height * 4:
                break  # safety cap

    def _dstar_init(self, goal):
        """Initialise / re-initialise D* Lite for a new goal."""
        self._g.clear()
        self._rhs.clear()
        self._U.clear()
        self._U_set.clear()
        self._km     = 0.0
        self._s_last = self.last_start

        self._rhs[goal] = 0.0
        self._U_push(goal, self._key(goal, self.last_start or goal))

    def _extract_path(self, start):
        """Follow the gradient of g[] from start to goal."""
        goal = self.goal
        path = [start]
        cur  = start
        seen = {start}
        for _ in range(self.grid_width * self.grid_height):
            if cur == goal:
                break
            best_nb   = None
            best_cost = INF
            for nb, step in self._neighbors(cur):
                if nb in self.obstacles:
                    continue
                total = step + self._g_val(nb)
                if total < best_cost:
                    best_cost = total
                    best_nb   = nb
            if best_nb is None or best_nb in seen:
                return None   # no path or loop
            path.append(best_nb)
            seen.add(best_nb)
            cur = best_nb
        return path if cur == goal else None

    # ==================================================================
    # Map — rasterised from turtlebot3_house model.sdf
    # ==================================================================
    def build_house_obstacles(self):
        walls = [
            # outer walls
            (-6.200,  5.275, 0.0,   1.375, 0.10),
            ( 1.250,  5.269, 0.0,   1.375, 0.10),
            ( 5.078,  5.269, 0.0,   2.500, 0.10),
            (-2.475,  5.275, 0.0,   2.419, 0.10),
            (-6.325, -3.925, 0.0,   1.250, 0.10),
            ( 6.200, -5.275, 0.0,   1.375, 0.10),
            (-7.500,  3.1 - 1.147, 1.5708, 1.103, 0.10),
            (-7.500,  3.1 + 1.769, 1.5708, 0.481, 0.10),
            (-7.500, -1.5 - 2.215, 1.5708, 0.285, 0.10),
            (-7.500, -1.5 + 0.052, 1.5708, 0.571, 0.10),
            (-7.500, -1.5 + 2.161, 1.5708, 0.339, 0.10),
            (-5.152, -1.496, 1.5708, 2.500, 0.10),
            ( 7.500, -2.725 - 2.329, 1.5708, 0.296, 0.10),
            ( 7.500, -2.725 - 0.037, 1.5708, 0.638, 0.10),
            ( 7.500, -2.725 + 2.438, 1.5708, 0.187, 0.10),
            ( 7.500,  2.5 - 1.998, 1.5708, 0.752, 0.10),
            ( 7.500,  2.5 + 2.069, 1.5708, 0.681, 0.10),
            ( 4.900, -2.725, 1.5708, 2.625, 0.10),
            (-0.050,  3.100, 1.5708, 2.250, 0.10),
            (-5.150,  3.1 - 1.621, 1.5708, 0.629, 0.10),
            (-5.150,  3.1 + 1.837, 1.5708, 0.413, 0.10),
            # interior walls
            ( 2.300,  3.1 - 0.584, 1.5708, 1.666, 0.10),
            ( 2.300,  3.1 + 2.116, 1.5708, 0.134, 0.10),
            ( 6.2 - 0.903, -0.175, 0.0,   0.472, 0.10),
            ( 6.2 + 0.922, -0.175, 0.0,   0.453, 0.10),
            (-1.375 - 0.856, -0.175, 0.0,  2.894, 0.10),
            (-1.375 + 3.344, -0.175, 0.0,  0.406, 0.10),
            ( 1.125,  0.925, 0.0,   1.250, 0.10),
            ( 2.300,  0.375 - 0.552, 1.5708, 0.074, 0.10),
            ( 2.300,  0.375 + 0.523, 1.5708, 0.102, 0.10),
            ( 3.600, -0.175, 0.0,   1.375, 0.10),
            (-6.325 - 0.842,  0.925, 0.0,  0.408, 0.10),
            (-6.325 + 0.858,  0.925, 0.0,  0.392, 0.10),
        ]
        res = self.resolution
        for (cx, cy, yaw, hl, ht) in walls:
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
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
                    dx = wx - cx
                    dy = wy - cy
                    local_x =  dx * cos_y + dy * sin_y
                    local_y = -dx * sin_y + dy * cos_y
                    if abs(local_x) <= hl + res * 0.6 and abs(local_y) <= ht + res * 0.6:
                        self.obstacles.add((r, c))

    def inflate_obstacles(self, radius=1):
        """Expand every obstacle cell by radius to keep robot body clear of walls."""
        inflated = set(self.obstacles)
        for (r, c) in self.obstacles:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                        inflated.add((nr, nc))
        self.obstacles = inflated

    # ==================================================================
    # Coordinate helpers
    # ==================================================================
    def world_to_grid(self, x, y):
        # FIX 2: use round() to match A* planner — eliminates coordinate drift
        col = int(round((x - self.origin_x) / self.resolution))
        row = int(round((y - self.origin_y) / self.resolution))
        col = max(0, min(self.grid_width  - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return (row, col)

    def grid_to_world(self, row, col):
        # FIX 2: no +res/2 offset — round-trips cleanly with world_to_grid
        x = col * self.resolution + self.origin_x
        y = row * self.resolution + self.origin_y
        return float(x), float(y)

    # ==================================================================
    # ROS Callbacks
    # ==================================================================
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f'First odom received: ({self.robot_x:.2f}, {self.robot_y:.2f})')
            self._dstar_init(self.goal)

        new_start = self.world_to_grid(self.robot_x, self.robot_y)
        if new_start != self.last_start:
            # FIX A1: update km as robot moves — keeps queue admissible
            if self._s_last is not None and self.last_start is not None:
                self._km += self._heuristic(self._s_last, self.last_start)
            self._s_last  = self.last_start
            self.last_start = new_start
            self._check_patrol_arrival()
            self._replan()

    def goal_callback(self, msg):
        new_goal = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        if new_goal == self.goal:
            return
        self.goal = new_goal
        self.get_logger().info(
            f'New goal received: grid {self.goal}')
        # Full reinit needed for new goal
        self._dstar_init(self.goal)
        if self.last_start is not None:
            self._replan()

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

        # FIX C2: symmetric add/remove of dynamic obstacles
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

        changed = bool(new_cells or removed_cells)
        if not changed:
            return

        if (path_blocked or path_cleared) and self.last_start is not None:
            # FIX A1: incremental update — only re-queue affected cells
            start = self.last_start
            changed_cells = new_cells | removed_cells
            for cell in changed_cells:
                self._update_vertex(cell, start)
                # Also update neighbors of changed cells
                for nb, _ in self._neighbors(cell):
                    if nb not in self.obstacles:
                        self._update_vertex(nb, start)
            self.get_logger().info(
                f'D* Lite incremental update: {len(changed_cells)} cells changed.')
            self._replan()

    # ==================================================================
    # Patrol arrival check
    # ==================================================================
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
                # Advance to next waypoint and reinitialise D* Lite
                self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_waypoints)
                self.goal = self._patrol_waypoints[self._patrol_idx]
                self._patrol_arrived = False
                self._dstar_init(self.goal)   # full reinit for new goal
                next_gx, next_gy = self.grid_to_world(*self.goal)
                self.get_logger().info(
                    f'[PATROL] Next goal: waypoint {self._patrol_idx + 1}/'
                    f'{len(self._patrol_waypoints)} '
                    f'grid={self.goal} world=({next_gx:.1f},{next_gy:.1f})'
                )
        else:
            self._patrol_arrived = False

    # ==================================================================
    # Replanning
    # ==================================================================
    def _replan(self):
        start = self.last_start
        goal  = self.goal
        if start is None:
            return
        if start in self.obstacles:
            self.get_logger().warn(f'Start {start} in obstacle — skipping.')
            return
        if goal in self.obstacles:
            self.get_logger().warn(f'Goal {goal} in obstacle — skipping.')
            return

        # Ensure goal has rhs=0 (can be cleared by a full re-init)
        if self._rhs_val(goal) != 0.0:
            self._rhs[goal] = 0.0
            self._U_push(goal, self._key(goal, start))

        t0 = time.monotonic()
        self._compute_shortest_path(start)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        new_path = self._extract_path(start)
        if new_path:
            self.path = new_path
            self.replan_count   += 1
            self.last_replan_ms  = elapsed_ms
            self.last_path_cost  = sum(
                math.sqrt(2) if (abs(new_path[i][0]-new_path[i-1][0]) == 1 and
                                 abs(new_path[i][1]-new_path[i-1][1]) == 1) else 1.0
                for i in range(1, len(new_path))
            )
            self.get_logger().info(
                f'D* Lite: {start}→{goal}  '
                f'steps={len(new_path)-1}  cost={self.last_path_cost:.2f}  '
                f'{elapsed_ms:.2f}ms  replans={self.replan_count}')
            self.publish_path(self.path)
            self._publish_metrics()
        else:
            self.get_logger().warn(f'D* Lite: No path from {start} to {goal}')

    # FIX C1: metrics for comparison
    def _publish_metrics(self):
        msg = String()
        msg.data = (
            f'planner=dstar_lite '
            f'replans={self.replan_count} '
            f'path_steps={len(self.path)-1} '
            f'path_cost={self.last_path_cost:.3f} '
            f'replan_ms={self.last_replan_ms:.2f}'
        )
        self.metrics_pub.publish(msg)

    # ==================================================================
    # Periodic republish
    # ==================================================================
    def republish(self):
        self.publish_grid()
        if self.path:
            self.publish_path(self.path)
        self.publish_goal()
        self.publish_goal_marker()
        self.publish_patrol_markers()

    # ==================================================================
    # Publishers
    # ==================================================================
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
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
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

    # FIX 6: large red sphere marker — visible in RViz and Gazebo overhead view
    def publish_goal_marker(self):
        """Publish the active goal as a bright green cylinder in RViz."""
        Marker = self._Marker
        msg = Marker()
        msg.header.frame_id    = 'map'
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.ns                 = 'goal'
        msg.id                 = 0
        msg.type               = Marker.CYLINDER
        msg.action             = Marker.ADD
        gx, gy = self.grid_to_world(*self.goal)
        msg.pose.position.x    = gx
        msg.pose.position.y    = gy
        msg.pose.position.z    = 0.5
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
        """Publish all patrol waypoints as cylinders in RViz."""
        Marker     = self._Marker
        room_names = ['Left Room', 'Centre Corridor (lower)',
                      'Right Upper Room', 'Bottom Corridor']
        array = MarkerArray()
        now   = self.get_clock().now().to_msg()

        for i, wp in enumerate(self._patrol_waypoints):
            wx, wy     = self.grid_to_world(*wp)
            is_current = (i == self._patrol_idx)
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
                m.pose.position.z = 0.5
                m.scale.x = 0.40; m.scale.y = 0.40; m.scale.z = 1.00
                m.color.r = 0.0;  m.color.g = 1.0;  m.color.b = 0.0; m.color.a = 1.0
            elif is_visited:
                m.pose.position.z = 0.25
                m.scale.x = 0.30; m.scale.y = 0.30; m.scale.z = 0.50
                m.color.r = 0.0;  m.color.g = 0.5;  m.color.b = 0.0; m.color.a = 0.4
            else:
                m.pose.position.z = 0.25
                m.scale.x = 0.30; m.scale.y = 0.30; m.scale.z = 0.50
                m.color.r = 0.9;  m.color.g = 0.9;  m.color.b = 0.9; m.color.a = 0.6

            m.lifetime.sec = 0
            array.markers.append(m)

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
            label.color.r = 1.0; label.color.g = 1.0
            label.color.b = 1.0; label.color.a = 1.0
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

        for i, wp in enumerate(self._patrol_waypoints):
            gx, gy = self.grid_to_world(*wp)
            if i == self._patrol_idx:
                ambient = '0 1 0 1'; diffuse = '0 1 0 1'; emissive = '0 0.6 0 1'
            else:
                ambient = '0.7 0.7 0.7 1'; diffuse = '0.7 0.7 0.7 1'; emissive = '0 0 0 1'

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
    node = DStarLitePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()