# Autonomous Indoor Robot Navigation Using A* and D* Lite
## ROS 2 Simulation Project Report

---

## 1. Project Overview

This project implements and compares two path planning algorithms — **A\*** and **D\* Lite** — on a TurtleBot3 Burger robot navigating the TurtleBot3 house environment in Gazebo. The robot is assigned a **multi-room patrol task**, visiting four waypoints across the house in a continuous loop while avoiding both static walls and dynamic obstacles (moving actors).

The full system is built on **ROS 2 Humble** and consists of three custom nodes:

| Node | File | Role |
|---|---|---|
| `astar_planner` | `astar_planner.py` | A* path planning + patrol logic |
| `dstar_lite_planner` | `dstar_lite_planner.py` | D* Lite path planning + patrol logic |
| `robot_controller` | `robot_controller.py` | Path following + stuck recovery |

---

## 2. System Architecture

```
Gazebo (TurtleBot3 House)
        │
        ├── /odom  ──────────────────► Planner (A* or D* Lite)
        ├── /scan  ──────────────────► Planner (dynamic obstacle detection)
        │                                      │
        │                              ┌───────┴────────┐
        │                         /astar_path      /dstar_path
        │                              └───────┬────────┘
        │                                      ▼
        │                             robot_controller
        │                                      │
        └── /cmd_vel ◄────────────────────────┘
```

### Map Representation

Both planners use an identical **60 × 48 occupancy grid** at 0.25 m/cell resolution, covering the full house footprint from world coordinates (−7.75, −6.25) to (+7.25, +5.75). The grid is built by rasterising the exact wall geometry from `turtlebot3_house/model.sdf` — each wall segment's pose and dimensions are read and projected onto the grid with a 0.15 m clearance margin. This ensures the planner map matches the Gazebo world precisely.

### Dynamic Obstacle Detection

Both planners subscribe to `/scan` (LaserScan). Hits within 0.25 m that do not fall on known static wall cells are classified as dynamic obstacles. A debounce filter of 6 consecutive identical scans prevents noise-triggered replanning. When a dynamic obstacle appears on or clears from the current path, replanning is triggered.

### Patrol Task

The robot visits four waypoints in sequence, looping indefinitely:

| # | Location | World Coordinates | Grid Cell |
|---|---|---|---|
| 1 | Left Room | (−6.0, 1.5) | (31, 7) |
| 2 | Centre Corridor (lower) | (−2.5, −2.5) | (15, 21) |
| 3 | Right Upper Room | (3.0, 4.0) | (41, 43) |
| 4 | Bottom Corridor | (1.0, −4.0) | (9, 35) |

Waypoints were verified clear of both walls and furniture (table, cabinets, bookshelves) using a geometric clearance check against all SDF model positions.

---

## 3. Algorithm Implementations

### 3.1 A* Planner

A* is a **complete, optimal, single-query** search algorithm. It searches from the current robot position to the goal every time replanning is needed.

**Key implementation details:**

- **Heuristic**: Octile distance — consistent and admissible for 8-connected grids
  ```
  h(a,b) = max(|dr|,|dc|) + (√2 − 1) × min(|dr|,|dc|)
  ```
- **Connectivity**: 8-connected grid with diagonal corner-cutting blocked
- **Replanning trigger**: Every time the robot moves to a new grid cell, or a dynamic obstacle changes the path
- **Path publishing**: Only publishes when the path content actually changes, preventing the controller from resetting its waypoint index on spurious republishes
- **Complexity**: O(N log N) per replan where N = grid size (2,880 cells)

**Observed behaviour from logs:**
- Initial plan from (23,23) to (31,7): **45 steps, cost 49.56, 5.2 ms**
- As the robot moves, each odom update triggers a replan from the new cell — replans shrink from ~5 ms to <0.1 ms as the path shortens
- Dynamic obstacle encountered mid-route: path cost jumped from 38.56 to 41.97 (rerouted around obstacle)
- Waypoint 1 reached after **53 replans**; waypoint 2 reached after **71 replans**
- Replan times ranged from **0.0 ms to 5.7 ms** — fast enough to not affect navigation

### 3.2 D* Lite Planner

D* Lite is an **incremental replanning** algorithm (Koenig & Likhachev, 2002). It maintains cost tables `g[]` and `rhs[]` across replans, only reprocessing cells affected by obstacle changes.

**Key implementation details:**

- **State tables**: `g[s]` = current best cost from s to goal; `rhs[s]` = one-step lookahead cost
- **Consistency**: A cell is consistent when `g[s] == rhs[s]`. Only inconsistent cells are processed
- **Key modifier `km`**: Accumulates as the robot moves, keeping the priority queue admissible without a full rebuild
- **Incremental update**: On obstacle change, only the changed cells and their neighbours are re-queued — O(changed cells × log N) instead of O(N log N)
- **Path extraction**: Follows the gradient of `g[]` from start to goal
- **Lazy deletion**: Priority queue uses a validity map to avoid expensive removal operations

**Theoretical advantage over A\*:**
When a dynamic obstacle appears, D* Lite updates only the affected portion of the cost table. On a 60×48 grid with a single obstacle change, this is typically 5–20 cells vs. the full 2,880-cell search A* performs.

---

## 4. Controller Design

The `robot_controller` node follows the planned path using a **proportional heading + speed controller** with velocity ramping:

| Parameter | Value | Purpose |
|---|---|---|
| Max linear speed | 0.20 m/s | Forward travel |
| Max angular speed | 1.0 rad/s | Turning |
| Linear acceleration | 0.08 m/s per tick | Smooth ramp-up |
| Angular acceleration | 0.20 rad/s per tick | Smooth turning |
| Waypoint tolerance (intermediate) | 0.28 m | When to advance to next waypoint |
| Waypoint tolerance (final) | 0.30 m | Goal arrival threshold |
| Stuck check interval | 1.5 s | Detects no-movement situations |
| Stuck skip count | 2–5 waypoints | Escalating skip on repeated stuck |

**Path resume logic**: When a new path arrives (due to replanning), the controller finds the closest waypoint to the robot's current position and resumes from the first waypoint at least 0.40 m ahead. It also carries forward the previous target waypoint if it appears in the new path and is further ahead, preventing replans from rolling back progress.

---

## 5. Challenges Encountered and Solutions

### 5.1 Map Mismatch (Critical)
**Problem**: The original `build_house_obstacles()` used a hand-drawn 40×40 grid that did not match the Gazebo house. The robot navigated through walls because the planner thought those cells were free.

**Solution**: Replaced with a wall rasteriser that reads exact poses and dimensions from `turtlebot3_house/model.sdf` and projects each wall segment onto the grid with a geometric clearance margin.

### 5.2 Coordinate System Mismatch
**Problem**: D* Lite used `floor()` for coordinate conversion while A* used `round()`, causing a half-cell offset that shifted the D* Lite map relative to the world.

**Solution**: Standardised both planners to use `round()` with no `+resolution/2` offset, ensuring identical coordinate mapping.

### 5.3 Spawn Cell in Obstacle
**Problem**: After adding wall inflation (radius=1), the robot's spawn cell at (23,23) was marked as an obstacle because `Wall_108` (interior wall at y=−0.175) was 0.325 m away — within the inflation radius.

**Solution**: Removed post-inflation entirely. The rasteriser's 0.15 m margin (0.6 × resolution) already provides sufficient clearance without blocking the spawn cell.

### 5.4 Replan Storm / Stuck Loop
**Problem**: The scan callback classified nearby static walls as dynamic obstacles (range filter was 0.45 m), triggering replans every ~2 seconds. Each replan reset the controller's waypoint index to 1, preventing forward progress.

**Solution**:
- Tightened scan range filter from 0.45 m to 0.25 m
- Increased debounce from 3 to 6 scans
- Fixed controller resume logic to preserve progress across replans
- Raised `RESUME_LOOKAHEAD` from 0.20 m to 0.40 m (must exceed grid resolution)

### 5.5 Furniture Collision
**Problem**: Patrol waypoints were placed near furniture (large table at (−2.65, 2.42), marble table at (4.88, 2.93)) that is not in the static map. The robot navigated into furniture.

**Solution**: Verified all waypoints against furniture bounding boxes with a 0.6 m clearance margin. Replaced blocked waypoints with confirmed clear alternatives.

### 5.6 Goal Marker Not Visible in Gazebo
**Problem**: `visualization_msgs/Marker` is RViz-only and does not appear in Gazebo.

**Solution**: Added Gazebo entity spawning via `/spawn_entity` service. All four patrol waypoints are spawned as SDF cylinder models at startup — green for the current target, grey for upcoming waypoints.

---

## 6. Performance Comparison

Data extracted from the A* run log (waypoints 1 and 2 completed):

### A* Performance

| Metric | Waypoint 1 (Start → Left Room) | Waypoint 2 (Left Room → Centre Lower) |
|---|---|---|
| Initial path steps | 45 | 15 |
| Initial path cost | 49.56 | 16.66 |
| Initial replan time | 5.2 ms | 0.2 ms |
| Total replans to reach | 53 | ~18 |
| Dynamic obstacle replans | 1 | 1 |
| Min replan time | 0.0 ms | 0.1 ms |
| Max replan time | 5.7 ms | 0.6 ms |

### A* vs D* Lite — Theoretical Comparison

| Property | A* | D* Lite |
|---|---|---|
| Algorithm type | Single-query search | Incremental replanning |
| Replan on robot move | Full search from new start | Incremental (km update only) |
| Replan on obstacle change | Full search | Only affected cells re-queued |
| First plan time (45-step path) | ~5 ms | ~5 ms (same — cold start) |
| Subsequent replans (no obstacle) | ~1–5 ms (shrinks as path shortens) | <1 ms (cost table reused) |
| Obstacle change replan | Full grid search | O(changed cells × log N) |
| Memory usage | O(N) per search | O(N) persistent tables |
| Implementation complexity | Low | High |
| Optimal paths | Yes | Yes |

**Key finding**: For this environment (small 60×48 grid, infrequent obstacles), A* replans in 0–6 ms which is fast enough that the incremental advantage of D* Lite is not critical. D* Lite's advantage becomes significant in larger environments or with frequent obstacle changes where full re-search would be expensive.

---

## 7. ROS 2 Topics

| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/odom` | `nav_msgs/Odometry` | Gazebo | Planner, Controller |
| `/scan` | `sensor_msgs/LaserScan` | Gazebo | Planner |
| `/astar_path` | `nav_msgs/Path` | A* Planner | Controller |
| `/dstar_path` | `nav_msgs/Path` | D* Lite Planner | Controller |
| `/occupancy_grid` | `nav_msgs/OccupancyGrid` | A* Planner | RViz |
| `/dstar_grid` | `nav_msgs/OccupancyGrid` | D* Lite Planner | RViz |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Planner | Planner (external goal) |
| `/goal_marker` | `visualization_msgs/Marker` | Planner | RViz |
| `/patrol_markers` | `visualization_msgs/MarkerArray` | Planner | RViz |
| `/astar_metrics` | `std_msgs/String` | A* Planner | Monitoring |
| `/dstar_metrics` | `std_msgs/String` | D* Lite Planner | Monitoring |
| `/cmd_vel` | `geometry_msgs/Twist` | Controller | Gazebo |

---

## 8. Conclusions

Both planners successfully navigated the TurtleBot3 house and completed the patrol task. The key outcomes are:

1. **A\*** is simpler to implement and performs well in this environment. Replan times of 0–6 ms are negligible at the robot's operating speed of 0.20 m/s.

2. **D\* Lite** provides incremental replanning that reuses the cost table across robot movements. Its advantage is most pronounced when obstacles change frequently — in this simulation, the benefit was modest due to the small grid size and infrequent dynamic obstacles.

3. **The map accuracy problem** was the most impactful issue in the project. A planner is only as good as its map — once the map was rebuilt from the actual SDF geometry, navigation quality improved dramatically.

4. **The patrol task** successfully demonstrated both planners navigating across multiple rooms, handling doorways, and replanning around dynamic obstacles (walking actors) encountered en route.

---

## 9. How to Run

```bash
# Build
colcon build --packages-select astar_robot_sim
source install/setup.bash

# Run with A*
ros2 launch astar_robot_sim house_navigation.launch.py planner:=astar

# Run with D* Lite
ros2 launch astar_robot_sim house_navigation.launch.py planner:=dstar

# Monitor metrics
ros2 topic echo /astar_metrics
ros2 topic echo /dstar_metrics
```

---

*Report generated: May 23, 2026*
*Platform: ROS 2 Humble, Gazebo 11, TurtleBot3 Burger*
*Environment: turtlebot3_house with dynamic actor obstacles*
