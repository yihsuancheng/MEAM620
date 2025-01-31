from heapq import heappush, heappop  # Recommended.
import numpy as np
import math

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.


def heuristic(curr, goal):
    return np.linalg.norm(np.array(curr) - np.array(goal))
    # dx = goal[0] - curr[0]
    # dy = goal[1] - curr[1]
    # dz = goal[2] - curr[2]
    # return math.sqrt(dx**2 + dy**2 + dz**2)

def path(parent, current):
    path = []
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    #occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    #start_index = tuple(occ_map.metric_to_index(start))
    #goal_index = tuple(occ_map.metric_to_index(goal))

    # Return a tuple (path, nodes_expanded)

    occ_map = OccupancyMap(world, resolution, margin)
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    print("start", start)
    print("goal", goal)
    open_set = []
    #heappush(open_set, (0 + heuristic(start_index, goal_index), start_index))
    heappush(open_set, (0 + heuristic(start, goal), start_index))
    parent = {}
    g_score = {start_index: 0}
    closed_set = set()
    nodes_expanded = 0
    
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0: # no movement
                    continue
                directions.append((dx, dy, dz))
#     directions = [
#     (1, 0, 0), (-1, 0, 0), # X axis
#     (0, 1, 0), (0, -1, 0), # Y axis
#     (0, 0, 1), (0, 0, -1), # Z axis
#     (1, 1, 0), (-1, -1, 0),
#     (-1, 1, 0), (1, -1, 0),
#     (1, 0, 1), (-1, 0, -1),
#     (-1, 0, 1), (1, 0, -1),
#     (0, 1, 1), (0, -1, -1),
#     (0, -1, 1), (0, 1, -1),
# ]

    while open_set:
        _, current = heappop(open_set)

        if current in closed_set:
            continue

        nodes_expanded += 1

        if current == goal_index:
            path_ = path(parent, current)
            metric_path = [start] + [occ_map.index_to_metric_center(index) for index in path_] + [goal]
            print(np.array(metric_path), nodes_expanded)
            return np.array(metric_path), nodes_expanded
        
        closed_set.add(current) # shortest path is found, no need to explore the closed nodes again
        
        for direction in directions:
            neighbor = tuple(np.array(current) + np.array(direction))
            if not occ_map.is_valid_index(neighbor) or occ_map.is_occupied_index(neighbor):
                continue

            current_metric = occ_map.index_to_metric_center(current)
            neighbor_metric = occ_map.index_to_metric_center(neighbor)
            #cummulative_g_score = g_score[current] + np.linalg.norm(np.array(direction) * resolution) 
            cummulative_g_score = g_score[current] + np.linalg.norm(neighbor_metric - current_metric)

            if neighbor not in g_score or cummulative_g_score < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = cummulative_g_score
                f_score = cummulative_g_score + heuristic(neighbor_metric, goal) if astar else cummulative_g_score
                # print("heuristic", heuristic(neighbor, goal_index))
                # print("f_score", f_score)
                if neighbor not in closed_set:
                    heappush(open_set, (f_score, neighbor))
            
    return None, nodes_expanded
