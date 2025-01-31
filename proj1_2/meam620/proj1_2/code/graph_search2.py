from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.parent_k = 0
        self.f = float("inf")
        self.g = float("inf")
        self.h = 0

def heuristic(curr, goal):
    return np.linalg.norm(np.array(curr) - np.array(goal))

def path(cells, goal_index, start, goal, occ_map):
    path = [goal]
    current = goal_index
    while not (cells[current].parent_i == current[0] and cells[current].parent_j == current[1] and cells[current].parent_k == current[2]):
        grid_position = (cells[current].parent_i, cells[current].parent_j, cells[current].parent_k)
        metric_position = occ_map.index_to_metric_center(grid_position)
        path.append(metric_position)
        current = grid_position
    path.append(start)
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
   
    open_set = []
    heappush(open_set, (0, start_index))
    cells = {}
    cells[start_index] = Cell()
    cells[start_index].g = 0
    cells[start_index].f = heuristic(start_index, goal_index)
    grid_shape = occ_map.map.shape
    #print("grid_shape", grid_shape)
    nodes_expanded = 0
    closed_set = set()
    
    # directions = []
    # for dx in [-1, 0, 1]:
    #     for dy in [-1, 0, 1]:
    #         for dz in [-1, 0, 1]:
    #             if dx == 0 and dy == 0 and dz == 0: # no movement
    #                 continue
    #             directions.append((dx, dy, dz))
    directions = [
    (1, 0, 0), (-1, 0, 0), # X axis
    (0, 1, 0), (0, -1, 0), # Y axis
    (0, 0, 1), (0, 0, -1), # Z axis
    (1, 1, 0), (-1, -1, 0),
    (-1, 1, 0), (1, -1, 0),
    (1, 0, 1), (-1, 0, -1),
    (-1, 0, 1), (1, 0, -1),
    (0, 1, 1), (0, -1, -1),
    (0, -1, 1), (0, 1, -1),
]
    print(len(directions))

    while open_set:
        _, current = heappop(open_set)

        if current in closed_set:
            continue
        
        nodes_expanded += 1
        if current == goal_index:
            metric_path = path(cells, current, start, goal, occ_map)
            print(np.array(metric_path), len(cells))
            return np.array(metric_path), nodes_expanded
        
        closed_set.add(current)
        
        for direction in directions:
            neighbor = tuple(np.array(current) + np.array(direction))
            if not occ_map.is_valid_index(neighbor) or occ_map.is_occupied_index(neighbor):
                continue
            current_metric = occ_map.index_to_metric_center(current)
            neighbor_metric = occ_map.index_to_metric_center(neighbor)
            cummulative_g_score = cells[current].g + np.linalg.norm(neighbor_metric - current_metric)
            if neighbor not in cells:
                cells[neighbor] = Cell()
            if cummulative_g_score < cells[neighbor].g:
                cells[neighbor].parent_i, cells[neighbor].parent_j, cells[neighbor].parent_k = current
                cells[neighbor].g = cummulative_g_score
                cells[neighbor].f = cummulative_g_score + heuristic(neighbor_metric, goal) if astar else cummulative_g_score
                heappush(open_set, (cells[neighbor].f, neighbor))
                # print("heuristic", heuristic(neighbor, goal_index))
                # print("f_score", f_score)
                # if neighbor not in closed_set:
                #     heappush(open_set, (float(f_score), neighbor))
            
    return None,  nodes_expanded
