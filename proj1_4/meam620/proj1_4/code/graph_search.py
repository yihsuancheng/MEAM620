from heapq import heappush, heappop  # Recommended.
import numpy as np
import math
from flightsim.world import World

from .occupancy_map import OccupancyMap  # Recommended.


def distance(current_node, node_neighbor):
    # return abs(current_node[0] - node_neighbor[0]) + abs(current_node[1] - node_neighbor[1]) + abs(current_node[2] - node_neighbor[2])
    return math.sqrt((current_node[0] - node_neighbor[0]) ** 2 + (current_node[1] - node_neighbor[1]) ** 2 + (current_node[2] - node_neighbor[2]) ** 2)


def heuristic(current_node, goal_node):
    return math.sqrt((current_node[0] - goal_node[0]) ** 2 + (current_node[1] - goal_node[1]) ** 2 + (current_node[2] - goal_node[2]) ** 2)
    # return distance(current_node, goal_node)


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
    occ_map = OccupancyMap(world, resolution, margin)
    print("map:", occ_map.map.shape)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.

    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    print("start_index:", start_index)

    # directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    #               (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    #               (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    #               (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    #               (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    #               (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1),
                  (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                  (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                  (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
                  (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                  (-1, 1, 1), (-1, 1, -1), (-1, -1, 1)]

    heap, path = [], list([goal])
    heappush(heap, (0, start_index))
    nodes_expanded = 0

    nodes_cost = {start_index: 0}
    nodes_pre = {start_index: None}

    weight = 1.5 #if astar else 0
    visited_set = set()

    while heap:
        cost, node_index = heappop(heap)

        if node_index in visited_set:
            continue

        visited_set.add(node_index)
        nodes_expanded += 1

        # reach the goal!
        if node_index == goal_index:
            print("Reach the goal!!!")
            current_index = goal_index
            while current_index != start_index:
                current_index = nodes_pre[current_index]
                current_pos = occ_map.index_to_metric_center(current_index)
                # path = np.vstack([current_pos, path])
                path.insert(0, current_pos)

            path.insert(0, start)
            path = np.array(path)
            # print(path)
            # path = np.vstack([start, path])

            return path, nodes_expanded

        # iterate all directions of the neighbor nodes
        for direction in directions:
            new_node_index = (node_index[0] + direction[0], node_index[1] + direction[1], node_index[2] + direction[2])

            # Skip node if it is occupied or visited.
            if new_node_index in visited_set or occ_map.is_occupied_index(new_node_index):
                continue

            new_nodes_cost = nodes_cost[node_index] + distance(node_index, new_node_index)

            if new_node_index not in nodes_cost or new_nodes_cost < nodes_cost[new_node_index]:
                nodes_cost[new_node_index] = new_nodes_cost
                # if astar:
                new_nodes_cost += weight * heuristic(new_node_index, goal_index)
                heappush(heap, (new_nodes_cost, new_node_index))
                nodes_pre[new_node_index] = node_index

    # Return a tuple (path, nodes_expanded)
    print("No valid path find!!")
    return None, 0