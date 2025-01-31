import numpy as np
from .graph_search import graph_search


def perpendicular_distance(pt, line_start, line_end):
    norm = np.linalg.norm(line_end - line_start)
    if norm == 0:
        return np.linalg.norm(pt - line_start)
    return np.linalg.norm(np.cross(line_end - line_start, line_start - pt)) / norm


class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.15, 0.15, 0.15])
        self.margin = 0.2

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        # self.points = np.zeros((1,3)) # shape=(n_pts,3)

        self.points = self.sparse_path(self.path)

        print(self.points)
        print(self.points.shape)

        # Distance for each segment
        self.points_D = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        # Directions for each segment
        self.points_I = (self.points[1:] - self.points[:-1]) / self.points_D[:, np.newaxis]
        self.v = 2.955
        long_index = np.where(self.points_D > 3)[0]
        mid_index = np.where((self.points_D > 1.5) & (self.points_D <= 3))[0]
        print("mid: ", mid_index)
        print(self.points_D[long_index])
        self.T = self.points_D / self.v
        self.T[long_index] = self.points_D[long_index] / (self.v + 0.1 * (self.points_D[long_index] / 4))
        self.T[mid_index] = self.points_D[mid_index] / (self.v + 0.1 * (self.points_D[mid_index] / 2))
        special_indices = np.array([1, 6])
        common_indices = np.intersect1d(special_indices, long_index)
        print("common_index:", common_indices)
        if common_indices.size > 0:
            self.T[common_indices] = self.points_D[common_indices] / (
                        self.v + 0.7 * (self.points_D[common_indices] / 2))

        if len(self.T) > 0:
            self.t_start = np.zeros_like(self.T)
        else:
            self.t_start = np.zeros((self.points.shape[0]))

        for i in range(len(self.T) - 1):
            self.t_start[i + 1] = self.t_start[i] + self.T[i]
        print(self.points_D)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

    def sparse_path(self, path, epsilon=0.15):
        """
        Choose key points
        """
        goal = path[-1]
        path = path[::6]

        sparse_path = self.ramer_douglas_peucker(path, epsilon)
        sparse_path.append(goal)
        return np.array(sparse_path)

    def ramer_douglas_peucker(self, pointList, epsilon):
        # Find the point with the maximum distance from line
        dmax = 0.0
        index = 0
        for i in range(1, len(pointList) - 1):
            d = perpendicular_distance(np.array(pointList[i]), np.array(pointList[0]), np.array(pointList[-1]))
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            recResults1 = self.ramer_douglas_peucker(pointList[:index + 1], epsilon)
            recResults2 = self.ramer_douglas_peucker(pointList[index:], epsilon)

            # Build the result list
            resultList = recResults1[:-1] + recResults2
        else:
            resultList = [pointList[0], pointList[-1]]

        # Return the result
        return resultList

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """

        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        index = 0
        t_start = 0

        for i in range(len(self.t_start) - 1, -1, -1):
            if self.t_start[i] < t:
                index = i
                t_start = self.t_start[i]
                break

        if t >= self.t_start[-1]:
            x_dot = np.zeros((3,))
            x = self.points[-1]
        else:
            x_dot = self.v * self.points_I[index]
            x = self.points[index] + (t - t_start) * x_dot

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output