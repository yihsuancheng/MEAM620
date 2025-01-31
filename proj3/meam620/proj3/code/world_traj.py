import numpy as np
from scipy.optimize import minimize
# from .graph_search_null import graph_search
from proj3.code.graph_search import graph_search

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
        
        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        
        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        #self.points = np.zeros((1,3)) # shape=(n_pts,3)


        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        # self.resolution = np.array([0.25, 0.25, 0.25])
        # self.margin = 0.5
        self.world = world
        self.start = start
        self.goal = goal
        self.flag = False
        self.reach_goal = False
        self.b = 2

        self.v = 9.6
        #self.v = 8.0
        self.time_scale = 1.65

        self.resolution = np.array([0.255, 0.25, 0.25])
        self.margin = 0.5
        self.average_speed = 3.0

        self.path = self.find_path_with_margin(self.margin)
        #self.points = path_prune(self.path) 

        #self.points = self.generate_sparse_waypoints(self.path)
        self.points = self.path_prune(self.path)

        self.d = np.linalg.norm(self.points[1::]-self.points[0:-1], axis=1).reshape(-1, 1)
        #self.points, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        #self.points = self.find_path_with_margin(self.margin)
        print("self.path.shape", self.path.shape)
        print("path start and end", self.path[0], self.path[-1], self.path.shape)
        print("simplified path start and end", self.points[0], self.points[-1])
        print("simplified full path", self.points)
        print("self.points shape", self.points.shape)
        #self.points = self.path
        self.segment_times, self.total_time = self.compute_segment_times(self.points)
        self.segment_durations = np.diff(self.segment_times)
        print("self.segment_durations", self.segment_durations)
        print("self.segment_times", self.segment_times)

        self.setup_optimization()

    def compute_segment_times(self, waypoints):
        segment_times = [0]
        total_distance = 0
        for i in range(len(waypoints)-1):
            distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
            total_distance += distance
            segment_time = distance / self.average_speed
            segment_times.append(segment_times[-1] + segment_time)
        total_time = total_distance / self.average_speed
        # print("segment times", segment_times)
        # print("total_time", total_time)
        # return segment_times, total_time
    
        time = self.d / self.v
        time[0] = 3 * time[0]   # allow more time to accelerate
        time[-1] = 3 * time[-1] # allow more time to decelerate
        coeff = np.sqrt(self.time_scale / time)
        time = time * coeff

        cumtime = np.vstack((np.zeros(1), np.cumsum(time, axis=0))).flatten()
        print("cum time", cumtime)
        total_time = np.sum(time)
        print("total_time", total_time)
        return cumtime, total_time
    
    def path_prune(self, points):
        pruned_points = list(points)
        count = 0
        while count != len(pruned_points) - 2:
            if count > len(pruned_points) - 2:
                break
            direction = np.cross(pruned_points[count] - pruned_points[count + 1], pruned_points[count + 1] - pruned_points[count + 2])    #check the direction vector
            dist = np.linalg.norm(pruned_points[count] - pruned_points[count + 1])     #check the distance between next two points
            if np.linalg.norm(direction) == 0:
                del pruned_points[count + 1] # remove the middle point as it is redundant
                #print("norm is 0")
                count -= 1
            elif dist > 0.01:
                del pruned_points[count]
                #print("dist > 0.01")
            count += 1

        pruned_points = np.delete(pruned_points, 1, 0)
        #pruned_points = np.array(pruned_points)
        return pruned_points

    def find_path_with_margin(self, min_margin = 0.1):
        path, _ = graph_search(self.world, self.resolution, self.margin, self.start, self.goal, astar=True)

        if path is not None or self.margin <= min_margin:
            return path
        
        # if no path is found
        self.margin -= 1
        return self.find_path_with_margin(min_margin)

    def generate_sparse_waypoints(self, path, epsilon=1.0, is_outermost_call=True):
        """
        Simplifies a path using Ramer-Douglas-Peucker algorithm

        Inputs:
            path: numpy array of shape (N, 3) representing dense path
            epsilon: distance tolerance 

        Returns:
            numpy array of waypoints after simplification
        """
        dmax = 0.0
        index = 0
        for i in range(1, len(path) - 1):
            d = self.perpendicular_distance(path[i], path[0], path[-1])
            if d > dmax:
                index = i
                dmax = d
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            rec_results1 = self.generate_sparse_waypoints(path[:index+1], epsilon, False)
            rec_results2 = self.generate_sparse_waypoints(path[index:], epsilon, False)

            result_list = np.vstack((rec_results1[:-1], rec_results2))

        else:
            result_list = np.array([path[0], path[-1]])

        if is_outermost_call:
            final_path = [result_list[0]]
            for i in range(1, len(result_list)):
                interpolated_waypoints = self.interpolate_waypoints(result_list[i-1], result_list[i], num_points=2)
                final_path.extend(interpolated_waypoints)
                final_path.append(result_list[i])
            result_list = np.array(final_path)

        return result_list
    
        
    def perpendicular_distance(self, point, line_start, line_end):
        """
        Calculates the perpendicular distance of a point from a line.

        Parameters:
            point: The point (np.array) from which the distance is measured.
            line_start: The start point (np.array) of the line.
            line_end: The end point (np.array) of the line.

        Returns:
            The perpendicular distance of the point from the line.
        """
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        return np.linalg.norm(np.cross(line_end-line_start, line_start-point)) / np.linalg.norm(line_end-line_start)

        # Source: https://github.com/fhirschmann/rdp/blob/master/rdp/__init__.py
    
    def interpolate_waypoints(self, start, end, num_points):
        direction = (end - start) / (num_points + 1)
        return [start + direction * i for i in range(1, num_points + 1)]
    
    def setup_optimization(self):
         # minimum jerk trajectory
        # m segments, 6m unknowns
        self.m = np.shape(self.points)[0] - 1

        A = np.zeros((6*self.m, 6*self.m))
        b = np.zeros((6*self.m, 3))

        # set the boundary condition
        p_dot = np.zeros(np.shape(self.points))
        p_ddot = np.zeros(np.shape(self.points))
        p_dddot = np.zeros(np.shape(self.points))

        # set the boundary conditions for A and B
        # first waypoint
        A[0, 0:6] = [0, 0, 0, 0, 0, 1]
        b[0, :] = self.points[0]
        A[1, 0:6] = [0, 0, 0, 0, 1, 0]
        b[1, :] = p_dot[0]
        A[2, 0:6] = [0, 0, 0, 2, 0, 0]
        b[2, :] = p_ddot[0]
        # last waypoint
        end_idx = 6 * self.m - 6
        A[-3, end_idx:end_idx+6] = [self.segment_durations[-1]**5, self.segment_durations[-1]**4, self.segment_durations[-1]**3, self.segment_durations[-1]**2, self.segment_durations[-1], 1]
        b[-3, :] = self.points[-1]
        A[-2, end_idx:end_idx+6] = [5*self.segment_durations[-1]**4, 4*self.segment_durations[-1]**3, 3*self.segment_durations[-1]**2, 2*self.segment_durations[-1], 1, 0]
        b[-2, :] = p_dot[-1]
        A[-1, end_idx:end_idx+6] = [20*self.segment_durations[-1]**3, 12*self.segment_durations[-1]**2, 6*self.segment_durations[-1], 2, 0, 0]
        b[-1, :] = p_ddot[-1]

        # for intermediate and continuity contraints
        for i in range(0, self.m - 1):
            # position constraints
            A[i*6+3, i*6:i*6+6] = [self.segment_durations[i]**5, self.segment_durations[i]**4, self.segment_durations[i]**3, self.segment_durations[i]**2, self.segment_durations[i], 1]
            b[i*6+3, :] = self.points[i+1]
            A[i*6+4, i*6+6:i*6+12] = [0, 0, 0, 0, 0, 1]
            b[i*6+4, :] = self.points[i+1]
            # continuity constraints
            A[i*6+5, i*6:i*6+6] = [5*self.segment_durations[i]**4, 4*self.segment_durations[i]**3, 3*self.segment_durations[i]**2, 2*self.segment_durations[i], 1, 0]
            A[i*6+5, i*6+6:i*6+12] = [0, 0, 0, 0, -1, 0]
            b[i*6+5, :] = 0
            A[i*6+6, i*6:i*6+6] = [20*self.segment_durations[i]**3, 12*self.segment_durations[i]**2, 6*self.segment_durations[i], 2, 0, 0]
            A[i*6+6, i*6+6:i*6+12] = [0, 0, 0, -2, 0, 0]
            b[i*6+6, :] = 0
            A[i*6+7, i*6:i*6+6] = [60*self.segment_durations[i]**2, 24*self.segment_durations[i], 6, 0, 0, 0]
            A[i*6+7, i*6+6:i*6+12] = [0, 0, -6, 0, 0, 0]
            b[i*6+7, :] = 0
            A[i*6+8, i*6:i*6+6] = [120*self.segment_durations[i], 24, 0, 0, 0, 0]
            A[i*6+8, i*6+6:i*6+12] = [0, -24, 0, 0, 0, 0]
            b[i*6+8, :] = 0
            
        print("A shape", A.shape)
        print("B shape", b.shape)

        self.coeff = np.linalg.solve(A, b)
        #print(self.coeff)

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
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        # given time t, calculate the position, velocity, acceleration, jerk, snap
        if t < 0:
            print("t is out of range", t)
            return
        #if t > self.time_duration * self.m:
        if t > self.segment_times[-1]:
            print("t is out of range", t)
            x = self.points[-1]
        
            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
            return flat_output
        
        # find the segment where t is in
        for i in range(0, self.m-1):
            #if t <= self.time_duration * (i + 1):
            if t <= self.segment_times[i + 1]:
                break

        t = t - self.segment_times[i]
        coefficients = self.coeff[i*6:i*6+6]
        x = np.array([t**5, t**4, t**3, t**2, t, 1]) @ coefficients
        x_dot = np.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0]) @ coefficients
        x_ddot = np.array([20*t**3, 12*t**2, 6*t, 2, 0, 0]) @ coefficients
        x_dddot = np.array([60*t**2, 24*t, 6, 0, 0, 0]) @ coefficients
        x_ddddot = np.array([120*t, 24, 0, 0, 0, 0]) @ coefficients

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output