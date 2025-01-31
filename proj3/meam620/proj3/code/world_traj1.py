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

        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.6
        self.average_speed = 2.0

        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        if self.path is None:
            self.flag = True
            self.resolution = np.array([0.2, 0.2, 0.2])
            self.margin = 0.2
            self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        #self.path = self.find_path_with_margin(self.margin)

        #self.points = self.generate_sparse_waypoints(self.path)
        self.points = self.path_prune(self.path)
        self.d = np.linalg.norm(self.points[1::]-self.points[0:-1], axis=1).reshape(-1, 1)
        self.v = 4.0
        self.time_scale = 1.65
        #self.points, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        #self.points = self.find_path_with_margin(self.margin)
        print("self.path.shape", self.path.shape)
        print("path start and end", self.path[0], self.path[-1], self.path.shape)
        print("simplified path start and end", self.points[0], self.points[-1])
        print("simplified full path", self.points)
        print("self.points shape", self.points.shape)
        #self.points = self.path
        self.segment_times, self.total_time = self.compute_segment_times(self.points)
        self.coefficients = self.compute_minimum_jerk_trajectory(self.points)
        print("self.coefficients shape", len(self.coefficients))

    def compute_segment_times(self, waypoints):
        segment_times = [0]
        total_distance = 0
        for i in range(len(waypoints)-1):
            distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
            total_distance += distance
            segment_time = distance / self.average_speed
            segment_times.append(segment_times[-1] + segment_time)
        total_time = total_distance / self.average_speed
        #return segment_times, total_time
    
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
    
    # def compute_segment_times(self, waypoints):
    #     segment_times = [0]
    #     total_distance = 0
    #     a = 3.1
    #     b = 1.1
    #     for i in range(len(waypoints)-1):
    #         distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
    #         total_distance += distance
    #         segment_time = (distance/a)**(1/b)
    #         segment_times.append(segment_times[-1] + segment_time)
    #     total_time = total_distance / self.average_speed
    #     return segment_times, total_time

    def find_path_with_margin(self, min_margin = 0.1):
        path, _ = graph_search(self.world, self.resolution, self.margin, self.start, self.goal, astar=True)

        if path is not None or self.margin <= min_margin:
            return path
        
        # if no path is found
        self.margin -= 1
        return self.find_path_with_margin(min_margin)

    def generate_sparse_waypoints(self, path, epsilon=0.4, is_outermost_call=True):
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
    
    def compute_minimum_jerk_trajectory(self, waypoints):
        self.coefficients = []
        segment_durations = np.diff(self.segment_times) # compute time durations for each segment
        velocities = self.precompute_velocities(waypoints, segment_durations)
        accelerations = self.precompute_accelerations(velocities, segment_durations)
        
        num_segments = len(waypoints)-1
        velocity_threshold = 10 # cannot exceed maximum velocity
        corridor_deviation = 0.1 # Maximum allowed deviation from the path

        for i in range(num_segments):
            p0, p1 = waypoints[i], waypoints[i+1]
            t0, t1 = 0, segment_durations[i]

            seg_coeff = []
            
            for dim in range(3):
                
                pd0 = p0[dim]
                pd1 = p1[dim]

                pdot0 = velocities[i][dim]
                pdot1 = velocities[i+1][dim]

                pddot0 = velocities[i][dim] / (t1 - t0)
                pddot1 = velocities[i+1][dim] / (t1 - t0)
                
                #pddot0 = accelerations[i][dim]
                #pddot1 = accelerations[i+1][dim]
                #print("pddot0 velocities", pddot0)
                #print("pddot1 velocities", pddot1)

                #direction = (p1 - p0) / np.linalg.norm(p1 - p0)
                #pdot0 = pdot1 = self.average_speed * direction # velocity vectors at start and end
                
                #velocity = (pd1 - pd0) / (t1 - t0)               
                #pdot0 = pdot1 = velocity

                # For the last segment at the last waypoint, set the velocity to 0
                # if i == num_segments - 1:
                #     pdot1 = 0
                # else:
                #     pdot1 = velocity
                
                #print("velocity", velocity)
                # pdot0 = pdot1 = self.average_speed

                #pddot0 = pddot1 = np.zeros(3) # Acceleration vectors at start and end
                #pddot0 = pddot1 = 0 # Acceleration  at start and end
                #pddot0 = pddot1 = pdot0 / (t1 - t0)
                
                A, b, H, f = self.setup_optimization(t0, t1, pd0, pd1, pdot0, pdot1, pddot0, pddot1)
                #c_initial = np.zeros((6,3)) # Initial coefficients for x, y, z
                c_initial = np.zeros(6)
                constraints = [{"type": "eq", "fun": lambda c: np.dot(A, c) - b},
                               {"type": "ineq", "fun": lambda c: velocity_threshold - np.abs(self.constrained_velocity(c, t0))},
                               {"type": "ineq", "fun": lambda c: velocity_threshold - np.abs(self.constrained_velocity(c, t1))},
                               #Corridor constraints at the start and end segment
                               {"type": "ineq", "fun": lambda c: corridor_deviation - np.abs(self.constrained_position(c, t0) - pd0)},
                               {"type": "ineq", "fun": lambda c: corridor_deviation - np.abs(self.constrained_position(c, t1) - pd1)}
                ]
                
                result = minimize(lambda c: np.dot(c.T, np.dot(H, c)) + np.dot(f.T, c), c_initial, constraints=constraints)
                if result.success:
                    seg_coeff.append(result.x)
                else:
                    print("Optimization failed", result.message)
                    seg_coeff.append(np.zeros(6))
            # Print velocities for the segment
            # print(f"Segment {i+1}:")
            # print(f"  Start velocity (before threshold): {velocities[i]}")
            # print(f"  End velocity (before threshold): {velocities[i+1]}")
            
            # Append coefficients for all dimensions for current segment
            self.coefficients.append(tuple(seg_coeff))  
            # print(f"Segment {i+1} Coefficients:")
            # for dim, coeffs in enumerate(seg_coeff, start=1):
            #     formatted_coeffs = ", ".join(f"{coeff:.4f}" for coeff in coeffs)
            #     print(f"  Dimension {dim}: {formatted_coeffs}")
        return self.coefficients

    def precompute_velocities(self, waypoints, segment_durations):
        velocities = []
        for i in range(len(waypoints)-1):
            p0, p1 = waypoints[i], waypoints[i+1]
            segment_velocity = []
            for dim in range(3):
                velocity_dim = (p1[dim]-p0[dim]) / segment_durations[i]
                segment_velocity.append(velocity_dim)
            velocities.append(segment_velocity)
        #velocities.append([0, 0, 0]) # last waypoint has velocity 0
        velocities.append(velocities[-1]) 
        #print("velocities", velocities)
        return velocities
    
    def precompute_accelerations(self, velocities, segment_durations):
        accelerations = []
        for i in range(len(velocities)-1):
            v0, v1 = velocities[i], velocities[i+1]
            segment_accelerations = []
            for dim in range(3):
                acceleration_dim = (v1[dim]-v0[dim]) / segment_durations[i]
                segment_accelerations.append(acceleration_dim)
            accelerations.append(segment_accelerations)
        accelerations.append([0, 0, 0])
        #accelerations.append(accelerations[-1])
        #print("accelerations", velocities)
        return accelerations
    
    def constrained_velocity(self, coefficients, t):
        c5, c4, c3, c2, c1, c0 = coefficients
        velocity = 5*t**4*c5 + 4*t**3*c4 + 3*t**2*c3 + 2*t*c2 + c1
        return abs(velocity)
    
    def constrained_position(self, coefficients, t):
        c5, c4, c3, c2, c1, c0 = coefficients
        position = c5*t**5 + c4*t**4 + c3*t**3 + c2*t**2 + c1*t + c0
        return position
        
    def setup_optimization(self, t0, t1, p0, p1, pdot0, pdot1, pddot0, pddot1):
        A = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [t1**5, t1**4, t1**3, t1**2, t1**1, 1],
            [5*t1**4, 4*t1**3, 3*t1**2, 2*t1**1, 1, 0],
            [20*t1**3, 12*t1**2, 6*t1**1, 2, 0, 0]
        ])

        b = np.array([
            p0,
            pdot0,
            pddot0,
            p1,
            pdot1,
            pddot1
        ])

        H = np.array([
            [720*t1**5, 360*t1**4, 120*t1**3, 0, 0, 0],
            [360*t1**4, 192*t1**3, 72*t1**2, 0, 0, 0],
            [120*t1**3, 72*t1**2, 36*t1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        f = np.zeros(6)
        return A, b, H, f

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
        segment_index = len(self.segment_times) - 1
        for i in range(1, len(self.segment_times)):
            if t < self.segment_times[i]:
                segment_index = i - 1
                break
            # else:
            #     segment_index = len(self.segment_times)
            #     break
        #print("segment index", segment_index)
        #print("len of self.segment times", len(self.segment_times))

        # Compute state within the current segment
        if segment_index < len(self.coefficients):
            coefficients = self.coefficients[segment_index]
            t_segment = t - self.segment_times[segment_index] # time since the start of the segment
            #a5, a4, a3, a2, a1, a0 = coefficients

            # Compute position, velocity, acceleration, jerk, and snap for each dimension
            for dim in range(3):
                # a5, a4, a3, a2, a1, a0 = coefficients[dim]
                # x[dim] = a5*t_segment**5 + a4*t_segment**4 + a3*t_segment**3 + a2*t_segment**2 + a1*t_segment + a0
                # x_dot[dim] = 5*a5*t_segment**4 + 4*a4*t_segment**3 + 3*a3*t_segment**2 + 2*a2*t_segment + a1
                # x_ddot[dim] = 20*a5*t_segment**3 + 12*a4*t_segment**2 + 6*a3*t_segment + 2*a2
                # x_dddot[dim] = 60*a5*t_segment**2 + 24*a4*t_segment + 6*a3
                # x_ddddot[dim] = 120*a5*t_segment + 24*a4

                coeffs_dim = coefficients[dim]
                x[dim] = np.polyval(coeffs_dim, t_segment)
                x_dot[dim] = np.polyval(np.polyder(coeffs_dim, 1), t_segment)
                x_ddot[dim] = np.polyval(np.polyder(coeffs_dim, 2), t_segment)
                x_dddot[dim] = np.polyval(np.polyder(coeffs_dim, 3), t_segment)
                x_ddddot[dim] = np.polyval(np.polyder(coeffs_dim, 4), t_segment)
        
        # else:
        #     x = x_dot = x_ddot = x_dddot = x_ddddot = np.zeros(3)
        # else:
        #     if self.coefficients:
        #         last_coefficients = self.coefficients[-1]
        #         for dim in range(3):
        #             x[dim] = last_coefficients[dim][-1]

        if segment_index >= len(self.coefficients):
            #x = self.points[-1]
            x = self.goal
            self.reach_goal = True
            x_dot = np.zeros((3,))
            #print("Reached goal", self.reach_goal)
        # else:
        #     x = self.points[-1]
        #     x_dot = np.zeros((3,))

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        
        # if np.linalg.norm(flat_output['x'][-1] - self.goal) <= 0.05:
        #     print("True")
        # else:
        #     print("False", np.linalg.norm(flat_output['x'][-1] - self.goal))
        #print("flat_output", flat_output)
        return flat_output