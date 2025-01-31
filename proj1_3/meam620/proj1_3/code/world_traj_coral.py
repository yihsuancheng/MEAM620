import numpy as np

from .graph_search import graph_search
# from graph_search import graph_search

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
        self.resolution = np.array([0.20, 0.20, 0.20])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        self.flag = False   # if the map is complicated and the path might not be straight

        if self.path is None:
            self.flag = True
            self.resolution = np.array([0.20, 0.20, 0.20])
            self.margin = 0.2

            self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        
        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)
        # omit redundant points
        self.points[0] = start
        self.dist = 1.5   # can be tuned
        if self.flag == True:
            self.dist = 0.55
        for i in range(1, len(self.path) - 1):
            # if np.linalg.norm(self.path[i] - self.points[-1]) > self.dist or if it's in the same line of the last two points
            if np.linalg.norm(self.path[i] - self.points[-1]) > self.dist:
                self.points = np.vstack((self.points, self.path[i]))
                continue
            # if the map is complicated and the path might not be straight
            if self.flag == True and (len(self.points) > 1):
                B = self.points[-1]
                A = self.points[-2]
                C = self.path[i]
                cross_product = np.cross(B - A, C - B)
                if np.linalg.norm(cross_product) > 0.00001:   # if the point is not in the same line of the last two points
                    self.points = np.vstack((self.points, self.path[i]))
            

        # add the goal point
        self.points[-1] = goal

        print("self.points.shape: ", self.points.shape)



        # minimum jerk trajectory
        # m segments, 6m unknowns
        self.m = np.shape(self.points)[0] - 1
        
        self.time_duration = 0.768   # for each segment, can be tuned
        if self.flag == True:
            self.time_duration = 0.43
        

        A = np.zeros((6*self.m, 6*self.m))
        b = np.zeros((6*self.m, 3))

        # set the boundary condition
        # can be changed
        p_dot = np.zeros(np.shape(self.points))
        p_ddot = np.zeros(np.shape(self.points))
        p_dddot = np.zeros(np.shape(self.points))


        # according to the boundary condition, set the first and last 3 rows of A and b
        # the first point
        A[0, 0:6] = [0, 0, 0, 0, 0, 1]
        b[0, :] = self.points[0]
        A[1, 0:6] = [0, 0, 0, 0, 1, 0]
        b[1, :] = p_dot[0]
        A[2, 0:6] = [0, 0, 0, 2, 0, 0]
        b[2, :] = p_ddot[0]
        # the last point
        A[6*self.m-3, 6*self.m-6:6*self.m] = [self.time_duration**5, self.time_duration**4, self.time_duration**3, self.time_duration**2, self.time_duration, 1]
        b[6*self.m-3, :] = self.points[self.m]
        A[6*self.m-2, 6*self.m-6:6*self.m] = [5*self.time_duration**4, 4*self.time_duration**3, 3*self.time_duration**2, 2*self.time_duration, 1, 0]
        b[6*self.m-2, :] = p_dot[self.m]
        A[6*self.m-1, 6*self.m-6:6*self.m] = [20*self.time_duration**3, 12*self.time_duration**2, 6*self.time_duration, 2, 0, 0]
        b[6*self.m-1, :] = p_ddot[self.m]

        # set the continuity condition
        for i in range(0, self.m):
            p_dot[i] = (self.points[i+1] - self.points[i]) / self.time_duration
            p_ddot[i] = (p_dot[i+1] - p_dot[i]) / self.time_duration
            p_dddot[i] = (p_ddot[i+1] - p_ddot[i]) / self.time_duration

        # according to the continuity condition, set the middle 6 rows of A and b
        for i in range(0, self.m - 1):
            A[i*6+3, i*6:i*6+6] = [self.time_duration**5, self.time_duration**4, self.time_duration**3, self.time_duration**2, self.time_duration, 1]
            b[i*6+3, :] = self.points[i+1]
            A[i*6+4, i*6+6:i*6+12] = [0, 0, 0, 0, 0, 1]
            b[i*6+4, :] = self.points[i+1]
            A[i*6+5, i*6:i*6+6] = [5*self.time_duration**4, 4*self.time_duration**3, 3*self.time_duration**2, 2*self.time_duration, 1, 0]
            A[i*6+5, i*6+6:i*6+12] = [0, 0, 0, 0, -1, 0]
            b[i*6+5, :] = 0
            A[i*6+6, i*6:i*6+6] = [20*self.time_duration**3, 12*self.time_duration**2, 6*self.time_duration, 2, 0, 0]
            A[i*6+6, i*6+6:i*6+12] = [0, 0, 0, -2, 0, 0]
            b[i*6+6, :] = 0
            A[i*6+7, i*6:i*6+6] = [60*self.time_duration**2, 24*self.time_duration, 6, 0, 0, 0]
            A[i*6+7, i*6+6:i*6+12] = [0, 0, -6, 0, 0, 0]
            b[i*6+7, :] = 0
            A[i*6+8, i*6:i*6+6] = [120*self.time_duration, 24, 0, 0, 0, 0]
            A[i*6+8, i*6+6:i*6+12] = [0, -24, 0, 0, 0, 0]
            b[i*6+8, :] = 0
            
        
         
        print("A.shape: ", A.shape)
        print("b.shape: ", b.shape)
        # solve the linear equation
        self.coeff = np.linalg.solve(A, b)
        print(self.coeff)



        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

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
            print("t is out of range")
            print("t: ", t)
            return
        if t > self.time_duration * self.m:
            print("t is out of range")
            print("t: ", t)
            x = self.points[-1]
            # x_dot = np.zeros((3,))
            # x_ddot = np.zeros((3,))
            # x_dddot = np.zeros((3,))
            # x_ddddot = np.zeros((3,))
            # yaw = 0
            # yaw_dot = 0
            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
            return flat_output
        
        # find the segment where t is in
        for i in range(0, self.m):
            if t <= self.time_duration * (i + 1):
                break

        t = t - self.time_duration * i
        x = np.array([t**5, t**4, t**3, t**2, t, 1]).dot(self.coeff[i*6:i*6+6])
        x_dot = np.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0]).dot(self.coeff[i*6:i*6+6])
        x_ddot = np.array([20*t**3, 12*t**2, 6*t, 2, 0, 0]).dot(self.coeff[i*6:i*6+6])
        x_dddot = np.array([60*t**2, 24*t, 6, 0, 0, 0]).dot(self.coeff[i*6:i*6+6])
        x_ddddot = np.array([120*t, 24, 0, 0, 0, 0]).dot(self.coeff[i*6:i*6+6])

        # calculate the yaw and yaw_dot if needed

        print("x: ", x)
        print("x_dot: ", x_dot)
        print("x_ddot: ", x_ddot)
        print("x_dddot: ", x_dddot)
        print("x_ddddot: ", x_ddddot)
        print("yaw: ", yaw)
        print("yaw_dot: ", yaw_dot)



        # for i in range(0, m - 1):
        #     # first row
        #     A[i*12, i*6:i*6+6] = [0, 0, 0, 0, 0, 1]
        #     b[i*12, :] = self.points[i]
        #     # second row
        #     A[i*12+1, i*6:i*6+6] = [0, 0, 0, 0, 1, 0]
        #     b[i*12+1, :] = p_dot[i]
        #     # third row
        #     A[i*12+2, i*6:i*6+6] = [0, 0, 0, 2, 0, 0]
        #     b[i*12+2, :] = p_ddot[i]
        #     # fourth row
        #     A[i*12+3, i*6+6:i*6+12] = [time_duration**5, time_duration**4, time_duration**3, time_duration**2, time_duration, 1]
        #     b[i*12+3, :] = self.points[i+1]
        #     # fifth row
        #     A[i*12+4, i*6+6:i*6+12] = [5*time_duration**4, 4*time_duration**3, 3*time_duration**2, 2*time_duration, 1, 0]
        #     b[i*12+4, :] = p_dot[i+1]
        #     # sixth row
        #     A[i*12+5, i*6+6:i*6+12] = [20*time_duration**3, 12*time_duration**2, 6*time_duration, 2, 0, 0]
        #     b[i*12+5, :] = p_ddot[i+1]
            
        #     # seventh row
        #     A[i*12+6, i*6:i*6+6] = [time_duration**5, time_duration**4, time_duration**3, time_duration**2, time_duration, 1]
        #     b[i*12+6, :] = self.points[i+1]
        #     # eighth row
        #     A[i*12+7, i*6+6:i*6+12] = [0, 0, 0, 0, 0, 1]
        #     b[i*12+7, :] = self.points[i+1]
        #     # ninth row
        #     A[i*12+8, i*6:i*6+6] = [5*time_duration**4, 4*time_duration**3, 3*time_duration**2, 2*time_duration, 1, 0]
        #     A[i*12+8, i*6+6:i*6+12] = [0, 0, 0, 0, -1, 0]
        #     b[i*12+8, :] = 0
        #     # tenth row
        #     A[i*12+9, i*6:i*6+6] = [20*time_duration**3, 12*time_duration**2, 6*time_duration, 2, 0, 0]
        #     A[i*12+9, i*6+6:i*6+12] = [0, 0, 0, -2, 0, 0]
        #     b[i*12+9, :] = 0
        #     # eleventh row
        #     A[i*12+10, i*6:i*6+6] = [60*time_duration**2, 24*time_duration, 6, 0, 0, 0]
        #     A[i*12+10, i*6+6:i*6+12] = [0, 0, -6, 0, 0, 0]
        #     b[i*12+10, :] = 0
        #     # twelfth row
        #     A[i*12+11, i*6:i*6+6] = [120*time_duration, 24, 0, 0, 0, 0]
        #     A[i*12+11, i*6+6:i*6+12] = [0, -24, 0, 0, 0, 0]
        #     b[i*12+11, :] = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
    