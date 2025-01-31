import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.points = points
        self.speed = 3.0

        # Calculate directions and magnitude between waypoints
        self.directions = np.diff(points, axis=0) # compute directions between consecutive segments (N-1)
        self.distances = np.linalg.norm(self.directions, axis=1)
        self.directions = self.directions / self.distances.reshape(-1,1)

        self.durations = self.distances / self.speed
        self.start_times = [0]
        for duration in self.durations:
            self.start_times.append(self.start_times[-1] + duration)
        self.start_times = np.array(self.start_times)
        
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
        #segment_index = np.searchsorted(self.start_times, t, side="right") - 1 # find the t between the segments and the start time of t
        segment_index = -1
        for i in range(len(self.start_times) - 1):
            if self.start_times[i] <= t < self.start_times[i+1]:
                segment_index = i
                break
        if segment_index == -1:
            segment_index = len(self.points) - 2
            
        segment_index = max(min(segment_index, len(self.points) - 2), 0) # make sure segment_index is within bounds
        #print("self.start_times", self.start_times)
        #print("self.distances", self.distances)
        #print("t", t)
        # Calculate current position and velocity
        if t >= self.start_times[-1]: # If t is past the last waypoint, the quadrotor hovers at last waypoint
            x = self.points[-1]
            x_dot = np.zeros((3,))
        else:         
            current_speed = 2.0
            x = self.points[segment_index] + self.directions[segment_index] * self.speed * (t - self.start_times[segment_index]) # Eq.24
            x_dot = self.directions[segment_index] * current_speed     

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
