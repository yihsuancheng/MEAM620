import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # Define PD control gains for position and velocity (translational dynamics)
        self.kp_position = np.array([7, 7, 20]) # proportional gains for x, y, z control
        self.kd_position = np.array([4.0, 4.0, 7]) # derivation gains for x, y, z control

        # Define PD control gains for attitude (rotational dynamics)
        self.kp_phi = np.diag(np.array([3500, 3500, 150]))#2700 # Roll control P gain
        self.kd_phi = np.diag(np.array([150, 150, 80]))#350 # Roll control D gain
        #self.kp_theta = 2500 # Pitch control P gain
        #self.kd_theta = 300 # Pitch control D gain
        #self.kp_psi = 200 # Yaw control P gain
        #self.kd_psi = 100# Yaw control D gain

        self.gamma = self.k_drag / self.k_thrust # k_m/k_f in pdf

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        position = np.array(state["x"])
        velocity = np.array(state["v"])
        quaternion = np.array(state["q"])
        angular_velocity = np.array(state["w"])

        position_desired = np.array(flat_output["x"])
        velocity_desired = np.array(flat_output["x_dot"])
        acceleration_desired = np.array(flat_output["x_ddot"])
        yaw_desired = np.array(flat_output["yaw"])
        yaw_dot_desired = np.array(flat_output["yaw_dot"])

        # Calculate position and velocity error
        position_error = position_desired - position
        velocity_error = velocity_desired - velocity
        #print("velocity_error", velocity_error)
        #print("position_error", position_error)
        
        acceleration_commanded = acceleration_desired + self.kd_position * velocity_error + self.kp_position * position_error
        F_des = self.mass * (acceleration_commanded + np.array([0, 0, self.g]))

        # Compute u1 from Eq.34
        R = Rotation.from_quat(quaternion).as_matrix()
        u1 = np.dot(F_des, R[:, 2])

        b3_des = F_des / np.linalg.norm(F_des)
        a_psi = np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0])
        b2_des = np.cross(b3_des, a_psi) 

        
        b2_des /= np.linalg.norm(b2_des)
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack((b1_des, b2_des, b3_des))

        # Find error vector
        skew_symmetric_matrix = R_des.T @ R - R.T @ R_des
        vec = np.array([skew_symmetric_matrix[2,1], skew_symmetric_matrix[0, 2], skew_symmetric_matrix[1,0]])
        e_R = 0.5 * (vec)

        # Compute u2 from Eq.40
        omega_des = np.zeros(3)
        e_omega = angular_velocity - omega_des
        u2 = self.inertia @ (-self.kp_phi @ e_R - self.kd_phi @ e_omega)

        mx, my, mz = u2 # Moments about x, y, z axes

        # Determine cmd_motor_speeds
        control_matrix = np.array([
            [1, 1, 1, 1],
            [0, self.arm_length, 0, -self.arm_length],
            [-self.arm_length, 0, self.arm_length, 0],
            [self.gamma, -self.gamma, self.gamma, -self.gamma]
        ])

        control_matrix_inv = np.linalg.inv(control_matrix)
        motor_thrusts = np.dot(control_matrix_inv, np.array([u1, mx, my, mz]))
        # Ensure motor thrusts are non negative
        motor_thrusts = np.clip(motor_thrusts, 0, None) # values greater than 0 remain unchanged
        motor_speeds = np.sqrt(motor_thrusts / self.k_thrust)
        motor_speeds = np.clip(motor_speeds, self.rotor_speed_min, self.rotor_speed_max) # Ensure motor speeds are within bounds

        cmd_motor_speeds = motor_speeds
        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(R_des).as_quat()
        #cmd_q = Rotation.from_euler("zyx", [yaw_desired, theta_desired, phi_desired]).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        #print("control_inputs \n", control_input)
        return control_input