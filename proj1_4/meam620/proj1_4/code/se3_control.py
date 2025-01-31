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
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2
        k = self.k_drag / self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1, 1, 1, 1],
                               [0, L, 0, -L],
                               [-L, 0, L, 0],
                               [k, -k, k, -k]])

        self.weight = np.array([0, 0, self.mass * self.g])

        # self.K_p = np.diag([6.85, 6.85, 5])
        # self.K_d = np.diag([4.5, 4.5, 4])
        # self.K_R = np.diag([250, 250, 90])
        # self.K_w = np.diag([20, 20, 15])

        self.K_p = np.diag([3.0, 3.0, 7.0]) 
        self.K_d = np.diag([2.7, 2.7, 4.4]) 
        self.K_R = np.diag([80, 80, 27])
        self.K_w = np.diag([10, 10, 7.5])
        

        # self.K_p = np.diag([6.75, 6.75, 5])
        # self.K_d = np.diag([4.2, 4.2, 4])
        # self.K_R = np.diag([250, 250, 95])
        # self.K_w = np.diag([22, 22, 15])

        # self.K_p = np.diag([6.75, 6.75, 5])
        # self.K_d = np.diag([4.3, 4.3, 4])
        # self.K_R = np.diag([250, 250, 100])
        # self.K_w = np.diag([20, 20, 15])

        # self.K_p = np.diag([7.5, 7.5, 20])
        # self.K_d = np.diag([4.4, 4.4, 7])
        # self.K_R = np.diag([2600, 2600, 150])
        # self.K_w = np.diag([130, 130, 80])





        # STUDENT CODE HERE

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
        # get information from present state and desired output
        x, v, q, w = state['x'], state['v'], state['q'], state['w']
        x_T, v_T, a_T, yaw_T, w_T = flat_output['x'], flat_output['x_dot'], flat_output['x_ddot'], flat_output['yaw'], \
        flat_output['yaw_dot']
        # compute desired thrust force F_des
        a_des = a_T - self.K_d @ (v - v_T) - self.K_p @ (x - x_T)
        F_des = self.mass * a_des + self.weight

        # compute u1
        R = Rotation.from_quat(q).as_matrix()
        b3 = R @ np.array([0, 0, 1])
        u1 = b3.T @ F_des

        b3_des = F_des / np.linalg.norm(F_des)
        a_yaw = np.array([np.cos(yaw_T), np.sin(yaw_T), 0])
        b2_des = np.cross(b3_des, a_yaw) / np.linalg.norm(np.cross(b3_des, a_yaw))

        R_des = np.column_stack((np.cross(b2_des, b3_des), b2_des, b3_des))

        e = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e[2, 1], e[0, 2], e[1, 0]])
        u2 = self.inertia @ (-self.K_R @ e_R - self.K_w @ (w - w_T))
        u2 = u2.reshape(-1, 1)
        u = np.vstack((u1, u2))
        F = np.linalg.inv(self.to_TM) @ u
        # Limit the rotor_speed
        F[F < 0] = 0
        F[F > 2500] = 2500


        # cmd_motor_speeds = np.zeros((4,))
        rotate_speed = np.sqrt(F / self.k_thrust)

        cmd_motor_speeds = rotate_speed
        # print(cmd_motor_speeds)
        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(R_des).as_quat()


        # STUDENT CODE HERE

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input
