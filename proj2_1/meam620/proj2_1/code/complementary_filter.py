# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    # TODO Your code here - replace the return value with one you compute
    gyro_rotation = Rotation.from_rotvec(angular_velocity * dt)
    gyro_matrix = gyro_rotation.as_matrix()
    R_initial = initial_rotation.as_matrix()
    g = 9.81 #m/s^2
    #gravity_vector = np.array([1, 0, 0])

    acc_normalized = linear_acceleration / norm(linear_acceleration)
    g_prime = R_initial @ gyro_matrix @ acc_normalized
    g_prime_normalized = g_prime / norm(g_prime)

    g_x = g_prime_normalized[0]
    g_y = g_prime_normalized[1]
    g_z = g_prime_normalized[2]

    delta_q_acc = np.array([np.sqrt((1+g_x)/2), 0, g_z/(np.sqrt(2*(1+g_x))), -g_y/(np.sqrt(2*(1+g_x)))])

    #error = np.abs(acc_normalized - 1.0)
    error = np.abs(norm(linear_acceleration) - g)/g
    #print("error", error)
    if error >= 0 and error <= 0.1:
        alpha = 1
    elif error >= 0.2:
        alpha = 0
    else:
        alpha = -10 * error + 2

    q_i = np.array([1, 0, 0, 0])

    delta_q_acc_prime = (1-alpha)*q_i + alpha*delta_q_acc
    delta_q_acc_prime /= norm(delta_q_acc_prime)
    delta_q_acc_quat = np.array([delta_q_acc_prime[1], delta_q_acc_prime[2], delta_q_acc_prime[3], delta_q_acc_prime[0]])

    acc_correction = Rotation.from_quat(delta_q_acc_quat)

    #acc_correction = acc_correction.as_matrix()
    #final_rotation = Rotation.from_matrix((acc_correction @ R_initial @ gyro_matrix))
    final_rotation = acc_correction * initial_rotation * gyro_rotation

    return final_rotation
