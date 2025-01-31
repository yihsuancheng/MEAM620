#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    # new_p = np.zeros((3, 1))
    # new_v = np.zeros((3, 1))
    # new_q = Rotation.identity()

    # Correct the measurements by subtracting the biases
    R = Rotation.as_matrix(q)

    w_m_corrected = w_m - w_b
    a_m_corrected = a_m - a_b

    # Update the position
    new_p = p + v * dt + 0.5 * (R @ a_m_corrected + g) * dt**2

    # Update the velocity
    new_v = v + (R @ a_m_corrected + g) * dt

    # Update the orientation by integrating the angular velocity
    omega = Rotation.from_rotvec(w_m_corrected.flatten() * dt)
    new_q = q * omega 

    return new_p, new_v, new_q, a_b, w_b, g

def skew_symmetric(v):
    """ 
    Returns the skew symmetric matrix of vector v 
    """

    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    P = error_state_covariance
    R = Rotation.as_matrix(q)
    w_corrected = w_m - w_b
    omega_ = Rotation.from_rotvec(w_corrected.flatten() * dt)
    R_T = omega_.as_matrix().T

    a_corrected = (a_m - a_b).flatten()
    a_skew = skew_symmetric(a_corrected)
    I = np.eye(3)

    F_x = np.zeros((18, 18))
    F_x[:3, :3] = I
    F_x[:3, 3:6] = I * dt
    F_x[3:6, 3:6] = I
    F_x[3:6, 6:9] = -(R @ a_skew) * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[3:6, 15:18] = I * dt
    F_x[6:9, 6:9] = R_T
    F_x[6:9, 12:15] = -I * dt
    F_x[9:12, 9:12] = I
    F_x[12:15, 12:15] = I 
    F_x[15:18, 15:18] = I

    F_i = np.zeros((18, 12))
    F_i[3:6, :3] = I
    F_i[6:9, 3:6] = I 
    F_i[9:12, 6:9] = I
    F_i[12:15, 9:12] = I

    V_i = (accelerometer_noise_density**2) * (dt**2) * I
    Theta_i = (gyroscope_noise_density**2) * (dt**2) * I
    A_i = (accelerometer_random_walk**2) * dt * I
    Omega_i = (gyroscope_random_walk**2) * dt * I

    Q_i = np.zeros((12, 12))
    Q_i[:3, :3] = V_i
    Q_i[3:6, 3:6] = Theta_i
    Q_i[6:9, 6:9] = A_i
    Q_i[9:12, 9:12] = Omega_i

    P_new = (F_x @ P @ F_x.T) + (F_i @ Q_i @ F_i.T)

    # return an 18x18 covariance matrix
    return P_new


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    sigma = error_state_covariance

    innovation = np.zeros((2, 1))
    R = Rotation.as_matrix(q)
    Pc = R.T @ (Pw - p)
    Pc_normalized = (Pc / Pc[2]).reshape(-1, 1)
    innovation = uv - Pc_normalized[:2]

    u_, v_ = Pc_normalized[0], Pc_normalized[1]
    #print("u_", u_)
    if norm(innovation) < error_threshold:
        P_c0 = R.T @ (Pw - p)
        P_c0 = P_c0.reshape(-1)
        d_Pc_d_theta = skew_symmetric(P_c0)
        d_Pc_d_p = -R.T

        d_zt_d_Pc = (1 / Pc[2]) * np.array([[1, 0, -u_[0]],
                                            [0, 1, -v_[0]]])
        
        d_zt_d_theta = d_zt_d_Pc @ d_Pc_d_theta
        d_zt_d_p = d_zt_d_Pc @ d_Pc_d_p

        H_t = np.zeros((2, 18))
        H_t[:, :3] = d_zt_d_p
        H_t[:, 6:9] = d_zt_d_theta

        K_t = sigma @ H_t.T @ np.linalg.inv((H_t @ sigma @ H_t.T) + Q)
        delta_x = K_t @ innovation

        new_sigma = ((np.eye(18) - (K_t @ H_t)) @ sigma @ (np.eye(18) - (K_t @ H_t)).T) + (K_t @ Q @ K_t.T)
        error_state_covariance = new_sigma

        p = p + delta_x[:3]
        v = v + delta_x[3:6]
        q_rot = Rotation.from_rotvec(delta_x[6:9].reshape(-1))
        q = q * q_rot
        a_b = a_b + delta_x[9:12]
        w_b = w_b + delta_x[12:15]
        g = g + delta_x[15:18]

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

