# Imports

import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import norm


# %%

def estimate_pose(uvd1, uvd2, pose_iterations, ransac_iterations, ransac_threshold):
    """
    Estimate Pose by repeatedly calling ransac

    :param uvd1:
    :param uvd2:
    :param pose_iterations:
    :param ransac_iterations:
    :param ransac_threshold:
    :return: Rotation, R; Translation, T; inliers, array of n booleans
    """

    R = Rotation.identity()

    for i in range(0, pose_iterations):
        w, t, inliers = ransac_pose(uvd1, uvd2, R, ransac_iterations, ransac_threshold)
        R = Rotation.from_rotvec(w.ravel()) * R

    return R, t, inliers

def solve_w_t(uvd1, uvd2, R0):
    """
    solve_w_t core routine used to compute best fit w and t given a set of stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    """

    # TODO Your code here replace the dummy return value with a value you compute
    #w = t = np.zeros((3,1))
    u1_prime, v1_prime, d1_prime = uvd1
    u2_prime, v2_prime, d2_prime = uvd2

    n = uvd1.shape[1]
    A = np.zeros((2 * n, 6)) 
    b = np.zeros(2 * n)
    for i in range(n):
        y = R0.as_matrix() @ np.array([u2_prime[i], v2_prime[i], 1])
        b[2*i:2*(i+1)] = -np.array([
            [1, 0, -u1_prime[i]], 
            [0, 1, -v1_prime[i]]]) @ y

        A[2*i:2*(i+1), :] = np.array([
            [-u1_prime[i]*y[1], y[2]+u1_prime[i]*y[0], -y[1], d2_prime[i], 0, -u1_prime[i]*d2_prime[i]],
            [-y[2]-v1_prime[i]*y[1], v1_prime[i]*y[0], y[0], 0, d2_prime[i], -v1_prime[i]*d2_prime[i]]])
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    w = x[:3]
    t = x[3:]

    return w.reshape(-1,1), t.reshape(-1,1)


def find_inliers(w, t, uvd1, uvd2, R0, threshold):
    """

    find_inliers core routine used to detect which correspondences are inliers

    :param w: ndarray with 3 entries angular velocity vector in radians/sec
    :param t: ndarray with 3 entries, translation vector
    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2:  3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :param threshold: Threshold to use
    :return: ndarray with n boolean entries : Only True for correspondences that pass the test
    """

    #print("uvd1.shape", uvd1.shape)

    n = uvd1.shape[1]
    #u1_prime, v1_prime, d1_prime = uvd1
    #u2_prime, v2_prime, d2_prime = uvd2

    inliers = []
    I = np.eye(3)
    omega_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
    R_approx = (I + omega_hat) @ R0.as_matrix()

    for i in range(n):
        #A = np.array([[1, 0, -u1_prime[i]], 
        #              [0, 1, -v1_prime[i]]])
        y_prime = R_approx @ np.array([uvd2[0, i], uvd2[1, i], 1])

        #y_prime_2 = R_approx @ np.array([u2_prime[i], v2_prime[i], 1])
        #print("y_prime_2", y_prime_2)

        delta = np.array([[1, 0, -uvd1[0, i]],
                          [0, 1, -uvd1[1, i]]]) @ (y_prime + uvd2[2, i]*t)
        # delta = np.array([[1, 0, -uvd1[0, i]],
        #                   [0, 1, -uvd1[1, i]]]) @ (y_prime + d2_prime[i]*t)
        #print("norm delta", norm(delta))
        if norm(delta) <= threshold:
            inliers.append(True)
        else:
            inliers.append(False)

    # TODO Your code here replace the dummy return value with a value you compute
    return np.array(inliers)


def ransac_pose(uvd1, uvd2, R0, ransac_iterations, ransac_threshold):
    """

    ransac_pose routine used to estimate pose from stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :param ransac_iterations: Number of RANSAC iterations to perform
    :ransac_threshold: Threshold to apply to determine correspondence inliers
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    :return: ndarray with n boolean entries : Only True for correspondences that are inliers

    """
    n = uvd1.shape[1]
    best_inliers_count = 0
    best_w = None
    best_t = None
    best_inliers = np.zeros(n, dtype=bool)

    # When ransac_iterations is zero, consider all correspondences as inliers
    if ransac_iterations == 0:
        best_w, best_t = solve_w_t(uvd1, uvd2, R0)
        best_inliers = np.ones(n, dtype=bool)
    #print("size of n", n)
    for _ in range(ransac_iterations):
        # Randomly sample a subset of indices from the available correspondences
        subset_indices = np.random.choice(n, size=3, replace=False)
        uvd1_subset = uvd1[:, subset_indices]
        uvd2_subset = uvd2[:, subset_indices]

        # Estimate pose from the random subset
        w, t = solve_w_t(uvd1_subset, uvd2_subset, R0)

        t = t.flatten()

        # Find inliers based on the estimated pose
        inliers = find_inliers(w, t, uvd1, uvd2, R0, ransac_threshold)

        #print("inliers are", inliers)
        # Update best estimates if the current one is better
        inliers_count = np.sum(inliers)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_w = w
            best_t = t
            best_inliers = inliers
    print("best_inliers", best_inliers.sum())
    return best_w.reshape(-1,1), best_t.reshape(-1,1), best_inliers
    # TODO Your code here replace the dummy return value with a value you compute
    #w = t = np.zeros((3,1))
    #return w, t, np.zeros(n, dtype='bool')
