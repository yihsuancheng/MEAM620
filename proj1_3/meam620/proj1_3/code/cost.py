import numpy as np
from scipy.optimize import minimize

# Extend the H matrix with zeros for c0 and c1
H = np.zeros((6, 6))
H[2:, 2:] = [
    [720*T**5, 360*T**4, 120*T**3],
    [360*T**4, 192*T**3, 72*T**2],
    [120*T**3, 72*T**2, 36*T]
]

# The f vector is zero in this case since we're only considering the quadratic term
f = np.zeros(6)

# Define the equality constraints matrix C and vector d
# This will depend on your specific boundary conditions for position, velocity, and acceleration
C = np.array([
    [1, 0, 0, 0, 0, 0],  # x(0) condition
    [0, 1, 0, 0, 0, 0],  # x_dot(0) condition
    [0, 0, 2, 0, 0, 0],  # x_ddot(0) condition
    [1, T, T**2, T**3, T**4, T**5],  # x(T) condition
    [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],  # x_dot(T) condition
    [0, 0, 2, 6*T, 12*T**2, 20*T**3]  # x_ddot(T) condition
])

d = np.array([
    x0,  # x(0)
    x_dot0,  # x_dot(0)
    x_ddot0,  # x_ddot(0)
    xT,  # x(T)
    x_dotT,  # x_dot(T)
    x_ddotT  # x_ddot(T)
])

# Define the cost function
def cost_function(c):
    return 0.5 * np.dot(c.T, np.dot(H, c))

# Define the constraints for minimize function
constraints
