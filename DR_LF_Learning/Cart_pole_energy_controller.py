#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import matplotlib.pyplot as plt


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

# def cart_pole_dynamics(x):
#     # Constants
#     mp = 1.0  # mass of pole
#     mc = 1.0  # mass of cart
#     l = 1.0   # length to center of mass of pendulum
#     g = 1.0   # gravity (simplified)

#     # Extract state variables
#     pos, cos_theta, sin_theta, pos_dot, theta_dot = x

#     # Dynamics calculations
#     denominator1 = mc + mp * sin_theta**2
#     denominator2 = l * (mc + mp * sin_theta**2)

#     theta_double_dot = (-mp * l * theta_dot**2 * sin_theta * cos_theta - (mc + mp) * g * sin_theta) / denominator2
#     x_double_dot = (mp * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denominator1
    

#     f_x = np.array([pos_dot, -sin_theta * theta_dot, cos_theta * theta_dot, x_double_dot, theta_double_dot])
#     g_x = np.array([0, 0, 0, 1 / denominator1, -cos_theta / denominator2])

#     return f_x, g_x

def cart_pole_dynamics(x):
    # Constants
    mp = 1.0  # mass of pole
    mc = 1.0  # mass of cart
    l = 1.0   # length to center of mass of pendulum
    g = 1.0   # gravity (simplified)

    # Extract state variables
    pos, theta, pos_dot, theta_dot = x
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Dynamics calculations
    denominator1 = mc + mp * sin_theta**2
    denominator2 = l * (mc + mp * sin_theta**2)

    theta_double_dot = (-mp * l * theta_dot**2 * sin_theta * cos_theta - (mc + mp) * g * sin_theta) / denominator2
    x_double_dot = (mp * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denominator1
    

    f_x = np.array([pos_dot, theta_dot, x_double_dot, theta_double_dot])
    g_x = np.array([0, 0, 1 / denominator1, -cos_theta / denominator2])

    return f_x, g_x
    


def simulate_cart_pole(controller, initial_state, time_steps, dt):
    state = np.array(initial_state)
    trajectory = [state]

    for _ in range(time_steps):
        f_x, g_x = cart_pole_dynamics(state)
        u = controller(state)
        state_dot = f_x + g_x * u
        state = state + state_dot * dt
        state[1] = angle_normalize(state[1])
        
        trajectory.append(state)

    return np.array(trajectory)

# def energy_based_controller(state):
#     theta_dot = state[4]
#     cos_theta = state[1]
#     sin_theta = state[2]
    
#     pos = state[0]
#     pos_dot = state[3]

#     E = 0.5 * theta_dot**2 - cos_theta
#     E_d = 1
#     tilde_E = E - E_d
    

#     k1 = 1.0  # Tunable parameter
    
#     k2 = 0.1
#     k3 = 0.01
#     u = k1 * theta_dot * cos_theta * tilde_E - k2 * pos - k3 * pos_dot
    


#     return u

def energy_based_controller(state):
    theta_dot = state[3]

    theta = state[1]
    
    cos_theta = np.cos(theta)
    
    pos = state[0]
    pos_dot = state[2]

    E = 0.5 * theta_dot**2 - cos_theta
    E_d = 1
    tilde_E = E - E_d
    

    k1 = 10.0  # Tunable parameter
    
    k2 = 0.1
    k3 = 0.01
    u = k1 * theta_dot * cos_theta * tilde_E - k2 * pos - k3 * pos_dot
    


    return u

# Example usage
initial_state = [0, np.cos(np.pi/2), np.sin(np.pi/2), 0, 0]  # pos, cos_theta, sin_theta, pos_dot, theta_dot
initial_state = [0, np.pi/2, 0, 0]  # pos, cos_theta, sin_theta, pos_dot, theta_dot
trajectory = simulate_cart_pole(energy_based_controller, initial_state, time_steps=3000, dt=0.02)

print(trajectory[-20:])
# Convert cos_theta and sin_theta to theta for plotting
theta = np.arctan2(trajectory[:, 2], trajectory[:, 1])
theta = trajectory[:, 1]

# Plotting the trajectory
plt.plot(trajectory[:, 0], theta, label='Trajectory')
plt.scatter(trajectory[0, 0], theta[0], color='green', label='Start')  # Start point
plt.scatter(trajectory[-1, 0], theta[-1], color='red', label='End')    # End point

plt.xlabel('Position')
plt.ylabel('Theta')
plt.title('Cart-Pole Trajectory')
plt.legend()
plt.show()