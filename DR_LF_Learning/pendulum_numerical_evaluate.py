#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np

import matplotlib.pyplot as plt

import torch

from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet

from Inverted_pendulum_controller import InvertedPendulum_Joint_Controller





def simulate_joint_trajectory(controller, initial_state, time_steps=300, dt=0.02):
    trajectory = [initial_state.detach().numpy().reshape(1,-1)]  # This ensures it starts as a 2D array
    state = initial_state.clone().detach().float()  # Starting from a detached copy
    state.requires_grad = True

    V_values = []
    relax_values = []

    for _ in range(time_steps):
        V, gradV = controller.compute_clf(state)

    
        # add perturbations 
        f_x, g_x = controller.inverted_pendulum_dynamics(state)
        
        LfV, LgV = controller.compute_lie_derivatives(state, f_x, g_x)
        
        loss = controller.lyapunov_derivative_loss(state)
        
        u_opt = controller.compute_policy(state)
        
        
        state_dot = f_x + g_x * u_opt
        state = (state + dt * state_dot).detach()  # Getting a detached tensor after update
        state.requires_grad = True
        
        trajectory.append(state.detach().numpy().reshape(1,-1))  # Convert to 2D numpy array immediately
        
        V_values.append(V.cpu().detach())
        relax_values.append(loss.cpu().detach())

    return np.vstack(trajectory), V_values, relax_values # Convert the list of 2D numpy arrays to a single 2D numpy array


# Function to plot the trajectories
def plot_trajectories(trajectories, x_data, y_data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    
    for traj_x, traj_y in zip(x_data, y_data):
        # Plot the trajectory
        plt.plot(traj_x, traj_y, lw=2, alpha=0.6)
        
        # Mark the initial state
        plt.scatter(traj_x[0], traj_y[0], c='green', marker='o', s=100, label='Initial State')
        
        # Mark the final state
        plt.scatter(traj_x[-1], traj_y[-1], c='red', marker='x', s=100, label='Final State')

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True)
    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(title + ".png", dpi=300)

def plot_values_over_time(data_list, title, ylabel):
    plt.figure(figsize=(10, 8))
    
    for data in data_list:
        plt.plot(data, lw=2, alpha=0.6)

    plt.xlabel('Time Steps', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True)

    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(title + ".png", dpi=300)



if __name__ == "__main__":
    
    
    
    #---------------------- load NN CLF and controller class -----------------------
    #clf_saved_model = "saved_models/clf_models/inverted_pendulum_200sample.pt"
    #clf_saved_model = "saved_models/clf_models/inverted_pendulum_3600sample.pt"
    
    
    #---------------------- load baseline joint CLF and policy -----------------------
    
    clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_asympto_joint_clf.pt"
    
    policy_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_asympto_controller.pt"
    
    #---------------------- load DRO joint CLF and policy -----------------------
    
    # clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_clf.pt"
    
    # policy_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_controller.pt"
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    n_input = 3
    n_hidden = 64
    n_output = 8
    num_of_layers = 3
    
    n_control_hidden = 32
    n_control = 1
    
    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(policy_saved_model))
    
    
    clf_controller = InvertedPendulum_Joint_Controller(net_nominal, net_policy, relaxation_penalty=0.1)
    

    # desired_state = torch.tensor([0. , 1. , 0.], dtype=torch.float32).unsqueeze(0)
    
    # desired_state.requires_grad = True
    
    # V, gradV = clf_controller.compute_clf(desired_state)
    
    # print('V:', V)
    # print('grad_V:', gradV)

    
    # List of initial states
    #initial_states = [torch.tensor([np.sin(angle), np.cos(angle), velocity]) for angle in [np.pi/2,  np.pi/6, -np.pi/3] for velocity in [1.2, 0.6, -0.8]]
    
    # Sample 5 random angles in the negative range and 5 in the positive range avoiding values too close to zero
    negative_angles = np.random.uniform(low=-np.pi/1.4, high=-np.pi/6, size=5)
    positive_angles = np.random.uniform(low=np.pi/6, high=np.pi/1.4, size=5)
    angles = np.concatenate((negative_angles, positive_angles), axis=0)
    np.random.shuffle(angles)  # Shuffle to ensure randomness
    
    # Sample 5 random positions in the negative range and 5 in the positive range avoiding values too close to zero
    negative_velocity = np.random.uniform(low=-2, high=-0.5, size=5)
    positive_velocity = np.random.uniform(low=0.5, high=2, size=5)
    velocity = np.concatenate((negative_velocity, positive_velocity), axis=0)
    np.random.shuffle(velocity)  # Shuffle to ensure randomness
    
    # Create 10 random initial states
    initial_states = [torch.tensor([np.sin(angles[i]), np.cos(angles[i]), velocity[i]]) for i in range(10)]

    
    #results = [simulate_trajectory(clf_controller, init_state) for init_state in initial_states]
    
    
    results = [simulate_joint_trajectory(clf_controller, init_state) for init_state in initial_states]
    trajectories, V_trajectories, relax_trajectories = zip(*results)
    
    # Prepare the data for plotting Angle vs Angular Velocity
    angles = [np.arctan2(traj[:, 0], traj[:, 1]) for traj in trajectories]
    angular_velocities = [traj[:, 2] for traj in trajectories]
    plot_trajectories(trajectories, angles, angular_velocities, 'Inverted_Pendulum Traj', 'Angle', 'Angular Velocity')

    
    plot_values_over_time(V_trajectories, 'Inverted_Pendulum Lyapunov function', 'V Value')
    plot_values_over_time(relax_trajectories, 'Inverted_Pendulum_Relaxation', 'Relax Value')

    
    plt.show()