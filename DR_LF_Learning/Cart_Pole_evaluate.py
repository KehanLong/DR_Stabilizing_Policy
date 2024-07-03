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
#from Cart_Pole_clf_controller import Cart_Pole_CLF_NN_Controller

from Cart_Pole_joint_controller import Cart_Pole_Joint_Controller

#import cvxpy as cp



def V_theta(controller, x):
    # Convert x to a tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)

    # Compute the Lyapunov function using the compute_clf method of the controller
    V, gradV = controller.compute_clf(x)
    
    return V, gradV

# def simulate_trajectory(controller, initial_state, time_steps=2000, dt=0.02):
#     trajectory = [initial_state.detach().numpy().reshape(1,-1)]  # This ensures it starts as a 2D array
#     state = initial_state.clone().detach()  # Starting from a detached copy
#     state.requires_grad = True
#     max_u = 10
    
#     V_values = []
#     relax_values = []

#     for _ in range(time_steps):
#         V, gradV = controller.compute_clf(state)
    
#         # Solve the CLF QP to get the optimal control input
#         u_opt, V_current, relax_current = clf_qp_learned(controller, state, max_u, 10.0)
    
        
#         # Simple Euler integration for now
#         f_x, g_x = controller.cart_pole_dynamics(state)
#         state_dot = f_x + g_x * u_opt
#         state = (state + dt * state_dot).detach()  # Getting a detached tensor after update
#         state.requires_grad = True
        
#         trajectory.append(state.detach().numpy().reshape(1,-1))  # Convert to 2D numpy array immediately
        
#         V_values.append(V_current[0][0])
#         relax_values.append(relax_current)

#     return np.vstack(trajectory), V_values, relax_values # Convert the list of 2D numpy arrays to a single 2D numpy array

def simulate_joint_trajectory(controller, policy, initial_state, time_steps=1000, dt=0.02):
    trajectory = [initial_state.detach().numpy().reshape(1,-1)]  # This ensures it starts as a 2D array
    state = initial_state.clone().detach().float()  # Starting from a detached copy
    state.requires_grad = True
    max_u = 10
    
    V_values = []
    relax_values = []

    for _ in range(time_steps):
        V, gradV = controller.compute_clf(state)
        
        f_x, g_x = controller.cart_pole_dynamics(state)
        
        LfV, LgV = controller.compute_lie_derivatives(state, f_x, g_x)
        
        #loss = controller.lyapunov_derivative_loss(state)
        
        #u_opt = policy(state)
        
        u_opt = controller.compute_policy(state)

        
        
        state_dot = f_x + g_x * u_opt
        state = (state + dt * state_dot).detach()  # Getting a detached tensor after update
        state.requires_grad = True
        
        trajectory.append(state.detach().numpy().reshape(1,-1))  # Convert to 2D numpy array immediately
        
        V_values.append(V.cpu().detach())
        #relax_values.append(loss.cpu().detach())

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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
        # Convert the data to a numpy array if it's a PyTorch tensor
        data = np.array(data)
        
        # Flatten the data if it has more than 2 dimensions
        if len(data.shape) > 1:
            data = data.reshape(-1)
        
        plt.plot(data, lw=2, alpha=0.6)

    plt.xlabel('Time Steps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(title + ".png", dpi=300)
    
    

def plot_control_inputs(controller, state_space, title, xlabel, ylabel):
    u_values = []

    for state in state_space:
        state = torch.tensor(state, dtype=torch.float32)
        u_val = controller.compute_policy(state).item()
        u_values.append(u_val)
    
    x_values = [state[0].item() for state in state_space]
    y_values = [np.arctan2(state[2].item(), state[1].item()) for state in state_space]



    plt.figure(figsize=(10,8))
    plt.scatter(x_values, y_values, c=u_values)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    
    
    
    #---------------------- load NN CLF and controller class -----------------------
    #clf_saved_model = "saved_models/clf_models/cart_pole_30000sample.pt"
    #clf_saved_model = "saved_models/clf_models/cart_pole/22500sample.pt"
    
    
    #---------------------- load joint CLF and policy -----------------------
    
    clf_saved_model = "saved_models/joint_clf_controller_models/cart_pole/baseline_clf.pt"
    
    policy_saved_model = "saved_models/joint_clf_controller_models/cart_pole/baseline_controller.pt"
    
    # clf_saved_model = "saved_models/joint_clf_controller_models/cart_pole/good_clf.pt"
    
    # policy_saved_model = "saved_models/joint_clf_controller_models/cart_pole/good_controller.pt"
    
    
    
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_input = 5
    n_hidden = 128
    n_output = 4
    num_of_layers = 5
    
    n_control_hidden = 64
    
    n_control = 1


    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(policy_saved_model))
    
    clf_controller = Cart_Pole_Joint_Controller(net_nominal, net_policy, relaxation_penalty=0.4)
    
    
    #---------------------- Evaluate the Learned CLF and CLF-QP controller------------------------------
    
    
    state_space = [torch.tensor([pos, np.cos(angle), np.sin(angle), 0.0, 0.0]) for angle in np.linspace(-np.pi, np.pi, 100) for pos in np.linspace(-10, 10, 100)]
    #plot_control_inputs(clf_controller, state_space, "Control Inputs - Position vs Theta", "Position", "Theta")

    
    # List of initial states
    #initial_states = [torch.tensor([position, np.cos(angle), np.sin(angle), 0. , 0.]) for angle in [-np.pi/8, np.pi/6, np.pi/4] for position in [2.7, 1.2, -2.4]]

    # Sample 5 random angles in the negative range and 5 in the positive range avoiding values too close to zero
    negative_angles = np.random.uniform(low=-np.pi/4, high=-np.pi/10, size=5)
    positive_angles = np.random.uniform(low=np.pi/10, high=np.pi/4, size=5)
    angles = np.concatenate((negative_angles, positive_angles), axis=0)
    np.random.shuffle(angles)  # Shuffle to ensure randomness
    
    # Sample 5 random positions in the negative range and 5 in the positive range avoiding values too close to zero
    negative_positions = np.random.uniform(low=-4, high=-1, size=5)
    positive_positions = np.random.uniform(low=1, high=4, size=5)
    positions = np.concatenate((negative_positions, positive_positions), axis=0)
    np.random.shuffle(positions)  # Shuffle to ensure randomness
    
    # Create 10 random initial states
    initial_states = [torch.tensor([positions[i], np.cos(angles[i]), np.sin(angles[i]), 0., 0.]) for i in range(10)]

    
    #results = [simulate_trajectory(clf_controller, init_state) for init_state in initial_states]
    
    results = [simulate_joint_trajectory(clf_controller, net_policy, init_state) for init_state in initial_states]
    
    
    trajectories, V_trajectories, relax_trajectories = zip(*results)
    
    
    # Prepare the data for plotting Position vs Theta
    x_data_position_theta = [traj[:, 0] for traj in trajectories]
    y_data_position_theta = [np.arctan2(traj[:, 2], traj[:, 1]) for traj in trajectories]
    plot_trajectories(trajectories, x_data_position_theta, y_data_position_theta, 'Cart_Pole_Traj_1', 'Position', 'Theta')
    
    # Prepare the data for plotting Linear Velocity vs Angular Velocity
    x_data_velocity = [traj[:, 3] for traj in trajectories]
    y_data_velocity = [traj[:, 4] for traj in trajectories]
    plot_trajectories(trajectories, x_data_velocity, y_data_velocity, 'Cart_Pole_Traj_2', 'Linear Velocity', 'Angular Velocity')
    
    
    number_of_time_steps = 500
    truncated_V_trajectories = [trajectory[:number_of_time_steps] for trajectory in V_trajectories]
    plot_values_over_time(truncated_V_trajectories, 'Lyapunov function values over time', 'V Value')
    plot_values_over_time(relax_trajectories, 'Relaxation values over time', 'Relax Value')

    
    plt.show()



