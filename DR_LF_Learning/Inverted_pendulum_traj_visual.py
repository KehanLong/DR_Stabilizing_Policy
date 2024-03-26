#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:13:16 2023

@author: kehan
"""

import numpy as np

import matplotlib.pyplot as plt

import torch

from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet

from Inverted_pendulum_controller import InvertedPendulum_Joint_Controller



def simulate_joint_trajectory(controller, initial_state, time_steps=4000, dt=0.02):
    trajectory = [initial_state.detach().numpy().reshape(1,-1)]  # This ensures it starts as a 2D array
    state = initial_state.clone().detach().float()  # Starting from a detached copy
    state.requires_grad = True
    
    V_values = []
    relax_values = []

    for _ in range(time_steps):

        V, gradV = controller.compute_clf(state)

    
        # add perturbations 
        f_x, g_x = controller.inverted_pendulum_dynamics(state, m=controller.m, l=controller.l, b=controller.b)
        
        LfV, LgV = controller.compute_lie_derivatives(state, f_x, g_x)
        
        
        loss = controller.lyapunov_derivative_loss(state, torch.zeros((2,2)))
        
        #loss = 0
        
        u_opt = controller.compute_policy(state)
        
        
        state_dot = f_x + g_x * u_opt
        
        state = (state + dt * state_dot).detach()  # Getting a detached tensor after update
        state.requires_grad = True
        
        trajectory.append(state.detach().numpy().reshape(1,-1))  # Convert to 2D numpy array immediately
        
        
        V_values.append(V.cpu().detach().squeeze())
        
        #print(V_values)
        relax_values.append(loss.cpu().detach())
        
        #relax_values.append(0)

    return np.vstack(trajectory), V_values, relax_values # Convert the list of 2D numpy arrays to a single 2D numpy array


# Function to plot the trajectories
def plot_trajectories(trajectories, x_data, y_data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    
    for traj_x, traj_y in zip(x_data, y_data):
        # Plot the trajectory
        plt.plot(traj_x, traj_y, lw=3, alpha=0.6)
        
        # Mark the initial state
        plt.scatter(traj_x[0], traj_y[0], c='green', marker='o', s=100, label='Initial State')
        
        # Mark the final state
        plt.scatter(traj_x[-1], traj_y[-1], c='red', marker='x', s=100, label='Final State')

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.title(title, fontsize=24)
    
    plt.grid(True)

    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize = 24)

    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(title + ".png", dpi=300)

def plot_values_over_time(data_list, title, ylabel):
    plt.figure(figsize=(10, 8))
    
    for data in data_list:
        plt.plot(data, lw=2, alpha=0.6)

    plt.xlabel('Time Steps', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=24)
    #plt.grid(True)
    
    plt.legend(fontsize=22)

    plt.tight_layout()

    # Save the figure in high resolution
    #plt.savefig(title + ".png", dpi=300)
    
    
def evaluate_clf_controller(clf_controller, model_type="Baseline"):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate a grid of points in the (theta, omega) plane
    theta = np.linspace(-np.pi * 2, np.pi * 2, 100)
    omega = np.linspace(-8, 8, 100)
    Theta, Omega = np.meshgrid(theta, omega)
    
    # Convert theta to cos(theta) and sin(theta)
    cos_theta = np.cos(Theta.ravel())
    sin_theta = np.sin(Theta.ravel())
    
    # Convert meshgrid to a tensor
    X_tensor = torch.Tensor(np.vstack([sin_theta, cos_theta, Omega.ravel()]).T)
    X_tensor.requires_grad = True
    
    # Initialize arrays to store the values of V and V_dot
    V_values = np.zeros_like(Theta)
    V_dot_values = np.zeros_like(Theta)
    U_values = np.zeros_like(Theta)
    
    # Define the desired state
    
    for i in range(X_tensor.shape[0]):
        # Compute V
        V, gradV = clf_controller.compute_clf(X_tensor[i])
        
        f_x, g_x = clf_controller.inverted_pendulum_dynamics(X_tensor[i])
    
        u_opt = clf_controller.compute_policy(X_tensor[i])
        

        
        # Compute LfV and LgV using the optimal control input
        LfV, LgV = clf_controller.compute_lie_derivatives(X_tensor[i], f_x, g_x)
    
        V_dot = LfV.detach().cpu().numpy() + LgV.detach().cpu().numpy() * u_opt.detach().cpu().numpy()
        
    
        # Convert V and V_dot to numpy and store them in the arrays
        V_values[i // Theta.shape[1], i % Theta.shape[1]] = V.item()
        V_dot_values[i // Theta.shape[1], i % Theta.shape[1]] = V_dot.item()
        
        U_values[i // Theta.shape[1], i % Theta.shape[1]] = u_opt.item()
    
    
    mask = np.zeros_like(V_dot_values)
    mask[V_dot_values >= 0] = 1
    
    # Set the violations (positive values) to zero for plotting purposes
    V_dot_plot = np.where(V_dot_values >= 0, 0, V_dot_values)
    
    # Identify the violation points
    violation_points = np.column_stack([Theta[mask == 1], Omega[mask == 1]])
    
    #plt.savefig("Inverted_pendulum_Vdot.png", dpi=300)
    
    #plt.show()
    
    plt.figure(figsize=(14, 6))

    # Plot for the Lyapunov function values
    plt.subplot(1, 2, 1)
    contour = plt.contourf(Theta, Omega, V_values, levels=50, cmap='viridis')
    plt.colorbar(contour, label='V value')
    plt.xlabel('Theta (radians)', fontsize=16)
    plt.ylabel('Omega (rad/s)', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'{model_type} Lyapunov Function', fontsize=18)

    # Plot for the control policy
    plt.subplot(1, 2, 2)
    contour = plt.contourf(Theta, Omega, U_values, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Control input u')
    plt.xlabel('Theta (radians)', fontsize=16)
    plt.ylabel('Omega (rad/s)', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'{model_type} Control Policy', fontsize=18)

    # Adding a major title for the figure
    plt.suptitle(f"{model_type} Lyapunov Evaluation", fontsize=20, y=1.05)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure with an appropriate name based on the model type
    #plt.savefig(f"{model_type}_Lyapunov_Evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def simulate_and_plot_trajectories(baseline_controller, dro_controller, simulate_joint_trajectory, plot_trajectories, plot_values_over_time):
    """
    Simulates and plots trajectories for given controllers.

    Parameters:
    - baseline_controller: The baseline controller object.
    - dro_controller: The distributionally robust controller object.
    - simulate_joint_trajectory: Function to simulate joint trajectory given a controller and initial state.
    - plot_trajectories: Function to plot trajectories.
    - plot_values_over_time: Function to plot values over time.
    """
    # Sample angles and velocities
    negative_angles = np.random.uniform(low=-np.pi, high=-np.pi/10, size=5)
    positive_angles = np.random.uniform(low=np.pi/10, high=np.pi, size=5)
    angles = np.concatenate((negative_angles, positive_angles), axis=0)
    np.random.shuffle(angles)  # Shuffle to ensure randomness

    negative_velocity = np.random.uniform(low=-6, high=-0.2, size=5)
    positive_velocity = np.random.uniform(low=0.2, high=6, size=5)
    velocity = np.concatenate((negative_velocity, positive_velocity), axis=0)
    np.random.shuffle(velocity)  # Shuffle to ensure randomness

    # Create initial states
    initial_states = [torch.tensor([np.sin(angles[i]), np.cos(angles[i]), velocity[i]]) for i in range(10)]
    
    #print('initial_states:', initial_states)

    for controller_type, controller in [('Baseline', baseline_controller), ('DR', dro_controller)]:
        print(f"Simulating for {controller_type} controller")

        results = [simulate_joint_trajectory(controller, init_state) for init_state in initial_states]
        trajectories, V_trajectories, relax_trajectories = zip(*results)

        # Process trajectories for plotting
        angles = [np.unwrap(np.arctan2(traj[:, 0], traj[:, 1])) for traj in trajectories]
        angular_velocities = [traj[:, 2] for traj in trajectories]

        # Plot trajectories and values over time
        plot_trajectories(trajectories, angles, angular_velocities, f'{controller_type} Controller: Inverted Pendulum Trajectories', 'Angle', 'Angular Velocity')
        plot_values_over_time(V_trajectories, f'{controller_type} Controller: Inverted Pendulum Lyapunov Function', 'V Value')

        # Optional: Plot relaxation values if needed
        # plot_values_over_time(relax_trajectories, f'{controller_type} Controller: Inverted Pendulum Relaxation', 'Relax Value')

    plt.show()



if __name__ == "__main__":
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    n_input = 3
    n_hidden = 64
    n_output = 8
    num_of_layers = 3
    
    n_control_hidden = 32
    n_control = 1
    
    #---------------------- load joint CLF and policy -----------------------
    
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_clf.pt"
    baseline_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_controller.pt"
    #baseline_policy_model = "net_policy_trained_to_mimic_ppo.pth"
    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    
    
    baseline_controller = InvertedPendulum_Joint_Controller(net_nominal, net_policy, relaxation_penalty=2.0, m=1.1, l=1.0, b=0.18)
    
    #evaluate_clf_controller(baseline_controller, model_type = "Baseline")

    
    
    # load dro controller 
    dro_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_clf.pt"
    dro_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_controller.pt"

    
    
    

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(dro_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(dro_policy_model))
    
    
    dro_controller = InvertedPendulum_Joint_Controller(net_nominal, net_policy, relaxation_penalty=2.0, m=1.1, l=1.0, b=0.18)
    
    #evaluate_clf_controller(dro_controller, model_type = "DR")
    

    simulate_and_plot_trajectories(baseline_controller, dro_controller, simulate_joint_trajectory, plot_trajectories, plot_values_over_time)
