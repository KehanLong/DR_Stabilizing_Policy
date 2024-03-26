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

from Mountain_car_controller import MountainCar_Joint_Controller




def simulate_joint_trajectory(controller, initial_state, time_steps=600, dt=0.02):
    trajectory = [initial_state.detach().numpy().reshape(1,-1)]  # This ensures it starts as a 2D array
    state = initial_state.clone().detach().float()  # Starting from a detached copy
    state.requires_grad = True
    
    V_values = []
    relax_values = []
    
    control_values = []

    for _ in range(time_steps):

        V, gradV = controller.compute_clf(state)

    
        # add perturbations 
        f_x, g_x = controller.mountain_car_dynamics(state, power = 0.0012)
        
        LfV, LgV = controller.compute_lie_derivatives(state, f_x, g_x)
        
        
        loss = controller.lyapunov_derivative_loss(state, torch.zeros((2,2)))
        
        #loss = 0
        
        u_opt = controller.compute_policy(state)
        
        
        
        
        state_dot = f_x + g_x * u_opt
        print('u_opt:', u_opt.cpu().detach().squeeze())
        print('state_dot:', state_dot[0,0,2])
        state = (state + dt * state_dot.view_as(state)).detach()  # Getting a detached tensor after update
        state.requires_grad = True
        
        trajectory.append(state.detach().numpy().reshape(1,-1))  # Convert to 2D numpy array immediately
        
        
        V_values.append(V.cpu().detach().squeeze())
        
        #print(V_values)
        relax_values.append(loss.cpu().detach())
        control_values.append(u_opt.cpu().detach().squeeze())
        
        #relax_values.append(0)

    return np.vstack(trajectory), V_values, control_values # Convert the list of 2D numpy arrays to a single 2D numpy array


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

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=24)
    
    plt.grid(True)

    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize = 16)

    plt.tight_layout()

    # Save the figure in high resolution
    #plt.savefig(title + ".png", dpi=300)

def plot_values_over_time(data_list, title, ylabel):
    plt.figure(figsize=(10, 8))
    
    for data in data_list:
        plt.plot(data, lw=2, alpha=0.6)

    plt.xlabel('Time Steps', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=24)
    #plt.grid(True)
    
    plt.legend(fontsize=18)

    plt.tight_layout()

    # Save the figure in high resolution
    #plt.savefig(title + ".png", dpi=300)



if __name__ == "__main__":
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    n_input = 3
    n_hidden = 16
    n_output = 4
    num_of_layers = 3
    
    n_control_hidden = 16
    
    n_control = 1
    
    #---------------------- load joint CLF and policy -----------------------
    
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_joint_clf_1.pt"
    
    baseline_policy_model = "saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_controller_1.pt"
    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    
    
    baseline_controller = MountainCar_Joint_Controller(net_nominal, net_policy, relaxation_penalty=0.2, power = 0.0012, control_bound=1.5)
    
    
    # load dro controller 
    dro_clf_saved_model = "saved_models/joint_clf_controller_models/mountain_car/dro_joint_clf_1.pt"
    dro_policy_model = "saved_models/joint_clf_controller_models/mountain_car/dro_controller_1.pt"
    
    
    

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(dro_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(dro_policy_model))
    
    
    dro_controller = MountainCar_Joint_Controller(net_nominal, net_policy, relaxation_penalty=0.2, power = 0.0012, control_bound=2.0)
    

  
    # Sample random angles in the negative range and 5 in the positive range avoiding values too close to zero
    negative_angles = np.random.uniform(low=-0.7, high=-0.5, size=3)
    positive_angles = np.random.uniform(low=-0.5, high=-0.3, size=3)
    angles = np.concatenate((negative_angles, positive_angles), axis=0)
    np.random.shuffle(angles)  # Shuffle to ensure randomness
    
    # Sample random positions in the negative range and 5 in the positive range avoiding values too close to zero
    negative_velocity = np.random.uniform(low=-0.0, high=0.0, size=3)
    positive_velocity = np.random.uniform(low=0.0, high=0.0, size=3)
    velocity = np.concatenate((negative_velocity, positive_velocity), axis=0)
    np.random.shuffle(velocity)  # Shuffle to ensure randomness
    
    # Create N random initial states
    initial_states = [torch.tensor([np.sin(angles[i]), np.cos(angles[i]), velocity[i]]) for i in range(6)]
    
    #initial_states = [torch.tensor([np.sin(-np.pi/6), np.cos(-np.pi/6), 0])]
    

    # Simulate trajectories for both controllers
    for controller_type in ['Baseline', 'DR']:
        print(f"Simulating for {controller_type} controller")
        if controller_type == 'Baseline':
            controller = baseline_controller  # Assuming baseline_controller is already initialized
        else:
            controller = dro_controller  # Assuming dro_controller is already initialized

        results = [simulate_joint_trajectory(controller, init_state) for init_state in initial_states]
        trajectories, V_trajectories, control_trajectories = zip(*results)

        # Plotting
        # angles = [np.arctan2(traj[:, 0], traj[:, 1]) for traj in trajectories]
        # angular_velocities = [traj[:, 2] for traj in trajectories]
        
        # Unwrap angles
        angles = [np.unwrap(np.arctan2(traj[:, 0], traj[:, 1])) for traj in trajectories]
        angular_velocities = [traj[:, 2] for traj in trajectories]

        # Function call with correct arguments
        plot_trajectories(trajectories, angles, angular_velocities, f'{controller_type} Controller: Mountain Car Trajectories', 'Position', 'Velocity')
        plot_values_over_time(V_trajectories, f'{controller_type} Controller: Mountain Car Lyapunov Function', 'V Value')
        plot_values_over_time(control_trajectories, f'{controller_type} Controller: Mountain Car Control', 'Control Value')


    plt.show()