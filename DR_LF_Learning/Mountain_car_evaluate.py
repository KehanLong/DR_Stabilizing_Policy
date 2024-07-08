#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math


from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet
from Mountain_car_controller import MountainCar_Joint_Controller



from itertools import product
import sys
sys.path.append('../')  

from Gymnasium_modified.gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv



import time


def plot_values_over_time(data_list, filename, ylabel):
    plt.figure(figsize=(10, 8))
    
    for data in data_list:
        plt.plot(data, lw=2, alpha=0.6)

    plt.xlabel('Time Steps', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    #plt.grid(True)
    
    plt.legend(fontsize=22)
    plt.savefig(filename, dpi=300)
    plt.tight_layout()

def plot_trajectories(x_data, y_data, title, xlabel, ylabel, xlim = None):
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

    # Set the x-axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.grid(True)

    
    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize = 24)

    plt.tight_layout()

    # Save the figure in high resolution
    plt.savefig(title + ".png", dpi=300)

    return plt.gca().get_xlim()

# Function to run and visualize the mountain car simulation
def simulate_mountaincar(env, controller, steps=3000):
    observation, info = env.reset()
    
    trajectory = [observation]  # Record the initial state
    actions = []
    Lyapunov_values = []

    for i in range(steps):
        position = observation[0]
        velocity = observation[1]
        converted_state = np.array([np.sin(position), np.cos(position), velocity])
        converted_state = torch.tensor(converted_state, dtype=torch.float32)
        converted_state.requires_grad = True

        action_tensor = controller.compute_policy(converted_state)
        action = action_tensor.detach().cpu().numpy().reshape(1)
        actions.append(action[0])  # Record the action
        #print('action:', action)
        
        V_value, _ = controller.compute_clf(converted_state)
        V_value = V_value.detach().cpu().numpy().reshape(1)
        Lyapunov_values.append(V_value[0])

        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append(observation)  # Record the state
        env.render()

    env.close()
    return np.array(trajectory), actions, Lyapunov_values


def simulate_and_plot_trajectories(baseline_controller, dro_controller, simulate_mountaincar, plot_trajectories):
    """
    Simulates and plots trajectories for given controllers.

    Parameters:
    - baseline_controller: The baseline controller object.
    - dro_controller: The distributionally robust controller object.
    - simulate_mountaincar: Function to simulate mountain car trajectory given a controller and initial state.
    - plot_trajectories: Function to plot trajectories.
    - plot_actions: Function to plot actions over time.
    """
    
    # Specify initial states
    # initial_states = [
    #     np.array([-0.5, 0.0]),   
    #     np.array([-0.8, 0.05]),    
    #     np.array([-0.2, -0.04]),   
    #     np.array([0.3, 0.04])   
    # ]

    num_samples = 10
    positions = np.random.uniform(-0.8, 0.4, size=num_samples)
    velocities = np.random.uniform(-0.07, 0.07, size=num_samples)
    initial_states = [np.array([pos, vel]) for pos, vel in zip(positions, velocities)]


    for controller_type, controller in [('Baseline', baseline_controller), ('DR', dro_controller)]:
        print(f"Simulating for {controller_type} controller")
        
        trajectories = []
        actions_list = []
        V_values_list = []

        for init_state in initial_states:
            # use 'human' render mode to see the graphics, use 'rgb_array' to save time
            env = Continuous_MountainCarEnv(render_mode="rgb_array", power=0.0012, controller=controller_type, initial_state=init_state)
            trajectory, actions, V_values = simulate_mountaincar(env, controller)
            trajectories.append(trajectory)
            actions_list.append(actions)
            V_values_list.append(V_values)

        # Process trajectories for plotting
        positions = [traj[:, 0] for traj in trajectories]
        velocities = [traj[:, 1] for traj in trajectories]

        if controller_type == 'Baseline':
            xlim = plot_trajectories(positions, velocities, f'{controller_type} Controller: Mountain Car Trajectories', 'Position', 'Velocity')
        else:
            #plot_trajectories(positions, velocities, f'{controller_type} Controller: Mountain Car Trajectories', 'Position', 'Velocity', xlim=xlim)
            plot_trajectories(positions, velocities, f'{controller_type} Controller: Mountain Car Trajectories', 'Position', 'Velocity')
            plot_values_over_time(V_values_list, f'{controller_type} Controller: Mountain Car LF', ylabel = "Lyapunov Value")

    plt.show()



def main():
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/mountain_car/baseline_clf.pt"
    baseline_policy_model = "saved_models/joint_clf_controller_models/mountain_car/baseline_controller.pt"
    dro_clf_saved_model = "saved_models/joint_clf_controller_models/mountain_car/dro_clf_test.pt"
    dro_policy_model = "saved_models/joint_clf_controller_models/mountain_car/dro_controller_test.pt"
    
    n_input = 3
    n_hidden = 16
    n_output = 4
    num_of_layers = 3
    n_control_hidden = 16
    n_control = 1

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    baseline_clf_controller = MountainCar_Joint_Controller(net_nominal, net_policy, control_bound=2.0)
    
    #env = Continuous_MountainCarEnv(render_mode="rgb_array", power=0.0012, controller='Baseline', initial_state=np.array([-math.pi/6, 0]))
    #baseline_trajectory, actions1, V_values1 = simulate_mountaincar(env, baseline_clf_controller)
    
    net_dro = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    net_dro.load_state_dict(torch.load(dro_clf_saved_model))
    net_policy_dro = PolicyNet(n_input, n_control_hidden, n_control, 3)
    net_policy_dro.load_state_dict(torch.load(dro_policy_model))
    dro_clf_controller = MountainCar_Joint_Controller(net_dro, net_policy_dro, control_bound=2.0)
    
    #env = Continuous_MountainCarEnv(render_mode="rgb_array", power=0.0012, controller='DR', initial_state= np.array([-math.pi/6, 0]))
    #dro_trajectory, actions2, V_values2 = simulate_mountaincar(env, dro_clf_controller)
    
    # plot_actions(actions1, actions2, "Comparison of Control Inputs Over Time", "mountain_car_controller_comparison.png")
    # plot_actions(V_values1, V_values2, "Comparison of Lyapunov Function Values Over Time", "mountain_car_compare_V_value.png", "Lyapunov value")
    
    simulate_and_plot_trajectories(baseline_clf_controller, dro_clf_controller, simulate_mountaincar, plot_trajectories)

if __name__ == "__main__":
    main()
    
    
    