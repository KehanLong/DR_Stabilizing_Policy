#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet
from Inverted_pendulum_controller import InvertedPendulum_Joint_Controller


from itertools import product
import sys
sys.path.append('../')  

from Gymnasium_modified.gymnasium.envs.classic_control.pendulum import PendulumEnv

from stable_baselines3 import PPO, SAC



import time
import csv

def read_rl_values(filename):
    sac_values = []
    ppo_values = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            # Remove brackets and convert to float
            sac_value = float(row[0].strip('[]'))
            ppo_value = float(row[1].strip('[]'))
            sac_values.append(sac_value)
            ppo_values.append(ppo_value)

    return sac_values, ppo_values

def plot_actions(actions1, actions2, title, filename, y_axis_name = "Control Input"):
    plt.figure(figsize=(10, 6))
    
    resample_factor = 4
    
    actions1= np.array(actions1)
    actions2 = np.array(actions2)
    
    actions1 = np.mean(actions1.reshape(-1, resample_factor), axis=1)
    actions2 = np.mean(actions2.reshape(-1, resample_factor), axis=1)
    
    # Use different line styles for the two plots
    plt.plot(actions1, label='Baseline Controller', linestyle='-', linewidth=4)
    plt.plot(actions2, label='DR Controller', linestyle='--', linewidth=4)

    # Increase font sizes
    #plt.title(title, fontsize=24)
    plt.xlabel('Time Step', fontsize=22)
    plt.ylabel(y_axis_name, fontsize=22)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Enlarge the legend
    plt.legend(fontsize=22)

    plt.tight_layout()

    #plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.show()
    
def plot_multiple_actions(list_of_actions, labels, title, filename, y_axis_name="Lyapunov Value"):
    plt.figure(figsize=(10, 6))

    # Define different line styles for distinction
    line_styles = ['-', '--', '-.', ':']
    linewidth = 3

    # Check if there are more line styles than actions, if not, repeat them
    if len(list_of_actions) > len(line_styles):
        line_styles = line_styles * (len(list_of_actions) // len(line_styles) + 1)

    # Plot each set of actions
    for i, actions in enumerate(list_of_actions):
        plt.plot(actions, label=labels[i], linestyle=line_styles[i], linewidth=linewidth)

    # Increase font sizes for readability
    plt.title(title, fontsize=24)
    plt.xlabel('Time Step', fontsize=20)
    plt.ylabel(y_axis_name, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)

    plt.savefig(filename, dpi=300)
    plt.show()
    
def plot_value_functions(lyapunov_values, ppo_values, sac_values, title, filename):
    plt.figure(figsize=(10, 6))
    
    resample_factor = 5
    
    lyapunov_values = np.array(lyapunov_values)
    ppo_values = np.array(ppo_values)
    sac_values = np.array(sac_values)
    
    lyapunov_values = np.mean(lyapunov_values.reshape(-1, resample_factor), axis=1)
    ppo_values = np.mean(ppo_values.reshape(-1, resample_factor), axis=1)
    sac_values = np.mean(sac_values.reshape(-1, resample_factor), axis=1)
    
    
    plt.plot(lyapunov_values, label='DR Lyapunov Values', linestyle='-', linewidth=3)
    plt.plot(-ppo_values, label='PPO Value Function', linestyle='--', linewidth=3)
    plt.plot(-sac_values, label='SAC Value Function', linestyle='-.', linewidth=3)
    plt.title(title, fontsize=24)
    plt.xlabel('Time Step', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(fontsize = 18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(filename)
    plt.show()
    
    
# def evaluate_rl_value_functions_on_dr_trajectory(model, observations):
#     value_function = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for obs in observations:
#         if isinstance(model, SAC):
#             # For SAC, get the minimum Q-value as the value function
            
#             action, _ = model.predict(obs, deterministic=True)
#             #obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
#             q_input = torch.cat([torch.tensor(obs).to(device), torch.tensor(action, dtype=torch.float32).to(device)], dim=0).unsqueeze(0)
#             q_values = [q_net(q_input).detach().cpu().numpy()[0][0] for q_net in model.critic.q_networks]
#             value = np.min(q_values)
#         elif isinstance(model, PPO):
#             # For PPO, directly evaluate the value function
#             obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
#             value = model.policy.predict_values(obs_tensor).cpu().detach().numpy()[0]
#         value_function.append(value)

#     return np.array(value_function)

    


# Function to run and visualize the pendulum simulation
def simulate_pendulum(env, controller, steps=1200, filename='dr_trajectory.npy'):
    observation, info = env.reset()
    
    
    actions = []
    Lyapunov_values = []

    observations = []
    for i in range(steps):
        cos_angle = observation[0]
        sin_angle = observation[1]
        #print('observation:', observation)
        
        converted_state = np.array([sin_angle, cos_angle, observation[2]])
        converted_state = torch.tensor(converted_state, dtype=torch.float32)
        converted_state.requires_grad = True

        action_tensor = controller.compute_policy(converted_state)
        action = action_tensor.detach().cpu().numpy().reshape(1)
        actions.append(action[0])  # Record the action
        #print('action:', action)
        
        V_value, _ = controller.compute_clf(converted_state)
        V_value = V_value.detach().cpu().numpy().reshape(1)
        Lyapunov_values.append(V_value[0])
        observations.append(observation)
        

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()
    
    # trajectory_data = {
    #     'observations': np.array(observations)
    # }
    # np.save(filename, trajectory_data)    
    
    return actions, Lyapunov_values




if __name__ == "__main__":
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_clf.pt"
    
    baseline_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_controller.pt"
    
    
    
    dro_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_clf_test1.pt"
    dro_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_controller_test1.pt"
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_input = 3
    n_hidden = 64
    n_output = 8
    num_of_layers = 3
    
    n_control_hidden = 32
    
    n_control = 1

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    
    baseline_clf_controller = InvertedPendulum_Joint_Controller(net_nominal, net_policy)
    
    
    # Create the Pendulum environment
    env = PendulumEnv(render_mode="human" , g=9.81, m=1.1, l=1.0, b=0.18, controller = 'Baseline', initial_state=np.array([-np.pi/2, 3.5]))

    # Simulate with the baseline controller
    actions1, V_values1 = simulate_pendulum(env, baseline_clf_controller)

    # Simulate with the dro controller
    net_dro = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_dro.load_state_dict(torch.load(dro_clf_saved_model))
    
    net_policy_dro = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy_dro.load_state_dict(torch.load(dro_policy_model))
    
    dro_clf_controller = InvertedPendulum_Joint_Controller(net_dro, net_policy_dro)    
    
    
    env = PendulumEnv(render_mode="human" , g=9.81, m=1.1, l=1.0, b=0.18, controller = 'DR', initial_state=np.array([-np.pi/2, 3.5]))
    actions2, V_values2 = simulate_pendulum(env, dro_clf_controller)

    # Plot and compare actions

    #plot_actions(actions1, actions2, "Control Actions Over Time", "controller_comparison.png")
    
    #plot_actions(V_values1, V_values2, "Lyapunov Function Values Over Time", "controller_compare_V_value.png", "Lyapunov value")
    