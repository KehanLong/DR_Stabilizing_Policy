#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 00:06:40 2023

@author: kehan
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet
from Mountain_car_controller import MountainCar_Joint_Controller


from itertools import product
import sys
sys.path.append('../')  

from Gymnasium.gymnasium.envs.classic_control.acrobot import AcrobotEnv



import time

def plot_actions(actions1, actions2, title, filename, y_axis_name = "Action Value"):
    plt.figure(figsize=(10, 6))
    plt.plot(actions1, label='Baseline Controller')
    plt.plot(actions2, label='DRO Controller')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel(y_axis_name)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()

# Function to run and visualize the pendulum simulation
def simulate_pendulum(env, controller, steps=800):
    observation, info = env.reset()
    
    
    actions = []
    Lyapunov_values = []

    for i in range(steps):
        position = observation[0]
        velocity = observation[1]
        converted_state = np.array([position, velocity])
        converted_state = torch.tensor(converted_state, dtype=torch.float32)
        converted_state.requires_grad = True

        action_tensor = controller.compute_policy(converted_state)
        action = action_tensor.detach().cpu().numpy().reshape(1)
        actions.append(action[0])  # Record the action
        print('action:', action)
        
        V_value, _ = controller.compute_clf(converted_state)
        V_value = V_value.detach().cpu().numpy().reshape(1)
        Lyapunov_values.append(V_value[0])

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

    env.close()
    return actions, Lyapunov_values




if __name__ == "__main__":
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_joint_clf.pt"
    
    baseline_policy_model = "saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_controller.pt"
    
    
    # dro_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_joint_clf.pt"
    # dro_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_controller.pt"
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_input = 2
    n_hidden = 32
    n_output = 8
    num_of_layers = 3
    
    n_control_hidden = 32
    
    n_control = 1

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    
    baseline_clf_controller = MountainCar_Joint_Controller(net_nominal, net_policy)
    
    # Create the Pendulum environment
    env = Continuous_MountainCarEnv(render_mode="human")

    # Simulate with the baseline controller
    actions1, V_values1 = simulate_pendulum(env, baseline_clf_controller)