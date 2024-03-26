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
from Just_pole_controller import Just_Pole_Joint_Controller


import sys
sys.path.append('../')  

from Gymnasium.gymnasium.envs.classic_control.cartpole import CartPoleEnv



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
def simulate_cartpole(env, controller, steps=1200):
    observation, info = env.reset()
    actions = []
    Lyapunov_values = []
    
    
    for _ in range(steps):
        #print(angle)
        angle = observation[2]
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        
        #print('state:', observation)
        
        converted_state = np.array([cos_angle, sin_angle, observation[3]])
        
        converted_state = torch.tensor(converted_state, dtype=torch.float32)
        converted_state.requires_grad = True
        
        # Compute the policy
        action_tensor = controller.compute_policy(converted_state)
        action = action_tensor.detach().cpu().numpy().reshape(1)
        #action = [0.0]

        actions.append(action[0])  # Record the action
        
        V_value, _ = controller.compute_clf(converted_state)
        V_value = V_value.detach().cpu().numpy().reshape(1)
        Lyapunov_values.append(V_value[0])

        observation, reward, terminated, truncated, info = env.step(action)
        
        '''
        following is for testing the dynamics equations are implemented correctly in Gym and controller class
        '''
        
        # angle_tmp = observation[2]
        # print('gym_observation:', np.array([observation[0], np.cos(angle_tmp), np.sin(angle_tmp), observation[1], observation[3]]))
        
        
        # f_x, g_x = controller.cart_pole_dynamics(converted_state)
        
        # x_dot = f_x + g_x * action[0]
        # tau = 0.02
        
        # controller_state = converted_state + x_dot * tau
        
        # print('controller state:', controller_state.detach().cpu().numpy())
        
        

        
        
        env.render()

    env.close()
    
    return actions, Lyapunov_values




if __name__ == "__main__":
    
    
    
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/just_pole/baseline_joint_clf_1.pt"
    baseline_policy_model = "saved_models/joint_clf_controller_models/just_pole/baseline_controller_1.pt"
    
    
    
    
    n_input = 3
    n_hidden = 32
    n_output = 4
    num_of_layers = 4
    n_control_hidden = 32
    n_control = 1
    n_control_layer = 3

    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, n_control_layer)
    
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    
    baseline_clf_controller = Just_Pole_Joint_Controller(net_nominal, net_policy)
    
    # Create the Pendulum environment
    env = CartPoleEnv(render_mode="human" , mc = 1.0, mp = 1.0, l = 1.0)
    
    # Simulate with the baseline controller
    actions1, V_values1 = simulate_cartpole(env, baseline_clf_controller)
    
    
    
    # net_dro = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    # net_dro.load_state_dict(torch.load(dro_clf_saved_model))
    
    # net_policy_dro = PolicyNet(n_input, n_control_hidden, n_control, n_control_layer)
    
    # net_policy_dro.load_state_dict(torch.load(dro_policy_model))
    
    #dro_clf_controller = Cart_Pole_Joint_Controller(net_dro, net_policy_dro)    
    
    dro_clf_controller = Cart_Pole_Joint_Controller(net_nominal, net_policy)
    
    
    env = CartPoleEnv(render_mode="human" , mc = 1.1, mp = 0.1, l = 0.6)
    
    # Simulate with the baseline controller
    actions2, V_values2 = simulate_cartpole(env, dro_clf_controller)
    
    
    plot_actions(actions1, actions2, "Comparison of Control Inputs Over Time", "cart_pole_controller_comparison.png")
    
    plot_actions(V_values1, V_values2, "Comparison of Lyapunov Function Values Over Time", "cart_pole_controller_compare_V_value.png", "Lyapunov value")