#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import torch


import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# Custom Pendulum Environment


import sys
sys.path.append('../')  
from Gymnasium_modified.gymnasium.envs.classic_control.pendulum import PendulumEnv
from Gymnasium_modified.gymnasium.envs.classic_control.cartpole import CartPoleEnv

import csv

def evaluate_rl_controller(model, theta_bins=100, omega_bins=100):
    # Create a grid of theta (angle) and omega (angular velocity) values
    theta = np.linspace(-np.pi * 2, np.pi * 2, theta_bins)
    omega = np.linspace(-8, 8, omega_bins)
    Theta, Omega = np.meshgrid(theta, omega)

    sin_theta = np.sin(Theta.ravel())
    cos_theta = np.cos(Theta.ravel())
    Omega_flat = Omega.ravel()

    V_values = np.zeros_like(Theta.ravel())
    U_values = np.zeros_like(Theta.ravel())

    for i in range(len(Theta.ravel())):
        obs = np.array([cos_theta[i], sin_theta[i], Omega_flat[i]])
        action, _states = model.predict(obs, deterministic=True)
        
        if isinstance(model, SAC):
            obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32).to(model.device)
            action_tensor = torch.tensor(action[None, :], dtype=torch.float32).to(model.device)
            with torch.no_grad():
                q_values = model.critic(obs_tensor, action_tensor)
                V_value = min(q_values).cpu().numpy()
        elif isinstance(model, PPO):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
            with torch.no_grad():
                V_value = model.policy.predict_values(obs_tensor).cpu().numpy()

        V_values[i] = V_value.item()
        U_values[i] = action.item()

    V_values = V_values.reshape(Theta.shape)
    U_values = U_values.reshape(Theta.shape)

    model_name = "SAC" if isinstance(model, SAC) else "PPO"

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    contour = plt.contourf(Theta, Omega, V_values, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Theta (radians)', fontsize=16)
    plt.ylabel('Omega (rad/s)', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'{model_name} Value Function', fontsize=18)

    plt.subplot(1, 2, 2)
    contour = plt.contourf(Theta, Omega, U_values, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Theta (radians)', fontsize=16)
    plt.ylabel('Omega (rad/s)', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f'{model_name} Control Policy', fontsize=18)
    
    plt.suptitle(f"{model_name} Evaluation", fontsize=18, y=1.05)
    

    plt.tight_layout()
    plt.savefig(f"{model_name}_Evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_actions(actions1, actions2, title, filename, y_axis_name = "Control Input"):
    plt.figure(figsize=(10, 6))
    
    # Use different line styles for the two plots
    plt.plot(actions1, label='SAC Controller', linestyle='-', color='r', linewidth=4)
    plt.plot(actions2, label='PPO Controller', linestyle='--', color='purple', linewidth=4)

    # Increase font sizes
    #plt.title(title, fontsize=24)
    plt.xlabel('Time Step', fontsize=22)
    plt.ylabel(y_axis_name, fontsize=22)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Enlarge the legend
    plt.legend(fontsize=22)

    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
    
def train_model(env, time_steps=100000):
    """
    Train the PPO model.

    Args:
    env: The training environment.
    time_steps: Number of training steps.

    Returns:
    The trained model.
    """
    
    # SAC Training
    # policy_kwargs = {
    #     "net_arch": {
    #         "pi": [32, 32],  # Actor network architecture
    #         "qf": [64, 64]   # Critic network architecture
    #     }
    # }
    # model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    
    
    
    # PPO Training
    # policy_kwargs = {
    #     "net_arch": [64, 64],  # Shared layers between policy and value function
    # }   
    # model = PPO("MlpPolicy", env, verbose = 1, policy_kwargs=policy_kwargs)
    
    model = PPO("MlpPolicy", env, verbose = 1)
    
    
    model.learn(total_timesteps=time_steps)
    
    
    return model

    
    
def test_model(model, env, num_steps=200):
    """
    Test the trained model and collect control actions and value function data.

    Args:
    model: The trained model.
    env: The testing environment.
    num_steps: Number of steps to test the model.

    Returns:
    A tuple of lists containing actions and value function data.
    """
    obs, _ = env.reset()
    actions = []
    value_function = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        actions.append(action)
        
        

        # Evaluate the value function
        if isinstance(model, SAC):
            # Assuming model.critic.q_networks[0] can be used for SAC value function estimation
            q_input = torch.cat([torch.tensor(obs).to(device), torch.tensor(action, dtype=torch.float32).to(device)], dim=0).unsqueeze(0)
            q_values = [q_net(q_input).detach().cpu().numpy()[0][0] for q_net in model.critic.q_networks]
            value = np.min(q_values)
        elif isinstance(model, PPO):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            value_hidden = model.policy.mlp_extractor.value_net(obs_tensor)
            value = model.policy.value_net(value_hidden).squeeze(0).detach().cpu().numpy()
            

        value_function.append(-value)

        if done:
            obs, _ = env.reset()

    return actions, value_function

def main():
    # Training
    
    # for inverted_pendulum 
    #train_env = make_vec_env('Pendulum-v1', n_envs = 1)
    train_env = make_vec_env(lambda: PendulumEnv(), n_envs=1)
    
    # for cart-pole
    # train_env = make_vec_env('CartPole-v1', n_envs=1)
    # train_env = make_vec_env(lambda: CartPoleEnv(), n_envs=1)
    
    #model = train_model(train_env)
    #model.save("saved_models/RL_models/ppo_pendulum_new")

    # train_env.close()

    # Testing/Visualization
    test_env = PendulumEnv(render_mode = 'human', g=9.81, m=1.1, l=1.0, b=0.18)
    
    
    model_sac = SAC.load("saved_models/RL_models/sac_pendulum")
    #evaluate_rl_controller(model_sac)

    # Example usage for PPO
    model_ppo = PPO.load("saved_models/RL_models/ppo_pendulum") 
    #evaluate_rl_controller(model_ppo)

    sac_actions, sac_values = test_model(model_sac, test_env)
    
    # model_ppo = PPO.load("ppo_pendulum")
    ppo_actions, ppo_values = test_model(model_ppo, test_env)
    
    plot_actions(sac_actions, ppo_actions, "Control Actions Over Time", "control_actions_RL.png")
    plot_actions(sac_values, ppo_values, "Value Function Over Time", "value_function_RL.png", y_axis_name="Value")
    
    # with open('rl_values_cartpole.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['sac_values', 'ppo_values'])
    #     for sac_val, ppo_val in zip(sac_values, ppo_values):
    #         writer.writerow([sac_val, ppo_val])
    

if __name__ == "__main__":
    main()