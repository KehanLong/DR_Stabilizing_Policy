#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO


from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet

from Inverted_pendulum_controller import InvertedPendulum_Joint_Controller

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def compute_sac_ppo_policy(model, theta, omega):

    obs = np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)
    obs = obs.reshape(1, -1)  # Reshape to ensure it's a 2D array
    action, _ = model.predict(obs, deterministic=True)
    return action

def plot_control_policies(models, titles, lyapunov_indices, file_prefix='Inverted_Pendulum'):
    """
    Plots the control policies of given models.

    Args:
    models: A list of models/controllers.
    titles: A list of titles for the subplots.
    lyapunov_indices: A list of indices indicating which models are Lyapunov based.
    file_prefix: Prefix for saving the figure.
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = np.linspace(-np.pi, np.pi, 80)
    omega = np.linspace(-10, 10, 80)
    Theta, Omega = np.meshgrid(theta, omega)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()  # Flatten to easily index with one number

    for idx, model in enumerate(models):
        U_values = np.zeros_like(Theta)

        for i in range(Theta.shape[0]):
            for j in range(Theta.shape[1]):
                th = Theta[i, j]
                om = Omega[i, j]

                if idx in lyapunov_indices:
                    # Convert theta to cos(theta) and sin(theta)
                    sin_th = np.sin(th)
                    cos_th = np.cos(th)
                    X_tensor = torch.Tensor([sin_th, cos_th, om])
                    u = model.compute_policy(X_tensor).detach().cpu().numpy()
                else:
                    # SAC and PPO models
                    u = compute_sac_ppo_policy(model, th, om)

                U_values[i, j] = u

        contour = axs[idx].contourf(Theta, Omega, U_values, levels=20)
        fig.colorbar(contour, ax=axs[idx], label='Control input u')
        axs[idx].set_xlabel('Theta')
        axs[idx].set_ylabel('Omega')
        axs[idx].set_title(titles[idx])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{file_prefix}_controllers.png", dpi=300)
    plt.show()





if __name__ == "__main__":
    

    n_input = 3
    n_hidden = 64
    n_output = 8
    num_of_layers = 3
    
    n_control_hidden = 32
    n_control = 1
    
    #---------------------- load CLF and policy -----------------------
    
    baseline_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_asympto_joint_clf.pt"
    baseline_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/baseline_asympto_controller.pt"
    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(baseline_policy_model))
    
    
    baseline_controller = InvertedPendulum_Joint_Controller(net_nominal, net_policy, relaxation_penalty=2.0, m=1.1, l=1.0, b=0.18)
    
    
    # load dro controller 
    dro_clf_saved_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_joint_clf.pt"
    dro_policy_model = "saved_models/joint_clf_controller_models/inverted_pendulum/dro_controller.pt"
    
    
    

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    
    net_nominal.load_state_dict(torch.load(dro_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    
    net_policy.load_state_dict(torch.load(dro_policy_model))
    
    
    dro_controller = InvertedPendulum_Joint_Controller(net_nominal, net_policy, relaxation_penalty=2.0, m=1.1, l=1.0, b=0.18)    
    
    
    model_SAC = SAC.load("sac_pendulum")
    model_PPO = PPO.load("ppo_pendulum")
    
    
    plot_control_policies([baseline_controller, dro_controller, model_SAC, model_PPO], ["Baseline Lyapunov", "DRO Lyapunov", "SAC", "PPO"], [0, 1])