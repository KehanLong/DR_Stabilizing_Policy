#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from lyapunov_net import LyapunovNet
from PolicyNet import PolicyNet


from Cart_pole_controller import Cart_Pole_Joint_Controller


from itertools import product

from torch.utils.data import DataLoader, TensorDataset

# def save_checkpoint(epoch, model, optimizer, filename="checkpoint.pth.tar"):
#     """Save a checkpoint"""
#     state = {
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     torch.save(state, filename)



def generate_uncertainty_samples(num_samples=4, alpha=2.0, beta=5.0, low=-0.05, high=0.2):
    """
    Generate samples of xi based on the beta distribution for length uncertainties.
    Args:
    - num_samples: Number of samples to generate.
    - alpha: Alpha parameter of the beta distribution.
    - beta: Beta parameter of the beta distribution.
    - low: Lower bound of the desired range.
    - high: Upper bound of the desired range.
    Returns:
    - A tensor of shape (num_samples, 1) containing samples of xi.
    """
    beta_dist = torch.distributions.Beta(alpha, beta)
    xi_samples = beta_dist.sample((num_samples,))
    xi_samples = xi_samples * (high - low) + low
    return xi_samples

def train_clf_nn_controller():
    n_input = 5
    n_hidden = 128
    n_output = 4
    num_of_layers = 5
    
    n_control_hidden = 64
    
    n_control = 1

    num_epochs = 10

    loss_threshold = 1e-4
    batch_size = 256

    learning_rate = 0.001
    
    xi_samples_num = 10
    
    xi_samples = generate_uncertainty_samples(xi_samples_num)
    
    print('xi_samples:', xi_samples)
    
    nominal_length = 1.0
    
    # Compute the average perturbations for mass and damping
    average_perturbation = torch.mean(xi_samples)

    length = nominal_length + average_perturbation
    
    print('average length:', length)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate training samples
    # For theta values
    num_theta = 32
    theta = torch.Tensor(num_theta).uniform_(-np.pi/6, np.pi/6)
    # For angular velocities
    num_angular = 32
    theta_dot = torch.Tensor(num_angular).uniform_(-1.2, 1.2)  # adjust this range if needed
    # For cart positions
    num_pos = 32
    pos = torch.Tensor(num_pos).uniform_(-1.2, 1.2)
    # For cart linear velocities
    num_linear = 32
    pos_dot = torch.Tensor(num_linear).uniform_(-1.2, 1.2)  # adjust this range if needed
    
    combinations = list(product(pos, theta, pos_dot, theta_dot))
    
    num_samples = num_theta * num_angular * num_pos * num_linear

    x_train = torch.zeros([num_samples, n_input], requires_grad=True)
    # Convert the combinations list to tensor
    x_original = torch.tensor(combinations, dtype=torch.float32)
    x_original.requires_grad = True
    
    x_train_temp = torch.zeros_like(x_train)
    
    x_train_temp[:, 0] = x_original[:, 0]                     #pos
    x_train_temp[:, 1] = torch.cos(x_original[:, 1])          #cos angle  
    x_train_temp[:, 2] = torch.sin(x_original[:, 1])          #sin angle  
    x_train_temp[:, 3] = x_original[:, 2]                     #linear velocity
    x_train_temp[:, 4] = x_original[:, 3]                     #angluar velocity
    
    x_train.data = x_train_temp.data
    x_train = x_train.to(device)

    xi_samples = xi_samples.to(device)


    # Create a TensorDataset from the training data
    train_dataset = TensorDataset(x_train)

    # Create a DataLoader for mini-batch training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Dictionary to store the trained controllers
    trained_controllers = {}

    #for model_type in ['nominal', 'dro']:
    for model_type in ['nominal']:
        print(f"\nTraining {model_type} model")

        # Reset model and optimizer for each training type
        net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers).to(device)
        net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3).to(device)
        
        # Warm initialization for dro
        if model_type == 'dro':
            # Load the trained weights from the nominal model
            net_nominal.load_state_dict(torch.load("saved_models/joint_clf_controller_models/cart_pole/baseline_clf.pt"))
            net_policy.load_state_dict(torch.load("saved_models/joint_clf_controller_models/cart_pole/baseline_controller.pt"))

        clf_controller = Cart_Pole_Joint_Controller(net_nominal, net_policy, length=1.0, relaxation_penalty=2.0, control_bounds=10.0)
   
        optimizer = torch.optim.Adam(list(net_nominal.parameters()) + list(net_policy.parameters()), lr=learning_rate, betas=(0.9, 0.999))

        print('num of mini batch:', len(train_loader))
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_x in train_loader:
                
                batch_x = batch_x[0].to(device)  # Get the batch data and move it to the device

                optimizer.zero_grad()
                batch_loss = 0.0

                if model_type == 'nominal':
                    # Compute the loss for the current batch (baseline training)
                    delta_opt = clf_controller.lyapunov_derivative_loss(batch_x, xi_samples)
        
                else:  # 'dro'
                    # Compute the loss for the current batch (DRO training)
                    delta_opt = clf_controller.dro_lyapunov_derivative_loss_uniform(batch_x, xi_samples)

                batch_loss = delta_opt
                batch_loss.backward()
                optimizer.step()
            
                total_loss += batch_loss
            
            total_loss = total_loss / len(train_loader)

        # for epoch in range(num_epochs):
        #     optimizer.zero_grad()
        #     total_loss = 0.0

        #     if model_type == 'nominal':
        #         # Compute the loss for all samples (baseline training)
        #         delta_opt = clf_controller.lyapunov_derivative_loss(x_train, xi_samples)
        #     else:  # 'dro'
        #         # Compute the loss for all samples (DRO training)
                
        #         # point-wise
        #         delta_opt = clf_controller.dro_lyapunov_derivative_loss_(x_train, xi_samples)

        #         # uniform
        #         # delta_opt = clf_controller.dro_lyapunov_derivative_loss_uniform(x_train, xi_samples)

        #     total_loss = delta_opt
        #     total_loss.backward()
        #     optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], {model_type} Loss: {total_loss.item():.6f}')

            if total_loss.item() < loss_threshold:
                print(f'Training stopped at epoch {epoch + 1}, {model_type} Loss: {total_loss.item():.4f}')
                break
            
        trained_controllers[model_type] = clf_controller

        # Saving model parameters
        if model_type == 'nominal':
            torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/baseline_clf.pt")
            torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/baseline_controller.pt")
        else:  # 'dro'
            torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/dro_clf.pt")
            torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/dro_controller.pt")
            
    return trained_controllers
           

if __name__ == "__main__":
    trained_controllers = train_clf_nn_controller()
    # nominal_controller = trained_controllers['nominal']
    # dro_controller = trained_controllers['dro']
