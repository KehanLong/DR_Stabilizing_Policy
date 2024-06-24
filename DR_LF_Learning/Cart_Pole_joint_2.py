#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:22:03 2023

@author: kehan
"""


import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from lyapunov_net import LyapunovNet
from PolicyNet import PolicyNet


from Cart_Pole_joint_controller import Cart_Pole_Joint_Controller


from itertools import product

def save_checkpoint(epoch, model, optimizer, filename="checkpoint.pth.tar"):
    """Save a checkpoint"""
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


if __name__ == "__main__":
    n_input = 5
    n_hidden = 128
    n_output = 4
    num_of_layers = 5
    
    n_control_hidden = 64
    
    n_control = 1

    num_epochs = 200

    loss_threshold = 1e-4
    batch_size = 256

    learning_rate = 0.001
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Instantiate the network and the CLF_NN_Controller
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    net_nominal.to(device)
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    net_policy.to(device)
    
    
    # fix \alpha_V to be a constant
    relaxation_penalty = 1.0
    
    clf_controller = Cart_Pole_Joint_Controller(net_nominal, net_policy, relaxation_penalty)


    # Generate training samples
    # For theta values
    num_theta = 50
    theta = torch.Tensor(num_theta).uniform_(-np.pi/2, np.pi/2)
    #theta = torch.Tensor(num_theta).uniform_(-2, 2)
    # For angular velocities
    num_angular = 30
    theta_dot = torch.Tensor(num_angular).uniform_(-3, 3)  # adjust this range if needed
    # For cart positions
    num_pos = 30
    pos = torch.Tensor(num_pos).uniform_(-4, 4)
    # For cart linear velocities
    num_linear = 30
    pos_dot = torch.Tensor(num_linear).uniform_(-3, 3)  # adjust this range if needed
    
    combinations = list(product(pos, theta, pos_dot, theta_dot))
    
    num_samples = num_theta * num_angular * num_pos * num_linear
    
    
    # Create an optimizer for the network parameters and control inputs
    #optimizer = torch.optim.Adam(list(net_nominal.parameters()) + list(net_policy.parameters()) + list(relaxation_penalty.parameters()),  lr=learning_rate, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(list(net_nominal.parameters()) + list(net_policy.parameters()),  lr=learning_rate, betas=(0.9, 0.999))


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
   
    #save checkpoints in training
    # checkpoint_dir = "saved_models/joint_clf_controller_models/cart_pole_checkpoints"
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0  # Initialize the total loss for this epoch
        
        
        #vectorize implementation

        # compute the loss for all samples
        delta_opt = clf_controller.lyapunov_derivative_loss(x_train)

        # Compute the average loss
        total_loss = delta_opt

        # Backpropagate the loss and update the weights and control inputs
        total_loss.backward()
        optimizer.step()
        
    
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.6f}')
            
        # Save a checkpoint every 50 epochs
        # if (epoch + 1) % 50 == 0:
        #     checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth.tar")
        #     save_checkpoint(epoch, net_nominal, optimizer, checkpoint_filename)
        #     print(f"Saved checkpoint at epoch {epoch+1}")
    
        if total_loss.item() < loss_threshold:
            print(f'Training stopped at epoch {epoch + 1}, Loss: {total_loss.item():.4f}')
            break
        
        

    '''
    save and load network parameters
    '''
    torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/baseline_clf.pt")
    torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/baseline_controller.pt")
    
    #torch.save(relaxation_penalty.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/method2_alpha_v.pt")

    # net_fewer_training = LyapunovNet(n_input, n_hidden, n_output)

    # net_fewer_training.load_state_dict(torch.load("saved_models/clf_models/cart_pole_2000sample.pt"))
    # clf_controller = Cart_Pole_CLF_NN_Controller(net_fewer_training, relaxation_penalty=1.0)
    

        
# #############################
# ### For Visualization of the learned CLF
# ###############################


    # Generate a grid of points in the (theta, position) plane
    
    # fix two dimension of state: theta, Omega, position, pos_dot
    
    # pos_dot = np.linspace(-8, 8, 100)
    # omega = np.linspace(-8, 8, 100)
    theta = np.linspace(-3, 3, 50)
    position = np.linspace(-6, 6, 50)
    
    Position, Theta = np.meshgrid(position, theta)
    
    # Convert theta to cos(theta) and sin(theta)
    cos_theta = np.cos(Theta.ravel())
    sin_theta = np.sin(Theta.ravel())
    
    # Convert meshgrid to a tensor
    X_tensor = torch.Tensor(np.vstack([Position.ravel(), cos_theta, sin_theta, np.zeros_like(sin_theta), np.zeros_like(sin_theta)]).T).to(device)
    X_tensor.requires_grad = True
    
    # Initialize arrays to store the values of V and V_dot
    V_values = np.zeros_like(Theta)
    V_dot_values = np.zeros_like(Theta)
    
    # initialize loss_value
    loss_values = np.zeros_like(Theta)
    
    # Define the desired state
    desired_state = torch.tensor([0., 1., 0., 0., 0.])
    
    for i in range(X_tensor.shape[0]):
        # Compute V
        V, gradV = clf_controller.compute_clf(X_tensor[i].unsqueeze(0))
        
        f_x, g_x = clf_controller.cart_pole_dynamics(X_tensor[i])
        
    
        # controller 
        u_opt = clf_controller.compute_policy(X_tensor[i])

        

    
        # Compute LfV and LgV using the optimal control input
        LfV, LgV = clf_controller.compute_lie_derivatives(X_tensor[i], f_x, g_x)
    
        V_dot = LfV.detach().cpu().numpy() + LgV.detach().cpu().numpy() * u_opt.detach().cpu().numpy()
    
        # Convert V and V_dot to numpy and store them in the arrays
        V_scalar = V.squeeze().item()  # Convert tensor to a scalar
        V_values[i // Theta.shape[1], i % Theta.shape[1]] = V_scalar
        
        V_dot_scalar = V_dot.squeeze().item()
        V_dot_values[i // Theta.shape[1], i % Theta.shape[1]] = V_dot_scalar
        
    
    
    
    mask = np.zeros_like(V_dot_values)
    mask[V_dot_values >= 0] = 1
    
    # Set the violations (positive values) to zero for plotting purposes
    V_dot_plot = np.where(V_dot_values >= 0, 0, V_dot_values)
    
    # Identify the violation points
    violation_points = np.column_stack([Position[mask == 1], Theta[mask == 1]])
    
    plt.figure(figsize=(10, 8))
    plt.title("Lyapunov Derivative Visualization")

    
    # Plot the Lyapunov function derivative values
    contour = plt.contourf(Position, Theta, V_dot_plot, levels=10)
    plt.colorbar(label='V_dot')
    plt.scatter(violation_points[:, 0], violation_points[:, 1], color='red', label='Violations')
    plt.xlabel('Position')
    plt.ylabel('Theta')
    plt.title("Lyapunov Derivative Values")
    plt.legend()
    

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig("Cart_Pole_Vdot.png", dpi=300)
    plt.show()
    
    
    
    plt.figure(figsize=(10, 8))
    plt.title("Lyapunov Value Visualization")
    
    # Plot the Lyapunov function derivative values
    contour = plt.contourf(Position, Theta, V_values, levels=20)
    plt.colorbar(label='V_value')
    #plt.scatter(violation_points[:, 0], violation_points[:, 1], color='red', label='Violations', s=10)
    plt.xlabel('Position')
    plt.ylabel('Theta')
    plt.title("Lyapunov Function Values")
    plt.legend()
    

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig("Cart_Pole_V.png", dpi=300)
    plt.show()    
    
    U_values = np.zeros_like(Theta)
    
    for i in range(X_tensor.shape[0]):
        u = clf_controller.compute_policy(X_tensor[i])

        
        
        U_values[i // Theta.shape[1], i % Theta.shape[1]] = u.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(Position, Theta, U_values, levels=20)
    plt.colorbar(label='Control input u')
    plt.xlabel('Position')
    plt.ylabel('Theta')
    plt.title("Learned Control Policy")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig("Cart_Pole_Controller.png", dpi=300)
    plt.show()