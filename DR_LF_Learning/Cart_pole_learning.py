#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt


from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet
from Cart_pole_joint_controller import Cart_Pole_Joint_Controller


from itertools import product

def train_clf_nn_controller():
    # Network parameters
    n_input = 5
    n_hidden = 32
    n_output = 4
    num_of_layers = 4
    n_control_hidden = 32
    n_control = 1
    n_control_layer = 4
    num_epochs = 30
    loss_threshold = 1e-4
    batch_size = 256  # Updated batch size
    

    # Training setup
    learning_rate = 0.002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate networks
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers).to(device)
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, n_control_layer).to(device)
    
    
    # LOAD 
    # baseline_clf_saved_model = "saved_models/joint_clf_controller_models/cart_pole/baseline_joint_clf_1.pt"
    # baseline_policy_model = "saved_models/joint_clf_controller_models/cart_pole/baseline_controller_1.pt"
    
    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers).to(device)
    
    #net_nominal.load_state_dict(torch.load(baseline_clf_saved_model))
    
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, n_control_layer).to(device)
    
    #net_policy.load_state_dict(torch.load(baseline_policy_model))
    # Set relaxation penalty
    
    relaxation_penalty = 0.4  # Fixed value
    clf_controller = Cart_Pole_Joint_Controller(net_nominal, net_policy, relaxation_penalty)

    # Optimizer
    optimizer = torch.optim.Adam(list(net_nominal.parameters()) + list(net_policy.parameters()), lr=learning_rate, betas=(0.9, 0.999))

    # Data generation
    num_theta = 60
    theta = torch.Tensor(num_theta).uniform_(-np.pi/2, np.pi/2)
    #theta = torch.Tensor(num_theta).uniform_(-np.pi, np.pi)
    #theta = torch.Tensor(num_theta).uniform_(-2, 2)
    # For angular velocities
    num_angular = 40
    theta_dot = torch.Tensor(num_angular).uniform_(-2.0, 2.0)  # adjust this range if needed
    # For cart positions
    num_pos = 20
    pos = torch.Tensor(num_pos).uniform_(-3, 3)
    # For cart linear velocities
    num_linear = 20
    pos_dot = torch.Tensor(num_linear).uniform_(-2.0, 2.0)  # adjust this range if needed
    
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
        

    # Convert to a PyTorch DataLoader
    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (batch_data,) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
    
            # Compute the loss
            loss = clf_controller.lyapunov_derivative_loss(batch_data)
            total_loss += loss.item()
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Mini-Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.6f}')
                
                # Write to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
    
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.6f}')
        writer.add_scalar('Loss/average', average_loss, epoch)
    
        if average_loss < loss_threshold:
            print(f'Training stopped at epoch {epoch + 1}, Average Loss: {average_loss:.4f}')
            break
        
    # Close the TensorBoard writer
    writer.close()


    # Save the trained models
    torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/baseline_joint_clf_1.pt")
    torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/baseline_controller_1.pt")

    #torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/dro_joint_clf.pt")
    #torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/cart_pole/dro_controller.pt")
    
    return clf_controller


def evaluate_clf_controller(clf_controller, file_prefix='Cart Pole'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    plt.savefig("Cart_Pole_Vdot.png", dpi=300)
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
    plt.savefig("Cart_Pole_V.png", dpi=300)
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
    plt.savefig("Cart_Pole_Controller.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    clf_controller = train_clf_nn_controller()
    evaluate_clf_controller(clf_controller)


