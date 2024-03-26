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
from Mountain_car_controller import MountainCar_Joint_Controller


from itertools import product


def generate_uncertainty_samples(num_samples=4):
    """
    Generate samples of xi based on the distributions for mass and damping uncertainties.

    Args:
    - num_samples: Number of samples to generate.

    Returns:
    - A tensor of shape (num_samples, 1) containing samples of xi.
    """
    xi_samples = torch.normal(mean=0.0, std = 0.0002, size = (num_samples,))
    #xi_m_samples = torch.normal(mean=0.04, std=0.04, size=(num_samples,))
    #xi_b_samples = torch.FloatTensor(num_samples).uniform_(-0.03, 0.02)
    #xi_b_samples = torch.normal(mean=0.0, std=0.02, size=(num_samples,))

    return xi_samples



def train_clf_nn_controller():
    n_input = 3
    n_hidden = 16
    n_output = 4
    num_of_layers = 3
    
    n_control_hidden = 16
    
    n_control = 1

    num_epochs = 3000

    learning_rate = 0.003
    loss_threshold = 1e-3
    batch_size = 128
    
    
    
    
    xi_samples_num = 3
    
    xi_samples = generate_uncertainty_samples(xi_samples_num)
    
    print('xi_samples:', xi_samples)
    
    nominal_power = 0.0015 
    
    # Compute the average perturbations for mass and damping
    average_perturbation = torch.mean(xi_samples)

    power = nominal_power + average_perturbation
    
    print('average power:', power)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers)
    net_nominal.to(device)
    net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3)
    net_policy.to(device)


    
    # Generate training samples
    # For theta values
    num_position = 40
    position = torch.Tensor(num_position).uniform_(-2.0, 2.0)
    # For angular velocities
    num_velocity = 40
    velocity = torch.Tensor(num_velocity).uniform_(-1.0, 1.0)  

    
    
    combinations = list(product(position, velocity))
    
    num_samples = num_position * num_velocity
    
    
    x_train = torch.zeros([num_samples, n_input], requires_grad=True)
    # Convert the combinations list to tensor
    x_original = torch.tensor(combinations, dtype=torch.float32)
    x_original.requires_grad = True
    
    x_train_temp = torch.zeros_like(x_train)
    
    x_train_temp[:, 0] = torch.sin(x_original[:, 0])
    x_train_temp[:, 1] = torch.cos(x_original[:, 0])
    x_train_temp[:, 2] = x_original[:, 1]
    
    
    x_train.data = x_train_temp.data
    
    
    x_train = x_train.to(device)
    xi_samples = xi_samples.to(device)
    
    # Dictionary to store the trained controllers
    trained_controllers = {}


    


    for model_type in ['nominal', 'dro']:
        print(f"\nTraining {model_type} model")

        # Reset model and optimizer for each training type
        net_nominal = LyapunovNet(n_input, n_hidden, n_output, num_of_layers).to(device)
        net_policy = PolicyNet(n_input, n_control_hidden, n_control, 3).to(device)
        
        
        
        # warm initialization for dro
        if model_type == 'dro':
            # Load the trained weights from the nominal model
            net_nominal.load_state_dict(torch.load("saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_joint_clf_1.pt"))
            net_policy.load_state_dict(torch.load("saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_controller_1.pt"))

        
        # Instantiate the controller with specific control bounds
        if model_type == 'nominal':
            clf_controller = MountainCar_Joint_Controller(net_nominal, net_policy, relaxation_penalty=0.2, power=0.0015, control_bound=1.5)
        else:  # 'dro'
            clf_controller = MountainCar_Joint_Controller(net_nominal, net_policy, relaxation_penalty=0.2, power=0.0015, control_bound=2.0)
    
        optimizer = torch.optim.Adam(list(net_nominal.parameters()) + list(net_policy.parameters()), lr=learning_rate, betas=(0.9, 0.999))

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            total_loss = 0.0

            if model_type == 'nominal':
                # Compute the loss for all samples (baseline training)
                delta_opt = clf_controller.lyapunov_derivative_loss(x_train, xi_samples)
            else:  # 'dro'
                # Compute the loss for all samples (DRO training)
                delta_opt = clf_controller.dro_lyapunov_derivative_loss_(x_train, xi_samples)

            total_loss = delta_opt
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], {model_type} Loss: {total_loss.item():.6f}')

            if total_loss.item() < loss_threshold:
                print(f'Training stopped at epoch {epoch + 1}, {model_type} Loss: {total_loss.item():.4f}')
                break
            
        trained_controllers[model_type] = clf_controller

        # Saving model parameters
        if model_type == 'nominal':
            torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_joint_clf_1.pt")
            torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/mountain_car/baseline_asympto_controller_1.pt")
        else:  # 'dro'
            torch.save(net_nominal.state_dict(), "saved_models/joint_clf_controller_models/mountain_car/dro_joint_clf_1.pt")
            torch.save(net_policy.state_dict(), "saved_models/joint_clf_controller_models/mountain_car/dro_controller_1.pt")
            
            
            
    return trained_controllers
    
    
def evaluate_clf_controller(clf_controller, file_prefix='Mountain'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate a grid of points in the (theta, omega) plane
    theta = np.linspace(-2.0, 2.0, 50)
    omega = np.linspace(-1.0, 1.0, 50)
    Theta, Omega = np.meshgrid(theta, omega)
    
    
    cos_theta = np.cos(Theta.ravel())
    sin_theta = np.sin(Theta.ravel())
    
    # Convert meshgrid to a tensor
    X_tensor = torch.Tensor(np.vstack([sin_theta, cos_theta, Omega.ravel()]).T).to(device)
    X_tensor.requires_grad = True
    
    # Initialize arrays to store the values of V and V_dot
    V_values = np.zeros_like(Theta)
    V_dot_values = np.zeros_like(Theta)
    
    # Define the desired state
    desired_state = torch.tensor([np.sin(np.pi/6), np.cos(np.pi/6), 0.])
    
    for i in range(X_tensor.shape[0]):
        # Compute V
        V, gradV = clf_controller.compute_clf(X_tensor[i])
        
        f_x, g_x = clf_controller.mountain_car_dynamics(X_tensor[i])
    
        # Solve the CLF QP to get the optimal control input
        u_opt = clf_controller.compute_policy(X_tensor[i])
        

        
        # Compute LfV and LgV using the optimal control input
        LfV, LgV = clf_controller.compute_lie_derivatives(X_tensor[i], f_x, g_x)
    
        V_dot = LfV.detach().cpu().numpy() + LgV.detach().cpu().numpy() * u_opt.detach().cpu().numpy()
        
    
        # Convert V and V_dot to numpy and store them in the arrays
        V_values[i // Theta.shape[1], i % Theta.shape[1]] = V.detach().cpu().numpy()
        V_dot_values[i // Theta.shape[1], i % Theta.shape[1]] = V_dot.item()
    
    
    mask = np.zeros_like(V_dot_values)
    mask[V_dot_values >= 0] = 1
    
    # Set the violations (positive values) to zero for plotting purposes
    V_dot_plot = np.where(V_dot_values >= 0, 0, V_dot_values)
    
    # Identify the violation points
    violation_points = np.column_stack([Theta[mask == 1], Omega[mask == 1]])
    
    plt.figure(figsize=(10, 8))
    
    # Plot the Lyapunov function derivative values
    contour = plt.contourf(Theta, Omega, V_dot_plot, levels=10)
    plt.colorbar(label='V_dot')
    plt.scatter(violation_points[:, 0], violation_points[:, 1], color='red', label='Violations', s=10)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Lyapunov Derivative Values")
    plt.legend()
    

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig("Mountain_Car_Vdot.png", dpi=300)
    
    plt.show()
    
    
    plt.figure(figsize=(10, 8))
    
    # Plot the Lyapunov function derivative values
    contour = plt.contourf(Theta, Omega, V_values, levels=20)
    plt.colorbar(label='V_value')
    #plt.scatter(violation_points[:, 0], violation_points[:, 1], color='red', label='Violations', s=10)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Mountain Car Lyapunov Function Values")
    plt.legend()
    

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Mountain_Car_V.png", dpi=300)
    plt.show()    
    
    U_values = np.zeros_like(Theta)
    
    for i in range(X_tensor.shape[0]):
        u = clf_controller.compute_policy(X_tensor[i])
        U_values[i // Theta.shape[1], i % Theta.shape[1]] = u.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(Theta, Omega, U_values, levels=20)
    plt.colorbar(label='Control input u')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Learned Control Policy")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Mountain_car_controller.png", dpi=300)
    plt.show()

    
    origin = [np.sin(np.pi/6), np.cos(np.pi/6), 0]
    origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(device)
    
    origin_test = clf_controller.compute_policy(origin_tensor)
    
    print('what is control here:', origin_test)
    

    


if __name__ == "__main__":
    trained_controllers = train_clf_nn_controller()
    nominal_controller = trained_controllers['nominal']
    dro_controller = trained_controllers['dro']
    #evaluate_clf_controller(nominal_controller)
    #evaluate_clf_controller(dro_controller)