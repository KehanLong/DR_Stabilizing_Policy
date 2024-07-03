#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
import numpy as np

 
    
class InvertedPendulum_Joint_Controller:
    def __init__(self, net, net_policy, relaxation_penalty = 1.0, m =1.0, l = 1.0, b = 0.13, control_bounds = 15.0):
        self.net = net
        self.net_policy = net_policy
        
        self.relaxation_penalty = relaxation_penalty
        
        self.g = 9.81  # Acceleration due to gravity
        self.m = m  # Mass of the pendulum
        self.l = l  # Length of the pendulum
        
        self.b = b # damping of the pendulum
        
        self.control_bounds = control_bounds

    def inverted_pendulum_dynamics(self, x, m = 1.0, l = 1.0, b = 0.13):
        # Ensure x is 2D (batch processing)
        if x.dim() == 1:
            x = x.unsqueeze(0)
    
        sin_theta = x[:, 0]  # cos Angle theta 
        cos_theta = x[:, 1]  # sin Angle theta
        theta_dot = x[:, 2]  # Angular velocity for the whole batch
        
        # Dynamics of the inverted pendulum with damping
        f_x = torch.stack([
            cos_theta * theta_dot,
            -sin_theta * theta_dot, 
            (1.0 * self.g / l * sin_theta) - (b * theta_dot / (m * l * l))], dim=1).to(x.device)
        
        
        # without damping
        # f_x = torch.stack([
        #     cos_theta * theta_dot, 
        #     -sin_theta * theta_dot, 
        #     (1.0 * self.g / l * sin_theta)], dim=1).to(x.device)
        

        # input matrix 
        g_x = torch.stack([
            torch.zeros_like(theta_dot), 
            torch.zeros_like(theta_dot), 
            1.0 / (m * l * l) * torch.ones_like(theta_dot)], dim=1).to(x.device)
        
        
        
        
        return f_x, g_x


    def compute_clf(self, x, origin=[np.sin(0), np.cos(0), 0], alpha=0.1):
        
        # Ensure x is 2D (batch processing)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        phi_x = self.net(x)
        phi_0 = self.net(origin_tensor)
        V = torch.norm(phi_x - phi_0, dim=1, keepdim=True)**2 + alpha * torch.norm(x - origin_tensor, dim=1, keepdim=True)**2
        
        
        gradV = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]

        return V, gradV
    
    # def compute_clf(self, x, origin=[np.sin(0), np.cos(0), 0], alpha=1.0, beta=0.1):
    #     # Ensure x is 2D (batch processing)
    #     if x.dim() == 1:
    #         x = x.unsqueeze(0)
            
    #     origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
    #     phi_x = self.net(x)
    #     phi_0 = self.net(origin_tensor)
        
    #     # Basic V based on distance from the origin
    #     V_distance = torch.norm(phi_x - phi_0, dim=1, keepdim=True)**2
        
    #     # Additional terms for energy and angular momentum
    #     sin_theta = x[:, 0]
    #     cos_theta = x[:, 1]
    #     theta_dot = x[:, 2]
        
        
    #     potential_energy = self.m * self.g * self.l * (1 - cos_theta)  # mgl(1 - cos(theta))
    #     kinetic_energy = 0.5 * self.m * (self.l * theta_dot)**2  # 0.5 * m * l^2 * theta_dot^2
        
    #     # Total energy
    #     V_energy = potential_energy + kinetic_energy
        
    #     # Combine the components
    #     V = V_energy 
        
    #     gradV = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    
    #     return V, gradV

    
    def compute_policy(self, x):
        """
        Computes the control policy and enforces constraints on the output.
    
        Args:
        - x: The state.
        - -self.control_bounds: Minimum permissible value for the control action.
        - self.control_bounds: Maximum permissible value for the control action.
    
        Returns:
        - Clamped control action.
        """
        if len(x.shape) == 1:  # If x is a single sample and not a batch
            x = x.unsqueeze(0)  # Add a batch dimension
            

        origin = [np.sin(0), np.cos(0), 0]
        
        
        
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        
        
        u_unbounded = self.net_policy(x) - self.net_policy(origin_tensor)
        
        
        #Clamp the control action
        u_bounded = torch.clamp(u_unbounded, -self.control_bounds, self.control_bounds)
        
        #u_tanh = torch.tanh(u_unbounded)

        # Scale the output of tanh to the control bounds
      
        #u_bounded = u_tanh * self.control_bounds
        
        
        return u_bounded
    
    
    def compute_lie_derivatives(self, x, f_x, g_x):
        # Ensure x is 2D (batch processing)
        if x.dim() == 1:
            x = x.unsqueeze(0)
    
        _, gradV = self.compute_clf(x)
        
        # Use bmm (batch matrix multiplication) for batched data
        LfV = torch.bmm(gradV.unsqueeze(1), f_x.unsqueeze(2)).squeeze(2)
        LgV = torch.bmm(gradV.unsqueeze(1), g_x.unsqueeze(2)).squeeze(2)
        
        return LfV, LgV

    
    #vectorized version
    def lyapunov_derivative_loss(self, x, xi_samples, gamma=0.02):
        
        
        u = self.compute_policy(x)
        
        # Compute the Lyapunov function for all samples
        V, GradV = self.compute_clf(x)
        
        average_mass_perturbation = torch.mean(xi_samples[:, 0])
        average_damping_perturbation = torch.mean(xi_samples[:, 1])
        
        
        mass = self.m + average_mass_perturbation
        damping = self.b + average_damping_perturbation
        
        f_x, g_x = self.inverted_pendulum_dynamics(x, mass, self.l, damping)
        
        #f_x, g_x = self.inverted_pendulum_dynamics(x, self.m, self.l, self.b)

        # Compute the Lie derivative of the Lyapunov function for all samples
        LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)

        V_dot = LfV + LgV * u 
        

        #positive_part = torch.relu(V_dot + self.relaxation_penalty * V + gamma * torch.norm(x))
        
        
        positive_part = torch.relu(V_dot + self.relaxation_penalty * V)
        
        
        return torch.mean(positive_part)
    
    
    # def compute_w1(self, x):
    #     """
    #     Computes the w1 perturbation vector for given state x, w1 represents the uncertainty in mass
    
    #     Args:
    #     - x: The state tensor.
    
    #     Returns:
    #     - w1 perturbation vector.
    #     """
        
    #     # Extract the angular velocity theta_dot from x for the whole batch
    #     theta_dot = x[:, 2]
    
    #     m = self.m
    #     l = self.l
    #     b = self.b
    
    #     # Compute the control policy u
    #     u = self.compute_policy(x)
        

    #     # Calculate the perturbation vector w1
    #     w1_drift = torch.stack([torch.zeros_like(theta_dot),
    #                             torch.zeros_like(theta_dot),
    #                             (b * theta_dot) / (m**2 * l**2)],
    #                             dim=1).to(x.device)
        
    #     w1_ctrl_value = -1.0 / (m**2 * l**2) * torch.ones_like(theta_dot).to(x.device)
    #     w1_ctrl = torch.stack([torch.zeros_like(theta_dot),
    #                            torch.zeros_like(theta_dot),
    #                            w1_ctrl_value],
    #                           dim=1).to(x.device) * u
    
    #     # Return the combined w1
    #     w1_combined = w1_drift + w1_ctrl
    #     return w1_combined
    
    
    def compute_w_matrices(self, x):
        """
        Computes the W perturbation matrix for given state x, considering uncertainties in mass (w1) and damping (w2).
    
        Args:
        - x: The state tensor.
    
        Returns:
        - W perturbation matrix.
        """
        theta_dot = x[:, 2]
        m = self.m
        l = self.l
        b = self.b
    
        u = self.compute_policy(x)
    
        # w1: Perturbation due to uncertainty in mass
        
        w1 = torch.stack([torch.zeros_like(theta_dot),
                          torch.zeros_like(theta_dot),
                          (b * theta_dot - u.squeeze()) / (m**2 * l**2)],
                         dim=1).to(x.device)
    
        # w2: Perturbation due to uncertainty in damping
        w2 = torch.stack([torch.zeros_like(theta_dot),
                          torch.zeros_like(theta_dot),
                          -theta_dot / (m * l**2)],
                         dim=1).to(x.device)
    
        # Combine w1 and w2 to form the W matrix
        W = torch.cat([w1.unsqueeze(2), w2.unsqueeze(2)], dim=2)
        return W  
    
    
    def dro_lyapunov_derivative_loss_(self, x, xi_samples, r=0.01, beta=0.1, gamma=0.02):
        """
        Computes the DR Lyapunov derivative loss.
        
        Args:
        - x: State tensor.
        - xi_samples: Perturbation tensor samples xi.
        - r: Wasserstein Radius.
        - beta: Risk parameter.
    
        Returns:
        - DR Lyapunov derivative loss.
        """
        
        # Compute w1 perturbation
        W = self.compute_w_matrices(x)
        V, V_grad = self.compute_clf(x)
        u = self.compute_policy(x)

        # Compute V_dot for all xi samples
        V_dot_samples = []
        for xi in xi_samples:
            f_x, g_x = self.inverted_pendulum_dynamics(x, m=(self.m+xi[0]), l=self.l, b=(self.b+xi[1]))
            LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)
            
            V_dot = LfV + LgV * u 
            V_dot_samples.append(V_dot)
            
        #print('V_dot_samples:', V_dot_samples)    
            
        
        V_dot_max = V_dot_samples[0]
        for V_dot in V_dot_samples[1:]:
            V_dot_max = torch.max(V_dot_max, V_dot)
            
      
        # Compute V_grad * w for the batch
        V_grad_w = torch.bmm(V_grad.view(-1, 1, 3), W)
        
        # Compute the infinity norm
        #V_grad_w_inf_norm = torch.norm(V_grad_w, dim=2, p=float('inf'))
        
        V_grad_w_inf_norm = torch.norm(V_grad_w, dim=2, p=2)
        
        # Compute the positive part for loss
        positive_part = torch.relu(r * V_grad_w_inf_norm / beta + V_dot_max + self.relaxation_penalty * V) 
        
        # Return the mean of positive part
        return torch.mean(positive_part)  

    
      
    def dro_lyapunov_derivative_loss_uniform(self, x, xi_samples, r=0.003, beta=0.1, gamma=0.02):
        """
        Computes the DR Lyapunov derivative loss with uniform formulation.
        Args:
        - x: State tensor samples.
        - xi_samples: Perturbation tensor samples xi.
        - r: Wasserstein Radius.
        - beta: Risk parameter.
        Returns:
        - DR Lyapunov derivative loss with uniform formulation.
        """
        # Compute w perturbation for the batch
        W = self.compute_w_matrices(x)
        V, V_grad = self.compute_clf(x)
        u = self.compute_policy(x)

        # Compute V_grad * w for the batch
        V_grad_w = torch.bmm(V_grad.view(-1, 1, 3), W)

        # Compute the maximum 2-norm among all x samples
        V_grad_w_norm_max = torch.max(torch.norm(V_grad_w, dim=2, p=2))

        # Compute V_dot for all x samples and xi samples
        V_dot_samples = []
        for xi in xi_samples:
            f_x, g_x = self.inverted_pendulum_dynamics(x, m=(self.m+xi[0]), l=self.l, b=(self.b+xi[1]))
            LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)
            V_dot = LfV + LgV * u
            V_dot_samples.append(V_dot)

        # Compute the LogSumExp approximation of the maximum V_dot among all x samples and xi samples
        V_dot_stack = torch.stack(V_dot_samples)
        V_dot_max = torch.logsumexp(V_dot_stack, dim=0)


        # Compute the loss
        positive_part = torch.relu(r * V_grad_w_norm_max / beta + V_dot_max + self.relaxation_penalty * V) 

        return torch.mean(positive_part)