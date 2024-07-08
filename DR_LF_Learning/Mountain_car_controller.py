#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
import numpy as np

 
    
class MountainCar_Joint_Controller:
    def __init__(self, net, net_policy, relaxation_penalty = 1.0, power = 0.0015, control_bound = 2.0):
        self.net = net
        self.net_policy = net_policy
        
        self.relaxation_penalty = relaxation_penalty
        
        self.power = power  # car power
        
        self.min_val = -control_bound
        self.max_val = control_bound

    def mountain_car_dynamics(self, x, power = 0.0015):
        # Ensure x is 2D (batch processing)
        if x.dim() == 1:
            x = x.unsqueeze(0)
    
        sin_position = x[:, 0]
        cos_position = x[:, 1]  
        velocity = x[:, 2]  
        
        position = torch.atan2(sin_position, cos_position)
        
        # Dynamics: writing as trig functions
        f_x = torch.stack([
            velocity * cos_position,
            -velocity * sin_position,
            -0.0025 * torch.cos(3 * position)], dim=1).to(x.device)
        g_x = torch.stack([
            torch.zeros_like(sin_position), 
            torch.zeros_like(sin_position),
            power * torch.ones_like(position)], dim=1).to(x.device)
        
        return f_x, g_x


    def compute_clf(self, x, origin=[np.sin(np.pi/6), np.cos(np.pi/6), 0], alpha=0.05):
        
        # Ensure x is 2D (batch processing)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        phi_x = self.net(x)
        phi_0 = self.net(origin_tensor)
        

        V = torch.norm(phi_x - phi_0, dim=1, keepdim=True)**2 + alpha * torch.norm(x - origin_tensor, dim=1, keepdim=True)**2
        gradV = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]

        return V, gradV
    
    def compute_policy(self, x):
        """
        Computes the control policy and enforces constraints on the output.
    
        Args:
        - x: The state.
        - min_val: Minimum permissible value for the control action.
        - max_val: Maximum permissible value for the control action.
    
        Returns:
        - Clamped control action.
        """
        if len(x.shape) == 1:  # If x is a single sample and not a batch
            x = x.unsqueeze(0)  # Add a batch dimension
            
        origin = [np.sin(np.pi/6), np.cos(np.pi/6), 0],
        
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        
        
        
        u_bounded = self.net_policy(x) - self.net_policy(origin_tensor)
        
        
        '''
        symmetric policy 
        '''
        
        # mirrored_state = torch.stack([-x[:, 0], x[:, 1], -x[:,2]], dim=1)
        
        # u_pos = self.net_policy(x) 
        # u_neg = self.net_policy(mirrored_state) 
    
        # # Adjust the sign of the policy
        # u_unbounded = u_pos - u_neg
        
        
        # Apply the sign to the control action
        #u_unbounded = u_unbounded * pos_sign
        
        # Clamp the control action
        u_bounded = torch.clamp(u_bounded, self.min_val, self.max_val)
        
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

    
    
    def lyapunov_derivative_loss(self, x, xi_samples, gamma=0.02):
        
        
        u = self.compute_policy(x)
        
        # Compute the Lyapunov function for all samples
        V, GradV = self.compute_clf(x)
        
        average_power_perturbation = torch.mean(xi_samples)
        
        
        power = self.power + average_power_perturbation
        
        f_x, g_x = self.mountain_car_dynamics(x, power)

        # Compute the Lie derivative of the Lyapunov function for all samples
        LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)

        V_dot = LfV + LgV * u 
        

        #positive_part = torch.relu(V_dot + self.relaxation_penalty * V + gamma * torch.norm(x))
        
        
        positive_part = torch.relu(V_dot + self.relaxation_penalty * V)
        
        
        return torch.mean(positive_part)
    
    
    def compute_w_matrices(self, x):
        """
        Computes the W perturbation matrix for given state x, considering uncertainties in power 
    
        Args:
        - x: The state tensor.
    
        Returns:
        - W perturbation matrix.
        """
        x_dot = x[:, 0]

    
        u = self.compute_policy(x)
    
        # w: Perturbation due to uncertainty in power
        
        w = torch.stack([torch.zeros_like(x_dot),
                         torch.zeros_like(x_dot),
                         u.squeeze()],
                         dim=1).to(x.device)
    
        W = w.unsqueeze(2)
        return W  
    
    
    def dro_lyapunov_derivative_loss_(self, x, xi_samples, r=0.0001, beta=0.1, gamma=0.02):
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
            f_x, g_x = self.mountain_car_dynamics(x, power = self.power + xi)
            LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)
            
            V_dot = LfV + LgV * u 
            V_dot_samples.append(V_dot)
            
        V_dot_max = V_dot_samples[0]
        for V_dot in V_dot_samples[1:]:
            V_dot_max = torch.max(V_dot_max, V_dot)     
            

        # Compute V_grad * w for the batch
        V_grad_w = torch.bmm(V_grad.view(-1, 1, 3), W)
        
        
        
        # Compute the infinity norm
        V_grad_w_inf_norm = torch.norm(V_grad_w, dim=2, p=2)
        
        # Compute the positive part for loss
        positive_part = torch.relu(r * V_grad_w_inf_norm / beta + V_dot_max + self.relaxation_penalty * V) 
        
        #positive_part = torch.relu(V_dot_max + self.relaxation_penalty * V) 
        
        
        # Return the mean of positive part
        return torch.mean(positive_part)
    

    def dro_lyapunov_derivative_loss_uniform(self, x, xi_samples, r=0.00008, beta=0.1, gamma=0.02):
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
        
        # Compute the maximum infinity norm among all x samples
        # V_grad_w_inf_norm_max = torch.max(torch.linalg.vector_norm(V_grad_w, ord=float('inf')))

        # 2-norm
        V_grad_w_norm_max = torch.max(torch.linalg.vector_norm(V_grad_w, dim=2))

        # print('V_grad_w_max:', V_grad_w_inf_norm_max)

        # Compute V_dot for all x samples and xi samples
        V_dot_samples = []
        for xi in xi_samples:
            f_x, g_x = self.mountain_car_dynamics(x, power = self.power + xi)
            LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)
            
            V_dot = LfV + LgV * u 
            V_dot_samples.append(V_dot)
            
        V_dot_max = V_dot_samples[0]
        for V_dot in V_dot_samples[1:]:
            V_dot_max = torch.max(V_dot_max, V_dot)  

        
        # Compute the positive part for loss
        positive_part = torch.relu(r * V_grad_w_norm_max / beta + V_dot_max + self.relaxation_penalty * V)
        
        # Return the loss value
        return torch.mean(positive_part)