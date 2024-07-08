#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch
import numpy as np


    
class Cart_Pole_Joint_Controller:
    def __init__(self, net, net_policy = None, length = 1.0, relaxation_penalty = 1.0, control_bounds = 10.0):
        self.net = net
        self.net_policy = net_policy
        
        self.relaxation_penalty = relaxation_penalty

        self.length = length

        self.control_bounds = control_bounds
           
    
    # vectorized version to enable batch training
    def cart_pole_dynamics(self, x, length = 1.0, mp = 1.0, mc = 1.0):
        
        
        # training baseline parameters
        g = 9.81  # gravity
        
        mp = mp  # mass of pole
        mc = mc # mass of cart
        l = length  # length to center of mass of pendulum
        
        if len(x.shape) == 1:  # If x is a single sample and not a batch
            x = x.unsqueeze(0)  # Add a batch dimension
    
    
        pos = x[:, 0]
        cos_theta = x[:, 1]
        sin_theta = x[:, 2]
        pos_dot = x[:, 3]
        theta_dot = x[:, 4]
    
        denominator1 = mc + mp * sin_theta**2
        denominator2 = l * denominator1
    
        theta_double_dot = (- mp * l * theta_dot**2 * sin_theta * cos_theta + (mc + mp) * g * sin_theta) / denominator2
        x_double_dot = (mp * sin_theta * (-l * theta_dot**2 + g * cos_theta)) / denominator1

    
        f_x = torch.stack([pos_dot, -sin_theta * theta_dot, cos_theta * theta_dot, x_double_dot, theta_double_dot], dim=1).to(x.device)
        g_x = torch.stack([torch.zeros_like(sin_theta), torch.zeros_like(sin_theta), torch.zeros_like(sin_theta),  1 / denominator1, cos_theta / denominator2], dim=1).to(x.device)
    
        return f_x, g_x
    
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
        
        '''
        following is the baseline (no symmetry)
        '''
        origin = [0, 1, 0, 0, 0]
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        u_unbounded = self.net_policy(x) - self.net_policy(origin_tensor)
        
        min_val = -self.control_bounds
        max_val = self.control_bounds
        # Clamp the control action
        u_bounded = torch.clamp(u_unbounded, min_val, max_val)
        
        return u_bounded

    
    #vectorized version 
    def compute_clf(self, x, origin=[0, 1, 0, 0, 0], alpha=0.04):
        # Ensure x is 2D (batch processing)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        phi_x = self.net(x.float())
        phi_0 = self.net(origin_tensor)

        V = torch.norm(phi_x - phi_0, dim=1, keepdim=True)**2 + alpha * torch.norm(x - origin_tensor, dim=1, keepdim=True)**2
        gradV = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        
        
    
        return V, gradV

    #vectorized version
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
    def lyapunov_derivative_loss(self, x, xi_samples):

        u = self.compute_policy(x)
        
        V, gradV = self.compute_clf(x)

        average_length_perturbation = torch.mean(xi_samples)
        
        
        length = self.length + average_length_perturbation
        
        f_x, g_x = self.cart_pole_dynamics(x, length)
        
        LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)
        
        V_dot = LfV + LgV * u 
        
        
        
        positive_part = torch.relu(V_dot + self.relaxation_penalty * V)
        
        
        return torch.mean(positive_part)
