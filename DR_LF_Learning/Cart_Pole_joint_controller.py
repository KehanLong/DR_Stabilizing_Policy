#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:22:24 2023

@author: kehan
"""

import torch
import numpy as np


    
class Cart_Pole_Joint_Controller:
    def __init__(self, net, net_policy = None, relaxation_penalty = 1.0, control_bounds = 15.0):
        self.net = net
        self.net_policy = net_policy
        
        self.relaxation_penalty = relaxation_penalty
           
    
    # vectorized version to enable batch training
    def cart_pole_dynamics(self, x):
        
        
        # training baseline parameters
        mp = 3.2  # mass of pole
        mc = 1.2 # mass of cart
        l = 1.4  # length to center of mass of pendulum
        g = 9.81  # gravity
        
        mp = 1.0  # mass of pole
        mc = 1.0 # mass of cart
        l = 1.0  # length to center of mass of pendulum
        #testing parameters, with uncertainty
        # mp = 4.8  # mass of pole
        # mc = 0.7 # mass of cart
        # l = 1.8  # length to center of mass of pendulum
        # g = 9.81  # gravity
        
        if len(x.shape) == 1:  # If x is a single sample and not a batch
            x = x.unsqueeze(0)  # Add a batch dimension
    
    
        pos = x[:, 0]
        cos_theta = x[:, 1]
        sin_theta = x[:, 2]
        pos_dot = x[:, 3]
        theta_dot = x[:, 4]
    
        denominator1 = mc + mp * sin_theta**2
        denominator2 = l * (mc + mp * sin_theta**2)
    
        theta_double_dot = - (mp * l * theta_dot**2 * sin_theta * cos_theta + (mc + mp) * g * sin_theta) / denominator2
        x_double_dot = (mp * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denominator1
    
        f_x = torch.stack([pos_dot, -sin_theta * theta_dot, cos_theta * theta_dot, x_double_dot, theta_double_dot], dim=1).to(x.device)
        g_x = torch.stack([torch.zeros_like(sin_theta), torch.zeros_like(sin_theta), torch.zeros_like(sin_theta),  1 / denominator1, -cos_theta / denominator2], dim=1).to(x.device)
    
        return f_x, g_x
    
    def compute_policy(self, x, min_val=-10.0, max_val=10.0):
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
        following is a central symmetric policy
        '''
    
        # Compute the policy for the absolute state values
        # mirrored_state = torch.stack([-x[:, 0], x[:, 1], -x[:,2], -x[:, 3], -x[:, 4]], dim=1)
    
        # origin = [0, 1, 0, 0, 0]
        # #origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        # u_pos = self.net_policy(x) 
        # u_neg = self.net_policy(mirrored_state) 
    
        # # Adjust the sign of the policy
        # u_unbounded = u_pos - u_neg
    
        # # Clamp the control action
        # u_bounded = torch.clamp(u_unbounded, min_val, max_val)



        '''
        following is the baseline (no symmetry)
        '''
        origin = [0, 1, 0, 0, 0]
        origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0).to(x.device)
        u_unbounded = self.net_policy(x) - self.net_policy(origin_tensor)
        
        
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
    def lyapunov_derivative_loss(self, x):

        u = self.compute_policy(x)
        
        V, gradV = self.compute_clf(x)
        
        f_x, g_x = self.cart_pole_dynamics(x)
        
        LfV, LgV = self.compute_lie_derivatives(x, f_x, g_x)
        
        V_dot = LfV + LgV * u 
        
        #positive_part = torch.relu(V_dot + gamma * torch.norm(x))
        
        
        positive_part = torch.relu(V_dot + self.relaxation_penalty * V)
        
        #making alpha_v adaptive / trainable
        # relaxation_values = self.relaxation_penalty(x)
        # positive_part = torch.relu(V_dot + relaxation_values * V)
        
        
        
        return torch.mean(positive_part)
