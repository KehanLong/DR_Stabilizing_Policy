import torch
import numpy as np

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the module search path
sys.path.insert(0, current_dir)

from PolicyNet import PolicyNet
from lyapunov_net import LyapunovNet
    


class MountainCarModel(torch.nn.Module):
    def __init__(self, lyapunov_net, controller_net, power=0.0015, origin_tensor=None):
        super().__init__()
        self.lyapunov_net = lyapunov_net
        self.controller_net = controller_net
        self.power = power

        self.origin_tensor = origin_tensor

    def forward(self, x):
        # Ensure the input tensor has requires_grad=True
        x = x.detach().requires_grad_(True)
        print(f"Input shape: {x.shape}")

        # Compute the control input
        u = self.controller_net(x)

        # Compute the system dynamics
        sin_position = x[:, 0]
        cos_position = x[:, 1]
        velocity = x[:, 2]
        position = torch.atan2(sin_position, cos_position)
        f_x = torch.stack([
            velocity * cos_position,
            -velocity * sin_position,
            -0.0025 * torch.cos(3 * position)], dim=1).to(x.device)
        g_x = torch.stack([
            torch.zeros_like(sin_position),
            torch.zeros_like(sin_position),
            self.power * torch.ones_like(position)], dim=1).to(x.device)

        phi_x = self.lyapunov_net(x)
        print(f"phi_x shape: {phi_x.shape}")

        phi_0 = self.lyapunov_net(self.origin_tensor)
        print(f"phi_0 shape: {phi_0.shape}")

        V = torch.norm(phi_x - phi_0, dim=1, keepdim=True)**2 + 0.05 * torch.norm(x - self.origin_tensor, dim=1, keepdim=True)**2
        print(f"V shape: {V.shape}")

        return V
        #gradV = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        

        # # Compute the Lie derivatives
        # LfV = torch.bmm(gradV.unsqueeze(1), f_x.unsqueeze(2)).squeeze(2)
        # LgV = torch.bmm(gradV.unsqueeze(1), g_x.unsqueeze(2)).squeeze(2)

        # # Compute V_dot
        # V_dot = LfV + LgV * u

        return V

def create_mountain_car_model(lyapunov_parameters, controller_parameters):
    lyapunov_net = LyapunovNet(**lyapunov_parameters)
    controller_net = PolicyNet(**controller_parameters)

    checkpoint_lyapunov = torch.load('DR_LF_Learning/saved_models/joint_clf_controller_models/mountain_car/baseline_clf.pt')
    checkpoint_controller = torch.load('DR_LF_Learning/saved_models/joint_clf_controller_models/mountain_car/baseline_controller.pt')

    lyapunov_net.load_state_dict(checkpoint_lyapunov)
    controller_net.load_state_dict(checkpoint_controller)

    origin = [np.sin(np.pi/6), np.cos(np.pi/6), 0]
    origin_tensor = torch.tensor(origin, dtype=torch.float32).unsqueeze(0)

    return MountainCarModel(lyapunov_net, controller_net, origin_tensor=origin_tensor)

def box_data(lower_limit, upper_limit, ndim, scale, hole_size):
    data_min = scale * torch.tensor(lower_limit, dtype=torch.get_default_dtype()).unsqueeze(0)
    data_max = scale * torch.tensor(upper_limit, dtype=torch.get_default_dtype()).unsqueeze(0)
    return {
        'data_min': data_min,
        'data_max': data_max,
    }
