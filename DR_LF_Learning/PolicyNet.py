#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch

class PolicyNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers = 3):
        super(PolicyNet, self).__init__()
        torch.manual_seed(2)

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_input, n_hidden))
        
        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))
        
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

        self.activation = torch.nn.Tanh()
        
        # Initialize weights
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
        
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        out = self.output_layer(x)
        return out