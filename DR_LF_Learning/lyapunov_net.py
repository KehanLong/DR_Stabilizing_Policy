#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kehan
"""

import torch

# class LyapunovNet(torch.nn.Module):
#     def __init__(self, n_input, n_hidden, n_output):
#         super(LyapunovNet, self).__init__()
#         torch.manual_seed(2)
#         self.layer1 = torch.nn.Linear(n_input, n_hidden)
#         self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
#         self.layer3 = torch.nn.Linear(n_hidden, n_hidden)
#         self.layer4 = torch.nn.Linear(n_hidden, n_output)

#         self.activation = torch.nn.Tanh()
#         torch.nn.init.xavier_uniform_(self.layer1.weight)
#         torch.nn.init.xavier_uniform_(self.layer2.weight)
#         torch.nn.init.xavier_uniform_(self.layer3.weight)
#         torch.nn.init.xavier_uniform_(self.layer4.weight)

#     def forward(self, x):
#         h_1 = self.activation(self.layer1(x))
#         h_2 = self.activation(self.layer2(h_1))
#         h_3 = self.activation(self.layer3(h_2))
#         out = self.layer4(h_3)
#         return out
    
    
class LyapunovNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=3):
        super(LyapunovNet, self).__init__()
        #torch.manual_seed(2)

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
    
