#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:18:29 2023

@author: kehan
"""

import sys
sys.path.append('../')  

from Gymnasium.gymnasium.envs.classic_control.acrobot import AcrobotEnv

# Instantiate the PendulumEnv class
env = AcrobotEnv(render_mode="human")



# Reset the environment to get the initial observation and info
observation, info = env.reset()


# Perform random actions for 100 steps
for _ in range(100):
    # For the Pendulum environment, the action space is Box(-2.0, 2.0, (1,), float32)
    # This means we should be sending a one-dimensional float array with one element as an action.
    # Here we sample a random action from the action space of the environment.
    action = env.action_space.sample()  # Replace with your own logic if necessary

    # Perform the action in the environment
    observation, reward, terminated, truncated, info = env.step(0)
    

    env.render()

    # If the episode is terminated, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment
env.close()