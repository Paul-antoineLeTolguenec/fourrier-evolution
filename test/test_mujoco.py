import mujoco_py
import os
import numpy as np
import matplotlib.pyplot as plt
import gym 
from

# Créer un environnement
env = gym.make('HalfCheetah-v2')
s=env.reset()
d=False
while not d:
    s,r,d,_=env.step(env.action_space.sample())
    env.render()