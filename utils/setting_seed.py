import random
import numpy as np
import gym

def set_all_seeds(seed, env):
    """
    Set the seed for all sources of randomness.
    
    Parameters:
        seed (int): The seed to use.
        env (gym.Env): The gym environment.
        
    Returns:
        None.
    """
    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for the gym environment
    env.seed(seed)

# Example usage:
env = gym.make('CartPole-v1')
set_all_seeds(0, env)
