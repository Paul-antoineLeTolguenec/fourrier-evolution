import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import numpy as np
from cmaes import CMA
from individual import Individual
from utils.wandb_server import WandbServer
from utils.setting_seed import set_all_seeds
import gym
import ray 
import wandb

class CMATRAINER:
    def __init__(self, env, genome_dim, population_size, sigma, epochs, seed=0):
        self.seed=seed
        self.env = env
        self.genome_dim = genome_dim
        self.population_size = population_size
        self.individuals = [Individual.remote(env, np.random.randn(genome_dim), seed) for _ in range(population_size)]
        self.sigma = sigma
        self.epochs = epochs
        # set seed
        set_all_seeds(self.seed, self.env)
        # Initialize the optimizer
        self.es = CMA(mean=np.zeros(genome_dim), sigma=self.sigma, population_size=self.population_size)

        # Initialize Weights & Biases
        wandb_config = { "genome_dim": self.genome_dim, "population_size": self.population_size, 
                        "sigma": self.sigma, "epochs": self.epochs , "env": self.env.spec.id}
        self.name = f"{self.env.spec.id}_cma-es"
        self.wandb_server = WandbServer(project_name="FourrierEvolution", name=self.name, config=wandb_config,group='cma-es')


    def train(self):
        """
        Run the CMA-ES optimization for a number of epochs.
        
        Returns:
            None.
        """
        for epoch in range(self.epochs):
            # Set individual genomes 
            genomes=ray.get([individual.set_genome.remote(self.es.ask()) for i, individual in enumerate(self.individuals)])
            # Evaluate the individuals
            rewards = ray.get([individual.evaluate.remote() for individual in self.individuals])
            # solutions
            solutions = [(genome, -reward) for genome, reward in zip(genomes, rewards)]
            # Update the optimizer
            self.es.tell(solutions)

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "max_reward": max(rewards),
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards)
            })

        # Log the best genome found
        # wandb.log({"best_genome": self.es.result.xbest})

# Example of use
env = gym.make('HalfCheetah-v2')
cma_optimizer = CMATRAINER(env, genome_dim=300, population_size=5, sigma=0.5, epochs=10)
cma_optimizer.train()
