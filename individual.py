import gym
import numpy as np
import ray 

@ray.remote
class Individual:
    def __init__(self, env, parameters, seed=0):
        self.env = env
        self.seed = seed
        self.state_dim = env.observation_space.shape[0]  # State space dimension
        self.action_dim = env.action_space.shape[0]  # Action space dimension
        self.parameters = parameters  # The list of parameters

        # Calculate the number of parameters for the controller weight matrix
        self.controller_params = self.state_dim * self.action_dim

        # Ensure parameters array has at least as many elements as needed for controller
        assert len(parameters) >= self.controller_params, f"Parameters array too small: {len(parameters)} < {self.controller_params}"

        # The remaining parameters are used for the signals
        signal_parameters = parameters[self.controller_params:]
        self.params_per_signal = len(signal_parameters) // (self.state_dim * 3)  # Each signal has three parameters (A, F, P)

        # Ensure there are enough signal parameters for each state dimension
        assert self.params_per_signal >= 1, "Not enough signal parameters for each state dimension"

        # set genome
        self.set_genome(parameters)

    def signal(self, parameters):
        """
        Create a function that represents a signal by summing multiple sinusoidal components.
    
        Parameters:
            parameters: numpy array, where each row contains three values - amplitude (A), frequency (F), and phase (P).
        
        Returns:
            A function that represents the generated signal.
        """
        def generated_signal(t):
            signal = 0
            for A, F, P in parameters:
                component = A * np.sin(2*np.pi*F*t + P)
                signal += component
            return signal

        return generated_signal
    
    def set_genome(self, parameters):
        """
        Set the parameters for the signals and the controller.
        
        Parameters:
            parameters: numpy array of parameters for all signals and the controller.
            
        Returns:
            parameters: numpy array of parameters for all signals and the controller.
        """
        # Ensure the number of parameters is correct
        # assert len(parameters) == self.state_dim * (self.params_per_signal * 3 + self.action_dim), f"Parameters array size mismatch: {len(parameters)} != {self.state_dim * (self.params_per_signal * 3 + self.action_dim)}"

        # Set the parameters for each signal
        signal_parameters = parameters[self.controller_params:]
        self.set_signals(signal_parameters)

        # Set the parameters for the controller
        self.controller_weights = np.array(parameters[:self.controller_params]).reshape((self.state_dim, self.action_dim))
        return parameters
    
    def set_signals(self, signal_parameters):
        """
        Set the parameters for the signals.
        
        Parameters:
            signal_parameters: numpy array of parameters for all signals.
            
        Returns:
            None.
        """
        # Ensure the number of parameters is correct
        # assert len(parameters) == self.state_dim * self.params_per_signal * 3, f"Parameters array size mismatch: {len(parameters)} != {self.state_dim * self.params_per_signal * 3}"

        # Set the parameters for each signal
        signals = []
        for i in range(self.state_dim):
            # Extract the parameters for each signal
            start = i * self.params_per_signal * 3
            end = start + self.params_per_signal * 3
            params = np.array(signal_parameters[start:end]).reshape((self.params_per_signal, 3))
            signals.append(self.signal(params))
            self.signals=signals
    def set_controller_weights(self, controller_weights):
        """
        Set the weights for the controller.
        
        Parameters:
            controller_weights: numpy array of weights for the controller.
            
        Returns:
            None.
        """
        # Ensure the number of parameters is correct
        assert len(controller_weights) == self.state_dim * self.action_dim, f"Controller weights array size mismatch: {len(controller_weights)} != {self.state_dim * self.action_dim}"
        self.controller_weights = np.array(controller_weights).reshape((self.state_dim, self.action_dim))

    def forward(self, state):
        """
        Passes the state through the signals to get the Fourier encoding, then uses the controller to get the action.
        
        Parameters:
            state: numpy array representing the state.
            
        Returns:
            The calculated action.
        """
        # Ensure the state dimension matches the number of signals
        assert len(state) == self.state_dim, f"State dimension mismatch: {len(state)} != {self.state_dim}"
        # Get the Fourier encoding of the state
        fourier_encoding = np.array([signal(s) for signal, s in zip(self.signals, state)])
        
        # Use the controller to get the action
        action = np.dot(fourier_encoding, self.controller_weights)

        return action
    
    def evaluate(self):
        """
        Run one episode with the given individual and return the total reward.
            
        Returns:
            The total reward from the episode.
        """
        self.env.seed(self.seed)
        state = self.env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward

if __name__ == "__main__":
    # Example of use
    env = gym.make('HalfCheetah-v2')
    parameters = np.random.rand(200).tolist()  # Random list of parameters
    # individual = Individual.remote(env, parameters)  # Create the individual
    # reward = ray.get(individual.evaluate.remote())  # Evaluate the individual
    individual=Individual(env,parameters)
    reward=individual.evaluate()
