import numpy as np
from numba import jit, njit, prange
import matplotlib.pyplot as plt
import config as cfg

@jit(nopython=True)
def calculate_SINR(H, V, p, noise_power):
    # Implementation of SINR calculation
    pass

@jit(nopython=True)
def calculate_rate(SINR, bandwidth):
    # Implementation of rate calculation
    pass

def plot_results(rewards, title="Training Rewards"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def save_model(agent, filename):
    """Save model weights for different agent types."""
    if hasattr(agent, 'model'):
        # For DQN, DDQN, RainbowDQN
        agent.model.save_weights(f"{filename}.weights.h5")
    elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
        # For PPO
        agent.actor.save_weights(f"{filename}_actor.weights.h5")
        agent.critic.save_weights(f"{filename}_critic.weights.h5")
    else:
        raise ValueError("Unsupported agent type for saving")

def load_model(agent, filename):
    """Load model weights for different agent types."""
    if hasattr(agent, 'model'):
        # For DQN, DDQN, RainbowDQN
        agent.model.load_weights(f"{filename}.weights.h5")
    elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
        # For PPO
        agent.actor.load_weights(f"{filename}_actor.weights.h5")
        agent.critic.load_weights(f"{filename}_critic.weights.h5")
    else:
        raise ValueError("Unsupported agent type for loading")

def preprocess_state(state):
    # Normalize state values
    normalized_state = np.zeros_like(state)
    normalized_state[0] = (state[0] - cfg.min_p) / (cfg.max_p - cfg.min_p)
    normalized_state[1] = state[1] / (1 + state[1])  # Normalize SINR
    normalized_state[2] = state[2] / (1 + state[2])  # Normalize rate
    return normalized_state

def postprocess_action(action):
    # Convert discrete action to power level
    return cfg.min_p + action * (cfg.max_p - cfg.min_p) / (cfg.ACTION_SIZE - 1)

def calculate_reward(rate, power):
    # Implementation of reward calculation
    # Can be modified based on specific requirements
    return rate - 0.1 * power  # Example: balance between rate and power consumption 