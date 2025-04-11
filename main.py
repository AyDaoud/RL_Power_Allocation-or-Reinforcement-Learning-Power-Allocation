import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Enable eager execution

from environment import M_MIMOEnv
from models import DQNAgent, DDQNAgent, RainbowDQNAgent, PPOAgent
from utils import plot_results, save_model, load_model, preprocess_state, postprocess_action
from visualization import show_visualization_menu
import config as cfg
from collections import deque

def train_agent(env, agent, episodes, test_mode=False, visualize=False):
    rewards = []
    best_reward = float('-inf')
    
    if visualize:
        visualizer = SystemVisualizer(env)
    
    for episode in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.reshape(state, [1, cfg.STATE_SIZE])
        total_reward = 0
        done = False
        
        while not done:
            if isinstance(agent, PPOAgent):
                action, old_probs = agent.select_action(state)
            else:
                action = agent.get_action(state)
            
            next_state, reward, done, info, dl_rate = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, cfg.STATE_SIZE])
            
            if isinstance(agent, PPOAgent):
                agent.append_sample(state, action, reward, next_state, done, old_probs)
            else:
                agent.append_sample(state, action, reward, next_state, done)
            
            if not test_mode:
                agent.train_model()
            
            state = next_state
            total_reward += reward
            
            if visualize:
                visualizer.update_plots(env.action_values, env.sinr.flatten(), rewards)
            
            if done:
                break
        
        rewards.append(total_reward)
        avg_reward = np.mean(rewards[-100:])
        
        if episode % 10 == 0:
            print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Average Reward: {avg_reward:.2f}")
        
        if total_reward > best_reward and not test_mode:
            best_reward = total_reward
            save_model(agent, f"{agent.__class__.__name__}_best")
    
    if visualize:
        plt.show()
    
    return rewards

def evaluate_agent(env, agent, episodes=10):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        state = np.reshape(state, [1, cfg.STATE_SIZE])
        total_reward = 0
        done = False
        
        while not done:
            if isinstance(agent, PPOAgent):
                action, _ = agent.select_action(state)
            else:
                action = agent.get_action(state)
            
            next_state, reward, done, info, dl_rate = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, cfg.STATE_SIZE])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
        print(f"Evaluation Episode {episode+1}: Reward = {total_reward:.2f}")
    
    return np.mean(rewards)

def compare_power_allocation_methods(env):
    print("\nComparing Power Allocation Methods:")
    print(f"Equal Power Allocation: {env.equal_PA():.2f} bits/s")
    print(f"Random Power Allocation: {env.random_PA():.2f} bits/s")
    print(f"Maximum Power Allocation: {env.maximum_PA():.2f} bits/s")
    print(f"MaxProb Power Allocation: {env.maxprob_PA():.2f} bits/s")

def main():
    # Initialize environment
    env = M_MIMOEnv(N=7, M=100, K=10)
    
    # Show visualization menu
    show_visualization_menu(env)

if __name__ == "__main__":
    main() 