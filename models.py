import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Enable eager execution
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda, Add, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import config as cfg
from collections import deque
import random
import sys
import gym
import math
import time
import pylab
import cmath
import random
import itertools
import threading
import tensorflow
import scipy as sp
import numpy as np
import pandas as pd
from gym import Env
import scipy.io as sc
from sys import version
from absl import logging
from numpy import ndarray
from scipy import special
from gym.utils import seeding
from scipy.constants import *
from scipy.special import erfinv
from scipy.integrate import quad
from scipy.linalg import toeplitz
from numba import jit, njit, prange
from gym.spaces import Discrete, Box
from collections import deque, Counter
from mpl_toolkits.mplot3d import Axes3D
from gym import Env, error, spaces, utils
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization

class DQNAgent:
    def __init__(self, state_size, action_size, test_mode):
        # Environment-specific variables
        self.state_size = state_size
        self.action_size = action_size
        self.test_mode = test_mode

        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 256
        self.train_start = 1000
        self.reward_clipping = 10

        # Replay memory
        self.memory = deque(maxlen=50000)

        # Model initialization
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        # Initialize target model with main model's weights
        self.update_target_model()

        # Load pretrained model if in test mode
        if self.test_mode:
            try:
                self.model.load_weights("./M_MIMO_DQN.h5")
                self.epsilon = 0
            except:
                print("No pretrained model found, starting from scratch")

    def build_model(self):
        # Build the neural network
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        # Epsilon-greedy policy for exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        # Clip rewards to stabilize training
        reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)
        self.memory.append((state, action, reward, next_state, done))
        if not self.test_mode and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        # Sample a mini-batch from memory
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # Predict Q-values for the current and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            # Update the target value
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.max(target_next[i])

        # Train the main model
        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

class DDQNAgent:
    def __init__(self, state_size, action_size, test_mode):
        # Environment-specific variables
        self.state_size = state_size
        self.action_size = action_size
        self.test_mode = test_mode

        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 256
        self.train_start = 1000
        self.reward_clipping = 10

        # Replay memory
        self.memory = deque(maxlen=50000)

        # Model initialization
        self.model = self.build_model()  # Main Q-network
        self.target_model = self.build_model()  # Target Q-network
        self.update_target_model()  # Initialize target model with main model's weights

        # Load pretrained model if in test mode
        if self.test_mode:
            self.model.load_weights("./M_MIMO_DDQN.h5")
            self.epsilon=0
            # self.epsilon_decay=0

    def build_model(self):
        # Build the neural network
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))  # Prevent overfitting
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        # Epsilon-greedy policy for exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        # Clip rewards to stabilize training
        reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)
        self.memory.append((state, action, reward, next_state, done))
        if not self.test_mode and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        # Sample a mini-batch from memory
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # Predict Q-values for the current and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)  # Main Q-network for action selection
        target_val = self.target_model.predict(next_states, verbose=0)  # Target Q-network for value estimation

        for i in range(batch_size):
            # Double DQN update
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Use the main model to select the best action
                best_action = np.argmax(target_next[i])
                # Use the target model to estimate the value of the best action
                target[i][actions[i]] = rewards[i] + self.discount_factor * target_val[i][best_action]

        # Train the main model
        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)



class RainbowDQNAgent:
    def __init__(self, state_size, action_size, test_mode):
        # Environment-specific variables
        self.state_size = state_size
        self.action_size = action_size
        self.test_mode = test_mode

        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 256
        self.train_start = 1000
        self.reward_clipping = 10
        self.n_steps = 3  # Multi-step learning

        # Replay memory with priorities
        self.memory = deque(maxlen=50000)
        self.priority = deque(maxlen=50000)
        self.alpha = 0.6  # Prioritization exponent

        # Model initialization
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if self.test_mode:
            self.model.load_weights("M_MIMO_RainbowDQN.h5")
            self.epsilon=0
            self.epsilon_decay=0

    def build_model(self):
      """
      Build a dueling network with noisy layers.
      """
      input_layer = Input(shape=(self.state_size,))

     # Feature extraction with NoisyDense
      x = NoisyDense(128, sigma_zero=0.5)(input_layer)
      x = tf.keras.layers.ReLU()(x)
      x = NoisyDense(128, sigma_zero=0.5)(x)
      x = tf.keras.layers.ReLU()(x)

      # Dueling architecture
      # State value
      value = NoisyDense(1, sigma_zero=0.5)(x)

      # Advantage values
      advantage = NoisyDense(self.action_size, sigma_zero=0.5)(x)
      advantage_mean = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
      advantage = Add()([advantage, -advantage_mean])

      # Combine state and advantage
      q_values = Add()([value, advantage])

      model = Model(inputs=input_layer, outputs=q_values)
      model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
      return model

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """Epsilon-greedy policy for exploration vs exploitation."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        """Add transition to prioritized replay buffer."""
        reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)
        self.memory.append((state, action, reward, next_state, done))
        # Default priority
        self.priority.append(max(self.priority) if self.priority else 1.0)

        if not self.test_mode and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sample_minibatch(self):
        """Sample a minibatch using prioritized experience replay."""
        priorities = np.array(self.priority) ** self.alpha
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[i] for i in indices]
        return batch, indices
    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        # Sample a mini-batch from prioritized replay memory
        batch, indices = self.sample_minibatch()

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = batch[i][0]
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_states[i] = batch[i][3]
            dones.append(batch[i][4])

        # Predict Q-values for the current and next states
        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            # Multi-step reward calculation
            n_step_reward = 0
            for k in range(self.n_steps):
                if i + k < len(batch):  # Avoid going out of bounds
                    n_step_reward += (self.discount_factor ** k) * batch[i + k][2]
                    if batch[i + k][4]:  # Stop if a terminal state is reached
                        break

            # Double DQN: select the best action from the main model
            best_action = np.argmax(target_next[i])
            if dones[i]:
                target[i][actions[i]] = n_step_reward  # No future reward after terminal state
            else:
                target[i][actions[i]] = n_step_reward + (self.discount_factor ** self.n_steps) * target_val[i][best_action]

        # Train the main model
        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)

        # Update priorities using the absolute TD error
        for i, idx in enumerate(indices):
            td_error = abs(target[i][actions[i]] - self.model.predict(states[i].reshape(1, -1), verbose=0)[0][actions[i]])
            self.priority[idx] = td_error + 1e-6  # Add small constant to avoid zero priority




class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, lam=0.2, clip_value=0.2,test_mode = False):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_value = clip_value

        # Build models
        self.actor = self.build_actor(lr_actor)
        self.critic = self.build_critic(lr_critic)

        if test_mode:
            self.actor.load_weights("PPO_Actor.h5")
            self.critic.load_weights("PPO_Critic.h5")

    def build_actor(self, lr):
        # Actor network
        state_input = Input(shape=(self.state_size,))
        x = Dense(256, activation="relu")(state_input)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        action_probs = Dense(self.action_size, activation="softmax")(x)

        model = Model(inputs=state_input, outputs=action_probs)
        return model

    def build_critic(self, lr):
        # Critic network
        state_input = Input(shape=(self.state_size,))
        x = Dense(256, activation="relu")(state_input)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        value = Dense(1, activation="linear")(x)

        model = Model(inputs=state_input, outputs=value)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return model

    def select_action(self, state):
        state = state.reshape([1, self.state_size])
        prob = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=prob)
        return action, prob

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (1 - dones[i]) * next_value - values[i]
            gae = delta + self.gamma * (1 - dones[i]) * self.lam * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return np.array(advantages)

    def ppo_loss(self, old_probs, advantages):
        # Define the PPO clipped loss function
        def loss(y_true, y_pred):
            prob_ratio = tf.reduce_sum(y_pred * y_true, axis=1) / tf.reduce_sum(y_true * old_probs, axis=1)
            clipped_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_value, 1 + self.clip_value)
            return -tf.reduce_mean(tf.minimum(prob_ratio * advantages, clipped_ratio * advantages))
        return loss

    def train(self, states, actions, rewards, next_states, dones, old_probs):
        values = self.critic.predict(states, verbose=0).flatten()  # Ensure values are 1D
        next_value = self.critic.predict(next_states[-1].reshape(1, -1), verbose=0)[0][0]

        # Compute advantages and discounted rewards
        advantages = self.compute_advantages(rewards, values, dones, next_value)
        discounted_rewards = advantages + values

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Train critic
        self.critic.fit(states, discounted_rewards.reshape(-1, 1), epochs=1, verbose=0, shuffle=True)

        # Train actor with PPO clipped loss
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.action_size)
        old_probs = np.array(old_probs)

        # Compile the actor with the PPO loss
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss=self.ppo_loss(old_probs, advantages))
        self.actor.fit(states, actions_one_hot, epochs=1, verbose=0, shuffle=True)



class NoisyDense(tf.keras.layers.Layer):
    """
    Noisy Dense Layer using independent Gaussian noise,
    as defined by Fortunato et al. (2017).
    """

    def __init__(self, units, sigma_zero=0.5, use_bias=True, **kwargs):
        """
        Args:
            units (int): Number of output features (neurons).
            sigma_zero (float): Initial noise scaling factor.
            use_bias (bool): Whether to include bias.
        """
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_zero = sigma_zero
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        sigma_init = self.sigma_zero / np.sqrt(input_dim)

        # Weight and bias parameters
        self.kernel_mu = self.add_weight(
            shape=(input_dim, self.units),
            initializer="he_uniform",
            trainable=True,
            name="kernel_mu"
        )
        self.kernel_sigma = self.add_weight(
            shape=(input_dim, self.units),
            initializer=tf.constant_initializer(sigma_init),
            trainable=True,
            name="kernel_sigma"
        )

        if self.use_bias:
            self.bias_mu = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias_mu"
            )
            self.bias_sigma = self.add_weight(
                shape=(self.units,),
                initializer=tf.constant_initializer(sigma_init),
                trainable=True,
                name="bias_sigma"
            )
        else:
            self.bias_mu = None
            self.bias_sigma = None

    def _scale_noise(self, x):
        """
        Scales Gaussian noise using the function f(x) = sign(x) * sqrt(|x|).
        """
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def _generate_noise(self, shape):
        """
        Generates factorized Gaussian noise for weights and biases.
        """
        row_noise = self._scale_noise(tf.random.normal((shape[0], 1)))
        col_noise = self._scale_noise(tf.random.normal((1, shape[1])))
        return tf.matmul(row_noise, col_noise)

    def call(self, inputs):
        # Generate Gaussian noise for weights
        weight_noise = self._generate_noise(self.kernel_mu.shape)
        noisy_kernel = self.kernel_mu + self.kernel_sigma * weight_noise

        # Compute output
        output = tf.matmul(inputs, noisy_kernel)

        # Add bias with noise if applicable
        if self.use_bias:
            bias_noise = self._scale_noise(tf.random.normal(self.bias_mu.shape))
            noisy_bias = self.bias_mu + self.bias_sigma * bias_noise
            output += noisy_bias

        return output