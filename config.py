# Environment Parameters
N = 7  # number of cells (BSs)
M = 100  # number of BS transmission antennas
K = 10  # number of UEs in a cell
BW = 10e6  # Bandwidth = 10MHz
NF = 7  # Power of noise figure [dBm]
Ns = 10  # Number of sample
min_p = -20  # Minimum transmission power [dBm]
max_p = 23  # Maximum transmission power [dBm]

# Training Parameters
EPISODES = 1000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Model Parameters
STATE_SIZE = 3  # [power, SINR, sum-rate]
ACTION_SIZE = 10  # Number of possible power levels
HIDDEN_LAYERS = [64, 32]  # DQN hidden layer sizes

# Rainbow DQN Parameters
N_STEP = 3
ATOMS = 51
V_MIN = -10
V_MAX = 10

# PPO Parameters
PPO_CLIP = 0.2
PPO_EPOCHS = 10
PPO_ENTROPY_COEF = 0.01 