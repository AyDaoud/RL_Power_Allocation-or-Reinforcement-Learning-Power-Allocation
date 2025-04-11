# Reinforcement Learning for Power Allocation in Multi-User Wireless Systems



## 🚀 Introduction

This project explores the application of various Deep Reinforcement Learning (DRL) algorithms to optimize power allocation strategies in a simulated multi-user, multi-cell Massive MIMO (Multiple-Input Multiple-Output) wireless communication environment. The primary goal is to train RL agents that can dynamically adjust power levels for different users to maximize the overall network throughput (sum-rate) while considering interference between users and cells.

The project includes:
*   A custom OpenAI Gym-compatible environment (`M_MIMOEnv`) simulating the wireless network dynamics.
*   Implementations of several DRL algorithms:
    *   Deep Q-Network (DQN)
    *   Double Deep Q-Network (DDQN)
    *   Rainbow DQN (incorporating Dueling DQN, Prioritized Replay, Multi-step Learning, Noisy Nets)
    *   Proximal Policy Optimization (PPO) for discrete action spaces.
*   Baseline power allocation algorithms for performance comparison.
*   A graphical user interface (GUI) built with PyQt to visualize the simulation process, including antenna/user placement, power allocation, and SINR levels in (near) real-time.

## ✨ Key Features

*   **Multiple RL Algorithms:** Compare DQN, DDQN, Rainbow DQN, and PPO side-by-side.
*   **Realistic Simulation:** Environment models multi-cell interference, channel fading, path loss, and Zero-Forcing (ZF) precoding.
*   **Baseline Comparisons:** Evaluate RL performance against standard power allocation schemes (Equal Power, Random Power, Maximum Power, Max-Product).
*   **Interactive GUI:** Visualize the complex interactions within the wireless network, observe how different algorithms allocate power, and monitor performance metrics dynamically.
*   **Modular Code:** Environment, agents, training loops, and GUI separated into distinct Python files (ideally).

## 🛠️ Installation & Setup

1.  **Prerequisites:**
    *   Python (>= 3.8 recommended, **Note:** The original notebook used TF 2.8.0/Keras 2.8.0, which might require Python <= 3.10)
    *   Git
    *   `pip` or `conda` for package management

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/AyDaoud/RL_Power_Allocation-or-Reinforcement-Learning-Power-Allocation.git
    cd RL_Power_Allocation-or-Reinforcement-Learning-Power-Allocation
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Dependencies:**
    *   **IMPORTANT:** First, create a `requirements.txt` file from your working environment using `pip freeze > requirements.txt` inside your activated `venv` and commit it to the repository.
    *   Then, install the requirements:
    ```bash
    pip install -r requirements.txt
    # Or, install manually (adjust versions based on your final setup):
    # pip install gym numpy scipy numba tensorflow==2.8.0 keras==2.8.0 keras-rl2 matplotlib pandas pyqt5
    # pip install protobuf==3.20.*
    ```
    *   **TensorFlow/Keras Note:** This project might rely on TensorFlow 1.x compatibility features via `tf.compat.v1` and specific older versions (TF 2.8.0, Keras 2.8.0) due to `keras-rl2`. Using newer versions might require code modifications in `agents.py`.

## ⚙️ Usage

### 1. Training the Agents (Optional - if not using pre-trained)

*   If you have separated the training loops into `train.py`:
    ```bash
    # Example for training DQN
    python train.py --algo DQN --N 7 --M 32 --K 10 --episodes 500 --output_model saved_models/dqn_model.h5

    # Example for training PPO (may need separate actor/critic outputs)
    python train.py --algo PPO --N 7 --M 32 --K 10 --episodes 500 --output_actor saved_models/ppo_actor.h5 --output_critic saved_models/ppo_critic.h5
    ```
    *(Adjust arguments based on your actual `train.py` script)*
*   If training within the notebook (`RL_project.ipynb`), run the relevant cells and ensure the models (`.h5` files) are saved.
*   **Note:** Model files (`.h5`) are typically large and are excluded by the `.gitignore` in this template. You need to train them locally or provide download links if sharing pre-trained models.

### 2. Running the GUI for Visualization & Evaluation

*   Launch the GUI application:
    ```bash
    python simulation_gui.py
    ```
*   **Inside the GUI:**
    1.  Set the simulation parameters (N, M, K).
    2.  Select the desired algorithm (DQN, DDQN, Rainbow, PPO, or a baseline) from the dropdown.
    3.  If an RL algorithm is chosen, click "Load Model" and select the corresponding pre-trained `.h5` weight file(s).
    4.  Click "Start" to run the simulation for one or more episodes (as configured in the GUI code).
    5.  Observe the visualizations update.
    6.  Click "Stop" to halt the simulation.

## 📊 Visualization & Results

The GUI provides visualizations of:

*   **Network Layout:** Shows the positions of Base Stations (BSs) and User Equipments (UEs). 
*   **Power Allocation:** Dynamically displays the power assigned to different users/cells by the selected algorithm.
*   **SINR Levels:** Visualizes the Signal-to-Interference-plus-Noise Ratio for users, indicating link quality.
*   **Performance Plot:** Shows the evolution of the sum-rate or cumulative reward during the simulation episode.

**Animated Simulation:**

**Performance Comparison:**

