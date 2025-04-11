import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.colors as mcolors
from environment import M_MIMOEnv
from models import DQNAgent, DDQNAgent, RainbowDQNAgent, PPOAgent
import config as cfg

class SystemVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = gridspec.GridSpec(3, 2, figure=self.fig)
        self.setup_plots()
        
        # Add event handler for window closing
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
    def on_close(self, event):
        plt.close(self.fig)
    
    def setup_plots(self):
        # System Parameters Plot
        self.ax_params = self.fig.add_subplot(self.gs[0, 0])
        self.ax_params.set_title('System Parameters')
        self.ax_params.axis('off')
        
        # Interference Pattern Plot
        self.ax_interf = self.fig.add_subplot(self.gs[0, 1])
        self.ax_interf.set_title('Interference Pattern')
        
        # Power Allocation Plot
        self.ax_power = self.fig.add_subplot(self.gs[1, 0])
        self.ax_power.set_title('Power Allocation')
        
        # SINR Distribution Plot
        self.ax_sinr = self.fig.add_subplot(self.gs[1, 1])
        self.ax_sinr.set_title('SINR Distribution')
        
        # Algorithm Performance Plot
        self.ax_perf = self.fig.add_subplot(self.gs[2, :])
        self.ax_perf.set_title('Algorithm Performance')
        
        plt.tight_layout()
    
    def update_system_params(self):
        params_text = f"""
        Number of Cells (N): {self.env.N}
        Number of Antennas (M): {self.env.M}
        Number of UEs per Cell (K): {self.env.K}
        Bandwidth: {self.env.BW/1e6:.1f} MHz
        Noise Figure: {self.env.NF} dBm
        Min Power: {self.env.min_p} dBm
        Max Power: {self.env.max_p} dBm
        """
        self.ax_params.clear()
        self.ax_params.text(0.1, 0.5, params_text, fontsize=10)
        self.ax_params.axis('off')
    
    def plot_interference(self):
        # Plot interference pattern between cells
        self.ax_interf.clear()
        # Add your interference visualization code here
        self.ax_interf.set_xlabel('Distance (m)')
        self.ax_interf.set_ylabel('Interference Level (dB)')
    
    def plot_power_allocation(self, power_levels):
        self.ax_power.clear()
        self.ax_power.bar(range(len(power_levels)), power_levels)
        self.ax_power.set_xlabel('UE Index')
        self.ax_power.set_ylabel('Power (dBm)')
    
    def plot_sinr_distribution(self, sinr_values):
        self.ax_sinr.clear()
        self.ax_sinr.hist(sinr_values, bins=20)
        self.ax_sinr.set_xlabel('SINR (dB)')
        self.ax_sinr.set_ylabel('Count')
    
    def plot_algorithm_performance(self, rewards_history):
        self.ax_perf.clear()
        self.ax_perf.plot(rewards_history)
        self.ax_perf.set_xlabel('Episode')
        self.ax_perf.set_ylabel('Total Reward')
    
    def update_plots(self, power_levels, sinr_values, rewards_history):
        try:
            self.update_system_params()
            self.plot_interference()
            self.plot_power_allocation(power_levels)
            self.plot_sinr_distribution(sinr_values)
            self.plot_algorithm_performance(rewards_history)
            plt.draw()
            plt.pause(0.1)
        except Exception as e:
            print(f"Visualization error: {e}")

class AlgorithmComparison:
    def __init__(self, env):
        self.env = env
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_comparison_plots()
        
        # Add event handler for window closing
        self.fig.canvas.mpl_connect('close_event', self.on_close)
    
    def on_close(self, event):
        plt.close(self.fig)
    
    def setup_comparison_plots(self):
        # Create subplots for different algorithms
        self.ax_rewards = self.fig.add_subplot(221)
        self.ax_rewards.set_title('Reward Comparison')
        
        self.ax_power = self.fig.add_subplot(222)
        self.ax_power.set_title('Power Allocation Comparison')
        
        self.ax_sinr = self.fig.add_subplot(223)
        self.ax_sinr.set_title('SINR Distribution Comparison')
        
        self.ax_convergence = self.fig.add_subplot(224)
        self.ax_convergence.set_title('Convergence Rate')
        
        plt.tight_layout()
    
    def plot_comparison(self, results):
        # Plot comparison of different algorithms
        algorithms = list(results.keys())
        
        # Plot rewards
        self.ax_rewards.clear()
        for algo in algorithms:
            self.ax_rewards.plot(results[algo]['rewards'], label=algo)
        self.ax_rewards.legend()
        self.ax_rewards.set_xlabel('Episode')
        self.ax_rewards.set_ylabel('Reward')
        
        # Plot power allocation
        self.ax_power.clear()
        for algo in algorithms:
            self.ax_power.plot(results[algo]['power_levels'], label=algo)
        self.ax_power.legend()
        self.ax_power.set_xlabel('UE Index')
        self.ax_power.set_ylabel('Power (dBm)')
        
        # Plot SINR distribution
        self.ax_sinr.clear()
        for algo in algorithms:
            self.ax_sinr.hist(results[algo]['sinr_values'], bins=20, alpha=0.5, label=algo)
        self.ax_sinr.legend()
        self.ax_sinr.set_xlabel('SINR (dB)')
        self.ax_sinr.set_ylabel('Count')
        
        # Plot convergence
        self.ax_convergence.clear()
        for algo in algorithms:
            convergence = np.cumsum(results[algo]['rewards']) / np.arange(1, len(results[algo]['rewards']) + 1)
            self.ax_convergence.plot(convergence, label=algo)
        self.ax_convergence.legend()
        self.ax_convergence.set_xlabel('Episode')
        self.ax_convergence.set_ylabel('Average Reward')
        
        plt.draw()
        plt.pause(0.1)

def visualize_training(env, agent, episodes):
    try:
        visualizer = SystemVisualizer(env)
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, -1])  # Ensure proper state shape
            total_reward = 0
            done = False
            
            while not done:
                if isinstance(agent, PPOAgent):
                    action, probs = agent.select_action(state)
                else:
                    action = agent.get_action(state)
                
                next_state, reward, done, info, dl_rate = env.step(action)
                next_state = np.reshape(next_state, [1, -1])
                
                # Training step
                if isinstance(agent, PPOAgent):
                    agent.train(np.array([state]), np.array([action]), 
                              np.array([reward]), np.array([next_state]), 
                              np.array([done]), np.array([probs]))
                else:
                    agent.append_sample(state, action, reward, next_state, done)
                    agent.train_model()
                
                # Update visualization
                power_levels = env.action_values
                sinr_values = env.sinr.flatten()
                rewards_history.append(reward)
                
                visualizer.update_plots(power_levels, sinr_values, rewards_history)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
        
        plt.show()
    except Exception as e:
        print(f"Training visualization error: {e}")
    finally:
        if plt.get_fignums():  # If there are still open figures
            plt.close('all')

def compare_algorithms(env, episodes):
    algorithms = {
        'DQN': DQNAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, False),
        'DDQN': DDQNAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, False),
        'RainbowDQN': RainbowDQNAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, False),
        'PPO': PPOAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, test_mode=False)
    }
    
    results = {}
    comparison = AlgorithmComparison(env)
    
    for name, agent in algorithms.items():
        print(f"Training {name}...")
        rewards = []
        power_levels = []
        sinr_values = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Handle both PPO and other agents
                if isinstance(agent, PPOAgent):
                    action, probs = agent.select_action(state)
                else:
                    action = agent.get_action(state)
                
                next_state, reward, done, info, dl_rate = env.step(action)
                
                # Training step
                if isinstance(agent, PPOAgent):
                    agent.train(np.array([state]), np.array([action]), 
                              np.array([reward]), np.array([next_state]), 
                              np.array([done]), np.array([probs]))
                else:
                    agent.append_sample(state, action, reward, next_state, done)
                    agent.train_model()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
            power_levels.append(env.action_values)
            sinr_values.append(env.sinr.flatten())
            
            # Update comparison plot every 10 episodes
            if episode % 10 == 0:
                print(f"{name} Episode {episode}: Total Reward = {total_reward:.2f}")
                results[name] = {
                    'rewards': rewards,
                    'power_levels': power_levels[-1],
                    'sinr_values': sinr_values[-1]
                }
                comparison.plot_comparison(results)
    
    plt.show()

class PhysicalSystemVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig = plt.figure(figsize=(15, 10))
        self.setup_plots()
        self.animation = None
        self.cell_colors = plt.cm.tab10(np.linspace(0, 1, env.N))
        
        # Add event handler for window closing
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
    def on_close(self, event):
        # Cleanup animation when window is closed
        if self.animation is not None:
            self.animation.event_source.stop()
            del self.animation
        plt.close(self.fig)
        
    def setup_plots(self):
        gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[3, 1])
        
        # Physical Layout Plot (Cells, BS, UEs)
        self.ax_layout = self.fig.add_subplot(gs[0, :])
        self.ax_layout.set_title('Physical System Layout')
        self.ax_layout.set_xlabel('Distance (m)')
        self.ax_layout.set_ylabel('Distance (m)')
        
        # Signal Strength Plot
        self.ax_signal = self.fig.add_subplot(gs[1, 0])
        self.ax_signal.set_title('Signal Strength')
        
        # Interference Plot
        self.ax_interference = self.fig.add_subplot(gs[1, 1])
        self.ax_interference.set_title('Interference Pattern')
        
        plt.tight_layout()
        
    def init_animation(self):
        # Initialize cell positions (hexagonal grid)
        self.cell_positions = self._generate_cell_positions()
        self.bs_positions = self._generate_bs_positions()
        self.ue_positions = self._generate_ue_positions()
        
        # Clear previous plots
        self.ax_layout.clear()
        self.ax_signal.clear()
        self.ax_interference.clear()
        
        # Set layout bounds
        self.ax_layout.set_xlim(-300, 300)
        self.ax_layout.set_ylim(-300, 300)
        
        return []
        
    def _generate_cell_positions(self):
        # Generate hexagonal cell positions
        positions = []
        cell_radius = 250 / np.sqrt(self.env.N)  # Adjust based on number of cells
        
        if self.env.N == 7:  # Special case for 7 cells
            # Center cell
            positions.append((0, 0))
            # Surrounding cells
            for i in range(6):
                angle = i * np.pi / 3
                x = cell_radius * 1.5 * np.cos(angle)
                y = cell_radius * 1.5 * np.sin(angle)
                positions.append((x, y))
        else:
            # Generate grid-based positions for other configurations
            grid_size = int(np.ceil(np.sqrt(self.env.N)))
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) < self.env.N:
                        x = (i - grid_size/2) * cell_radius * 1.5
                        y = (j - grid_size/2) * cell_radius * 1.5
                        positions.append((x, y))
        
        return np.array(positions)
    
    def _generate_bs_positions(self):
        # Base stations are at cell centers
        return self.cell_positions.copy()
    
    def _generate_ue_positions(self):
        # Generate UE positions within each cell
        ue_positions = []
        for cell_pos in self.cell_positions:
            for _ in range(self.env.K):
                # Random position within cell radius
                radius = np.random.uniform(25, 250/np.sqrt(self.env.N))
                angle = np.random.uniform(0, 2*np.pi)
                x = cell_pos[0] + radius * np.cos(angle)
                y = cell_pos[1] + radius * np.sin(angle)
                ue_positions.append((x, y))
        return np.array(ue_positions)
    
    def animate(self, frame):
        self.ax_layout.clear()
        self.ax_signal.clear()
        self.ax_interference.clear()
        
        # Draw cells
        for i, (x, y) in enumerate(self.cell_positions):
            cell = Circle((x, y), 250/np.sqrt(self.env.N), 
                        fill=False, color=self.cell_colors[i], linestyle='--')
            self.ax_layout.add_patch(cell)
            
            # Draw BS (triangle)
            bs_marker = self.ax_layout.plot(x, y, '^', 
                                          color=self.cell_colors[i], 
                                          markersize=10, 
                                          label=f'BS {i+1}')
            
            # Draw antennas
            antenna_spread = 10  # meters
            for m in range(min(8, self.env.M)):  # Show max 8 antennas for clarity
                ant_x = x + antenna_spread * np.cos(m * 2*np.pi/8)
                ant_y = y + antenna_spread * np.sin(m * 2*np.pi/8)
                self.ax_layout.plot(ant_x, ant_y, 'k.', markersize=5)
        
        # Draw UEs
        for i, (x, y) in enumerate(self.ue_positions):
            cell_idx = i // self.env.K
            self.ax_layout.plot(x, y, 'o', 
                              color=self.cell_colors[cell_idx], 
                              markersize=8)
            
            # Draw connection lines (animated)
            bs_pos = self.bs_positions[cell_idx]
            alpha = 0.5 + 0.5 * np.sin(frame * 0.1 + i * 0.5)  # Pulsing effect
            self.ax_layout.plot([x, bs_pos[0]], [y, bs_pos[1]], 
                              '--', color=self.cell_colors[cell_idx], 
                              alpha=alpha, linewidth=0.5)
        
        # Update signal strength plot
        signal_strength = self._calculate_signal_strength(frame)
        self.ax_signal.plot(signal_strength)
        self.ax_signal.set_xlabel('UE Index')
        self.ax_signal.set_ylabel('Signal Strength (dB)')
        
        # Update interference plot
        interference = self._calculate_interference(frame)
        self.ax_interference.imshow(interference, 
                                  aspect='auto', 
                                  cmap='hot')
        self.ax_interference.set_xlabel('Interfering Cell')
        self.ax_interference.set_ylabel('Target Cell')
        
        self.ax_layout.grid(True)
        self.ax_layout.legend()
        
        return []
    
    def _calculate_signal_strength(self, frame):
        # Simulate signal strength variation over time
        base_signal = -50 - 20 * np.log10(np.arange(1, self.env.K*self.env.N + 1))
        variation = 5 * np.sin(frame * 0.1 + np.arange(self.env.K*self.env.N) * 0.5)
        return base_signal + variation
    
    def _calculate_interference(self, frame):
        # Simulate interference between cells
        interference = np.zeros((self.env.N, self.env.N))
        for i in range(self.env.N):
            for j in range(self.env.N):
                if i != j:
                    d = np.linalg.norm(self.cell_positions[i] - self.cell_positions[j])
                    interference[i,j] = 1/(d + 1) * (1 + 0.2*np.sin(frame * 0.1))
        return interference
    
    def start_animation(self):
        try:
            self.animation = FuncAnimation(self.fig, self.animate,
                                         init_func=self.init_animation,
                                         frames=200, interval=50,
                                         blit=True)
            plt.show()
        except Exception as e:
            print(f"Animation error: {e}")
        finally:
            if plt.get_fignums():  # If there are still open figures
                plt.close(self.fig)

def visualize_physical_system(env):
    """Function to create and start the physical system visualization"""
    visualizer = PhysicalSystemVisualizer(env)
    visualizer.start_animation()

# Add to the main visualization menu
def show_visualization_menu(env):
    while True:
        print("\n=== MIMO System Visualization Menu ===")
        print("1. Show Physical System Animation")
        print("2. Show Training Visualization")
        print("3. Compare Algorithms")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            visualize_physical_system(env)
        elif choice == "2":
            print("\nSelect Algorithm:")
            print("1. DQN")
            print("2. DDQN")
            print("3. RainbowDQN")
            print("4. PPO")
            
            algo_choice = input("Enter algorithm choice (1-4): ")
            episodes = int(input("Enter number of episodes: "))
            
            if algo_choice == "1":
                agent = DQNAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, False)
            elif algo_choice == "2":
                agent = DDQNAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, False)
            elif algo_choice == "3":
                agent = RainbowDQNAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, False)
            elif algo_choice == "4":
                agent = PPOAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, test_mode=False)
            else:
                print("Invalid choice")
                continue
            
            visualize_training(env, agent, episodes)
        elif choice == "3":
            episodes = int(input("Enter number of episodes: "))
            compare_algorithms(env, episodes)
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.") 