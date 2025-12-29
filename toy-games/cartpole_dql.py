import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import torch
import torch.nn as nn
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots
from collections import deque


class DQN(nn.Module):
    def __init__(self, in_states, hidden_size, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class CartPoleDDQN:
    def __init__(self, render_mode=None, use_double_dqn=True):
        # Hyperparameters
        self.learning_rate = 0.001
        self.discount_factor = 0.99  # Higher for CartPole (longer episodes matter)
        self.replay_buffer_size = 10000
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Multiplicative decay per episode
        self.network_sync_rate = 100  # Steps between target network updates
        self.hidden_size = 128
        self.use_double_dqn = use_double_dqn
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Environment setup
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.num_actions = self.env.action_space.n  # 2: left, right
        self.num_states = self.env.observation_space.shape[0]  # 4: cart_pos, cart_vel, pole_angle, pole_vel
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
    
    def build_model(self):
        """Initialize policy and target networks"""
        self.policy_network = DQN(self.num_states, self.hidden_size, self.num_actions).to(self.device)
        self.target_network = DQN(self.num_states, self.hidden_size, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def state_to_tensor(self, state):
        """Convert numpy state to tensor - no one-hot needed for continuous states"""
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_network(state_tensor)
                return torch.argmax(q_values).item()
            
    def optimize(self):
        """Vectorized batch optimization with Double DQN support"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack and tensorize batch (vectorized)
        states = torch.tensor(np.array([s for s, _, _, _, _ in batch]), 
                             dtype=torch.float32, device=self.device)
        actions = torch.tensor([a for _, a, _, _, _ in batch], 
                              dtype=torch.long, device=self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], 
                              dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([ns for _, _, _, ns, _ in batch]), 
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor([d for _, _, _, _, d in batch], 
                            dtype=torch.float32, device=self.device)

        # Current Q-values for actions taken
        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute targets
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: policy network SELECTS, target network EVALUATES
                best_actions = self.policy_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN: target network does both
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            targets = rewards + self.discount_factor * next_q * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def train(self, episodes):
        """Main training loop"""
        self.build_model()
        
        rewards_per_episode = []
        losses = []
        epsilon_history = []
        step_count = 0
        
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                # Select and execute action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, terminated)
                
                # Optimize
                loss = self.optimize()
                if loss is not None:
                    episode_losses.append(loss)
                
                # Sync target network
                step_count += 1
                if step_count % self.network_sync_rate == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                
                state = next_state
                episode_reward += reward
            
            # Decay epsilon (per episode, multiplicative)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            rewards_per_episode.append(episode_reward)
            epsilon_history.append(self.epsilon)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"\nEpisode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        self.env.close()
        
        # Save models
        torch.save(self.policy_network.state_dict(), "cartpole_ddqn_policy.pth")
        torch.save(self.target_network.state_dict(), "cartpole_ddqn_target.pth")
        
        return rewards_per_episode, epsilon_history, losses

    def test(self, episodes=5, render=True):
        """Test the trained agent"""
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        
        self.epsilon = 0  # No exploration during testing
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
            
            print(f"Test Episode {episode + 1}: Reward = {total_reward}")
        
        env.close()


def plot_results(rewards, epsilon_history, losses, window=100):
    """Plot training results"""
    episodes = len(rewards)
    
    # Calculate moving average
    moving_avg = []
    for i in range(episodes):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Episode Rewards (Moving Avg: {window})',
            'Epsilon Decay',
            'Training Loss',
            'Raw Episode Rewards'
        )
    )
    
    # Moving average rewards
    fig.add_trace(
        go.Scatter(x=list(range(episodes)), y=moving_avg, 
                   mode='lines', name='Moving Avg Reward',
                   line=dict(color='#636EFA', width=2)),
        row=1, col=1
    )
    
    # Epsilon decay
    fig.add_trace(
        go.Scatter(x=list(range(episodes)), y=epsilon_history,
                   mode='lines', name='Epsilon',
                   line=dict(color='#FFA15A', width=2)),
        row=1, col=2
    )
    
    # Training loss
    if losses:
        fig.add_trace(
            go.Scatter(x=list(range(len(losses))), y=losses,
                       mode='lines', name='Loss',
                       line=dict(color='#EF553B', width=1)),
            row=2, col=1
        )
    
    # Raw rewards
    fig.add_trace(
        go.Scatter(x=list(range(episodes)), y=rewards,
                   mode='lines', name='Raw Reward',
                   line=dict(color='#00CC96', width=1, dash='dot')),
        row=2, col=2
    )
    
    # Add solved threshold line (CartPole is "solved" at 475)
    fig.add_hline(y=475, line_dash="dash", line_color="red", 
                  annotation_text="Solved (475)", row=1, col=1)
    fig.add_hline(y=475, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="CartPole Double DQN Training Results",
        showlegend=True
    )
    
    fig.show()


if __name__ == "__main__":
    # Train with Double DQN
    print("=" * 50)
    print("Training CartPole with Double DQN")
    print("=" * 50)
    
    agent = CartPoleDDQN(render_mode=None, use_double_dqn=True)
    rewards, epsilon_history, losses = agent.train(episodes=5000)
    
    # Plot results
    plot_results(rewards, epsilon_history, losses)
    
    # Test the trained agent
    print("\n" + "=" * 50)
    print("Testing trained agent...")
    print("=" * 50)
    agent.test(episodes=3, render=True)