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
        self.fc2 = nn.Linear(hidden_size, out_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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

class FrozenLakeDQL:
    def __init__(self, render_mode="human"):
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.replay_buffer_size = 10000
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.0001
        self.network_sync_rate = 100
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode=render_mode)
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n
        self.ACTIONS = ["L", "D", "R", "U"]
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
    
    def build_model(self, in_states, hidden_size, out_actions):
        self.policy_network = DQN(in_states, hidden_size, out_actions).to(self.device)
        self.target_network = DQN(in_states, hidden_size, out_actions).to(self.device)
    

    def state_to_dql_input(self, state):
        return torch.nn.functional.one_hot(torch.tensor(state), self.num_states).float().to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state = self.state_to_dql_input(state)
                q_values = self.policy_network(state)
                return torch.argmax(q_values).item()
            
    def optimize(self, batch):
        current_q_values = []
        target_q_values = []
        for state, action, reward, next_state, done in batch:
            state = self.state_to_dql_input(state)
            target = reward + self.discount_factor * self.target_network(self.state_to_dql_input(next_state)).max() * (1 - done)
            current_q = self.policy_network(state)
            current_q_values.append(current_q)
            target_q = self.target_network(state)
            target_q[action] = target
            target_q_values.append(target_q)
        
        loss = self.loss_fn(torch.stack(current_q_values), torch.stack(target_q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, episodes, render_mode=None):
        self.build_model(self.num_states, self.num_states, self.num_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0 # to sync the networks 
        for episode in tqdm(range(episodes), desc="Training", total=episodes):
            state, info = self.env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                rewards_per_episode[episode] += reward
                self.replay_buffer.push(state, action, reward, next_state, terminated)
                state = next_state
                step_count += 1

            if len(self.replay_buffer) >= self.batch_size and np.sum(rewards_per_episode) > 0:
                batch = self.replay_buffer.sample(self.batch_size)
                self.optimize(batch)
                self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
                if step_count % self.network_sync_rate == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                    step_count = 0
            epsilon_history.append(self.epsilon)

        self.env.close()
        torch.save(self.policy_network.state_dict(), "frozen_lake_dql_policy_network.pth")
        torch.save(self.target_network.state_dict(), "frozen_lake_dql_target_network.pth")
        return rewards_per_episode, epsilon_history
        

    def load_model(self, policy_network_path, target_network_path):
        self.policy_network.load_state_dict(torch.load(policy_network_path))
        self.target_network.load_state_dict(torch.load(target_network_path))

    def run(self, episodes, render_mode=None):
        self.train(episodes, render_mode)
        self.env.close()
        return self.rewards_per_episode, self.epsilon_history


if __name__ == "__main__":
    dql = FrozenLakeDQL(render_mode=None)
    episodes = 10000
    rewards_per_episode, epsilon_history = dql.train(episodes)
    sum_rewards = np.zeros(episodes)
    for i in range(episodes):
        sum_rewards[i] = np.sum(rewards_per_episode[max(0, i - 100):i])

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Sum of Rewards (Last 100 Episodes)', 'Epsilon')) # add two plots in one figure, one on the left and one on the right
    fig.add_trace(go.Scatter(x=list(range(episodes)), y=sum_rewards, mode='lines', name='Sum of Rewards (Last 100 Episodes)', line=dict(color='#636EFA', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(episodes)), y=epsilon_history, mode='lines', name='Epsilon', line=dict(color='#FFA15A', width=1)), row=1, col=2)
    fig.show()
