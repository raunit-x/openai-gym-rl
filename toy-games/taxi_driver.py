import gymnasium as gym
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go


def get_q_table_filename() -> str:
    return "taxi_driver_q_table.npy"


def run(is_training=False, training_episodes=10000, render_mode="human", is_slippery=True):
    """
    Run TaxiDriver environment with Q-learning.
    
    Args:
        is_training: Whether to train the agent or run inference
        training_episodes: Number of episodes to run
        render_mode: Render mode for the environment ("human", None, etc.)
        is_slippery: Whether the ice is slippery (stochastic transitions)
    
    Returns:
        List of rewards per episode
    """
    # Initialize the environment
    env = gym.make(
        "Taxi-v3",
        render_mode=render_mode
    )
    ACTIONS = env.action_space.n
    STATES = env.observation_space.n

    # Reset the environment to generate the first observation
    state, info = env.reset()
    
    # Hyperparameters
    epsilon = 1.0
    alpha = 0.1
    gamma = 0.99
    
    rewards_per_episode = []

    # Initialize or load Q-table
    q_table_filename = get_q_table_filename()
    q_table = np.zeros([STATES, ACTIONS])
    
    if not is_training:
        q_table = np.load(q_table_filename)

    # Training/inference loop
    for episode in tqdm(range(training_episodes), desc="Training"):
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            # Q-learning update
            if is_training:
                q_table[state, action] = q_table[state, action] + alpha * (
                    reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
                )
            
            state = new_state

        # Decay epsilon
        epsilon = max(0.01, epsilon - 1 / training_episodes)
        rewards_per_episode.append(episode_reward)

        # Reduce learning rate when exploration is done
        if epsilon == 0:
            alpha = 0.0001

    env.close()

    # Save Q-table after training
    if is_training:
        np.save(q_table_filename, q_table)
        print(f"Q-table saved to {q_table_filename}")

        # Plot rewards using Plotly
        sum_rewards = np.zeros(training_episodes)
        for i in range(training_episodes):
            sum_rewards[i] = np.sum(rewards_per_episode[max(0, i - 100):i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(training_episodes)),
            y=sum_rewards,
            mode='lines',
            name='Sum of Rewards (Last 100 Episodes)',
            line=dict(color='#636EFA', width=2)
        ))
        fig.update_layout(
            title='Sum of Rewards per Episode',
            xaxis_title='Episode',
            yaxis_title='Sum of Rewards',
            template='plotly_dark',
            hovermode='x unified'
        )
        fig.show()

    return rewards_per_episode


if __name__ == "__main__":
    is_training = True
    render_mode = "human" if not is_training else None
    rewards_per_episode = run(
        is_training=is_training,
        training_episodes=50000,
        render_mode=render_mode
    )
    run(
        is_training=False,
        training_episodes=3,
        render_mode="human"
    )

