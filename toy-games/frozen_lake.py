import gymnasium as gym
import numpy as np
import plotly.graph_objects as go


ACTIONS = [0, 1, 2, 3]
STATES = list(range(64))


def get_q_table_filename(is_slippery: bool) -> str:
    if is_slippery:
        return "frozen_lake_q_table_slippery.npy"
    return "frozen_lake_q_table_non_slippery.npy"


def run(is_training=False, training_episodes=10000, render_mode="human", is_slippery=True):
    """
    Run FrozenLake environment with Q-learning.
    
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
        "FrozenLake-v1",
        render_mode=render_mode,
        map_name="8x8",
        is_slippery=is_slippery
    )

    # Reset the environment to generate the first observation
    state, info = env.reset()
    
    # Hyperparameters
    epsilon_decay = 0.00003
    epsilon = 1.0
    alpha = 0.1
    gamma = 0.9
    
    rewards_per_episode = []

    # Initialize or load Q-table
    q_table_filename = get_q_table_filename(is_slippery)
    q_table = np.zeros([len(STATES), len(ACTIONS)])
    
    if not is_training:
        q_table = np.load(q_table_filename)

    # Training/inference loop
    for episode in range(training_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if reward == 1:
                print(f"State: {state}, Action: {action}, New State: {new_state}, "
                      f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            
            # Q-learning update
            if is_training:
                q_table[state, action] = q_table[state, action] + alpha * (
                    reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
                )
            
            state = new_state

        # Decay epsilon
        epsilon = max(0, epsilon - epsilon_decay)
        rewards_per_episode.append(reward)

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
    is_training = False
    render_mode = "human" if not is_training else None
    rewards_per_episode = run(
        is_training=is_training,
        training_episodes=1,
        render_mode=render_mode,
        is_slippery=True
    )
