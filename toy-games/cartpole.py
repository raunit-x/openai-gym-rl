import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


def get_q_table_filename() -> str:
    return "toy-games/q-tables/cartpole_q_table.npy"


def run(is_training=False, training_episodes=10000, render_mode="human"):
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
        "CartPole-v1",
        render_mode=render_mode
    )
    cart_positions = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    cart_velocities = np.linspace(-4, 4, 20)
    pole_angles = np.linspace(-0.418, 0.418, 20)
    pole_velocities = np.linspace(-4.0, 4.0, 20)
    q_table = np.zeros([len(cart_positions), len(cart_velocities), len(pole_angles), len(pole_velocities), env.action_space.n])
    if not is_training:
        q_table = np.load(get_q_table_filename())
        print(f"Q-table loaded from {get_q_table_filename()}")

    # Reset the environment to generate the first observation
    epsilon = 1.0
    epsilon_decay = 1 / training_episodes
    min_epsilon = 0.01
    alpha = 0.1
    gamma = 0.99
    rewards_per_episode = []
    for episode in tqdm(range(training_episodes), desc="Training", total=training_episodes):
        state, info = env.reset()
        rewards = 0
        terminated = False
        while not terminated and rewards < 10000:
            cart_position = np.digitize(state[0], cart_positions) - 1
            cart_velocity = np.digitize(state[1], cart_velocities) - 1
            pole_angle = np.digitize(state[2], pole_angles) - 1
            pole_velocity = np.digitize(state[3], pole_velocities) - 1
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[cart_position, cart_velocity, pole_angle, pole_velocity, :])
            new_state, reward, terminated, _, _ = env.step(action)
            rewards += reward
            state = new_state
            if is_training:
                new_cart_position = np.digitize(new_state[0], cart_positions) - 1
                new_cart_velocity = np.digitize(new_state[1], cart_velocities) - 1
                new_pole_angle = np.digitize(new_state[2], pole_angles) - 1
                new_pole_velocity = np.digitize(new_state[3], pole_velocities) - 1
                q_table[cart_position, cart_velocity, pole_angle, pole_velocity, action] = q_table[cart_position, cart_velocity, pole_angle, pole_velocity, action] + alpha * (
                    reward + gamma * np.max(q_table[new_cart_position, new_cart_velocity, new_pole_angle, new_pole_velocity, :]) - q_table[cart_position, cart_velocity, pole_angle, pole_velocity, action]
                )
                epsilon = max(min_epsilon, epsilon - epsilon_decay)
        rewards_per_episode.append(rewards)
    env.close()

    # Save Q-table after training
    if is_training:
        np.save(get_q_table_filename(), q_table)
        print(f"Q-table saved to {get_q_table_filename()}")

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
        training_episodes=100000,
        render_mode=render_mode,
    )
    run(
        is_training=False,
        training_episodes=3,
        render_mode="human"
    )
