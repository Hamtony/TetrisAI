import gymnasium as gym
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tetrisEnv import TetrisEnv
import collections
from collections import deque
env = TetrisEnv()
observation_space = env.observation_space
action_space = env.action_space
class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()

        # Define layers based on observation space complexity
        # (Adjust based on your observations' dimensionality)
        self.fc1 = nn.Linear(observation_space.n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_space.n)

    def forward(self, x):
        x = x.float()  # Ensure float tensor for calculations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return torch.cat(states), torch.tensor(actions), torch.tensor(rewards), torch.cat(next_states), torch.tensor(dones)

    def __len__(self):
        return len(self.memory)
def to_tensor(observation):
    """Converts observation dictionary to a PyTorch tensor."""
    # Extract relevant data from the observation dictionary
    x_piece = torch.tensor(observation['x_piece'], dtype=torch.float32).unsqueeze(0)
    y_piece = torch.tensor(observation['y_piece'], dtype=torch.float32).unsqueeze(0)
    piece_type = torch.tensor(observation['piece_type'], dtype=torch.float32).unsqueeze(0)
    piece_rotation = torch.tensor(observation['piece_rotation'], dtype=torch.float32).unsqueeze(0)
    field = torch.tensor(observation['field'].reshape(20, 10), dtype=torch.float32).unsqueeze(0)
    hold = torch.tensor(observation['hold'], dtype=torch.float32).unsqueeze(0)
    queue = torch.tensor(observation['queue'], dtype=torch.float32).unsqueeze(0)

    # Concatenate all elements into a single tensor
    observation_tensor = torch.cat((x_piece, y_piece, piece_type, piece_rotation, field, hold, queue), dim=1)
    return observation_tensor
# Hyperparameters
learning_rate = 0.001
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate (decay during training)
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 10000
target_update = 200  # How often to update target network

# Create DQN model, target network (for stability), optimizer, and replay memory
dqn_model = DQN(observation_space.n, action_space.n)
target_model = DQN(observation_space.n, action_space.n)
optimizer = optim.Adam(dqn_model.parameters(), lr=learning_rate)
replay_memory = ReplayMemory(memory_size)

# Training loop
num_episodes = 1000  # Adjust training episodes as needed
total_reward = 0
for episode in range(num_episodes):
    state = env.reset()
    state_tensor = to_tensor(state)  # Convert initial state to tensor
    done = False

    while not done:
        # Choose action with epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                action = torch.argmax(q_values).item()

        # Take action, observe reward and next state
        next_state, reward, done, _ = env.step(action)
        next_state_tensor = to_tensor(next_state)  # Convert next state to tensor

        # Store experience in replay memory
        replay_memory.push(state_tensor, action, reward, next_state_tensor, done)
        total_reward += reward

        if len(replay_memory) > batch_size:
            # Sample experiences from replay memory
            experiences = replay_memory.sample(batch_size)
            states, actions, rewards, next_states, dones = experiences

            # Calculate Q-values using DQN
            q_values = dqn_model(states)
            next_q_values = target_model(next_states)

            # Calculate expected Q-values (Bellman equation)
            expected_q_values = rewards + gamma * next_q_values.max(1)[0].unsqueeze(1)

            # Calculate loss (huber loss for robustness)
            criterion = nn.SmoothL1Loss()
            loss = criterion(q_values, expected_q_values)

            # Optimize DQN model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network periodically
            if episode % target_update == 0:
                target_model.load_state_dict(dqn_model.state_dict())

        # Update epsilon for exploration decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        state = next_state
        state_tensor = next_state_tensor

    # Print episode statistics (optional)
    print(f"Episode: {episode+1}, Reward: {total_reward}")
    total_reward = 0

env.close()
