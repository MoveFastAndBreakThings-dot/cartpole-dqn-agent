import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 400
MAX_STEPS = 500

# 2. Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 3. Neural Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 4. Epsilon greedy policy
def select_action(state, policy_net, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()

# 5. Optimize function (single training step)
def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.stack(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(np.stack(next_states))
    dones = torch.FloatTensor(dones).unsqueeze(1)

    curr_q = policy_net(states).gather(1, actions)
    next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q = rewards + (GAMMA * next_q * (1 - dones))

    loss = nn.MSELoss()(curr_q, expected_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 6. Main training loop with visualization
def train():
    env = gym.make(ENV_NAME, render_mode="human")  # Enable human rendering (GUI window)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPS_START

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0

        for t in range(MAX_STEPS):
            env.render()  # <-- Render GUI window every step

            action = select_action(state, policy_net, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model(policy_net, target_net, memory, optimizer)

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode: {episode}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    train()
