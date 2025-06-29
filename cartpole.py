import gymnasium as gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import sys
import time
from typing import Tuple, List, Any
import os
import glob

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
SAVE_INTERVAL = 1  # Save every episode instead of every 50 episodes
BEST_MODEL_THRESHOLD = 450  # Save when reward exceeds this
CHECKPOINT_DIR = "checkpoints/"

# Pygame visualization constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CART_WIDTH = 100
CART_HEIGHT = 30
POLE_LENGTH = 150
POLE_WIDTH = 10
GRAVITY = 9.8

# 2. Replay Memory
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

# 3. Neural Network (DQN) - Improved architecture
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 4. Epsilon greedy policy with temperature
def select_action(state: np.ndarray, policy_net: DQN, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()

# 5. Optimize function with gradient clipping
def optimize_model(policy_net: DQN, target_net: DQN, memory: ReplayMemory, optimizer: optim.Optimizer) -> float:
    if len(memory) < BATCH_SIZE:
        return 0.0

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

    loss = nn.HuberLoss()(curr_q, expected_q)  # More robust loss function

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()
    
    return loss.item()

# 6. Pygame visualization class
class CartPoleVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("CartPole DQN Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        
    def draw_cartpole(self, state: np.ndarray, episode: int, total_reward: float, epsilon: float, loss: float):
        self.screen.fill(self.WHITE)
        
        # Extract state variables
        cart_pos, cart_vel, pole_angle, pole_vel = state
        
        # Scale cart position to screen coordinates
        cart_x = int(WINDOW_WIDTH // 2 + cart_pos * 200)
        cart_y = WINDOW_HEIGHT - 100
        
        # Draw ground
        pygame.draw.line(self.screen, self.BLACK, (0, cart_y + CART_HEIGHT//2), 
                        (WINDOW_WIDTH, cart_y + CART_HEIGHT//2), 2)
        
        # Draw cart (black rectangle)
        cart_rect = pygame.Rect(cart_x - CART_WIDTH//2, cart_y - CART_HEIGHT//2, 
                               CART_WIDTH, CART_HEIGHT)
        pygame.draw.rect(self.screen, self.BLACK, cart_rect)
        pygame.draw.rect(self.screen, self.BLACK, cart_rect, 2)
        
        # Draw pole (brown rectangle, rotated) - pole base at top of cart
        pole_color = (181, 101, 29)  # Brown
        pole_length_px = POLE_LENGTH
        pole_width_px = POLE_WIDTH
        pole_angle_rad = pole_angle
        
        # Create a surface for the pole
        pole_surface = pygame.Surface((pole_width_px, pole_length_px), pygame.SRCALPHA)
        pole_surface.fill(pole_color)
        
        # Rotate the pole surface
        pole_rotated = pygame.transform.rotate(pole_surface, -np.degrees(pole_angle_rad))
        
        # Get the new rect and position it correctly
        pole_rect = pole_rotated.get_rect()
        
        # Calculate the pivot point (where pole connects to cart)
        pivot_x = cart_x
        pivot_y = cart_y - CART_HEIGHT//2
        
        # Position the pole so its bottom center is at the pivot point
        # The pole surface is created with the base at the bottom, so we use midbottom
        pole_rect.midbottom = (pivot_x, pivot_y)
        
        # Draw the pole
        self.screen.blit(pole_rotated, pole_rect)
        
        # Draw blue circle at pole base (connection point) to show the pivot
        pygame.draw.circle(self.screen, (50, 60, 200), (pivot_x, pivot_y), 8)
        
        # Draw information panel
        info_y = 20
        info_spacing = 30
        texts = [
            f"Episode: {episode}",
            f"Reward: {total_reward:.1f}",
            f"Epsilon: {epsilon:.3f}",
            f"Loss: {loss:.4f}",
            f"Cart Pos: {cart_pos:.2f}",
            f"Pole Angle: {np.degrees(pole_angle):.1f}°"
        ]
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (20, info_y + i * info_spacing))
        
        # Draw performance indicator
        if total_reward > 450:
            color = self.GREEN
        elif total_reward > 200:
            color = (255, 165, 0)  # Orange
        else:
            color = self.RED
        pygame.draw.circle(self.screen, color, (WINDOW_WIDTH - 30, 30), 15)
        
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint file in the directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_episode_*.pth"))
    if not checkpoint_files:
        return None
    
    # Extract episode numbers and find the latest
    episode_numbers = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        try:
            episode_num = int(filename.split('_')[2].split('.')[0])
            episode_numbers.append((episode_num, file_path))
        except (IndexError, ValueError):
            continue
    
    if not episode_numbers:
        return None
    
    # Return the file path with the highest episode number
    latest_checkpoint = max(episode_numbers, key=lambda x: x[0])[1]
    return latest_checkpoint

# 7. Enhanced training loop with visualization
def train():
    env = gym.make(ENV_NAME)
    # Type assertions for gymnasium spaces
    assert env.observation_space.shape is not None
    n_states = env.observation_space.shape[0]
    assert hasattr(env.action_space, 'n')
    n_actions = env.action_space.n

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR, weight_decay=1e-5)
    memory = ReplayMemory(MEMORY_SIZE)
    visualizer = CartPoleVisualizer()

    epsilon = EPS_START
    episode_rewards = []
    running_loss = 0.0
    best_reward = 0.0
    start_episode = 1
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Try to load the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        try:
            start_episode, epsilon, best_reward = load_checkpoint(
                latest_checkpoint, 
                policy_net, 
                target_net, 
                optimizer
            )
            start_episode += 1  # Start from the next episode
            print(f"Resuming training from episode {start_episode} because we have already save till {start_episode - 1}")
            print(f"Loaded epsilon: {epsilon:.3f}, best reward: {best_reward:.2f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training...")
            start_episode = 1
    else:
        print("No checkpoint found. Starting fresh training...")

    print("Starting CartPole DQN Training with Pygame Visualization...")
    print("Press ESC to exit")

    for episode in range(start_episode, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        episode_loss = 0.0

        for t in range(MAX_STEPS):
            # Handle pygame events
            visualizer.handle_events()
            
            # Draw the current state using pygame
            visualizer.draw_cartpole(state, episode, total_reward, epsilon, running_loss)

            action = select_action(state, policy_net, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, float(reward), next_state, done)
            state = next_state
            total_reward += float(reward)

            # Update running loss
            episode_loss = optimize_model(policy_net, target_net, memory, optimizer)
            running_loss = episode_loss if episode_loss > 0 else running_loss

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode: {episode}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

        # Save checkpoints and best model
        if episode % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_episode_{episode}.pth")
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_reward': best_reward,
                'episode_rewards': episode_rewards
            }, checkpoint_path)
             
        
        # Save best model based on performance
        if total_reward > best_reward:
            best_reward = total_reward
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'best_reward': best_reward
            }, best_model_path)
            print(f"New best model saved! Reward: {best_reward:.2f}")

    env.close()
    pygame.quit()

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save({
        'episode': NUM_EPISODES,
        'policy_net_state_dict': policy_net.state_dict(),
        'final_reward': total_reward,
        'best_reward': best_reward
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best reward achieved: {best_reward:.2f}")

def load_checkpoint(checkpoint_path: str, policy_net: DQN, target_net: DQN, optimizer: optim.Optimizer):
    checkpoint = torch.load(checkpoint_path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['epsilon'], checkpoint['best_reward']

if __name__ == "__main__":
    train()