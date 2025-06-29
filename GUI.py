import pygame
import sys
import numpy as np
from typing import Tuple

# Pygame visualization constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CART_WIDTH = 100
CART_HEIGHT = 30
POLE_LENGTH = 150
POLE_WIDTH = 10
GRAVITY = 9.8

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
        """Draw the CartPole environment with current state and training information"""
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
        
    def handle_events(self) -> bool:
        """Handle Pygame events. Returns False if the window should be closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                    return False
        return True
    
    def close(self):
        """Clean up Pygame resources"""
        pygame.quit()

class TrainingVisualizer:
    """Enhanced visualizer for training progress and statistics"""
    
    def __init__(self, window_width: int = 800, window_height: int = 600):
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("CartPole DQN Training Progress")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        
        # Training history
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_history = []
        
    def update_training_data(self, episode: int, reward: float, loss: float, epsilon: float):
        """Update training statistics"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        self.epsilon_history.append(epsilon)
        
        # Keep only last 100 episodes for display
        if len(self.episode_rewards) > 100:
            self.episode_rewards = self.episode_rewards[-100:]
            self.episode_losses = self.episode_losses[-100:]
            self.epsilon_history = self.epsilon_history[-100:]
    
    def draw_training_progress(self, episode: int, current_reward: float, current_loss: float, 
                             current_epsilon: float, best_reward: float):
        """Draw training progress charts and statistics"""
        self.screen.fill(self.WHITE)
        
        # Update training data
        self.update_training_data(episode, current_reward, current_loss, current_epsilon)
        
        # Draw title
        title = self.large_font.render(f"CartPole DQN Training - Episode {episode}", True, self.BLACK)
        self.screen.blit(title, (20, 20))
        
        # Draw current statistics
        stats_y = 80
        stats = [
            f"Current Reward: {current_reward:.2f}",
            f"Best Reward: {best_reward:.2f}",
            f"Current Loss: {current_loss:.4f}",
            f"Current Epsilon: {current_epsilon:.3f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, self.BLACK)
            self.screen.blit(text, (20, stats_y + i * 25))
        
        # Draw reward chart
        if len(self.episode_rewards) > 1:
            self._draw_chart(self.episode_rewards, 200, 200, 400, 150, "Reward History", self.GREEN)
        
        # Draw loss chart
        if len(self.episode_losses) > 1:
            self._draw_chart(self.episode_losses, 200, 400, 400, 150, "Loss History", self.RED)
        
        # Draw epsilon chart
        if len(self.epsilon_history) > 1:
            self._draw_chart(self.epsilon_history, 650, 200, 120, 150, "Epsilon", self.BLUE)
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS for progress display
    
    def _draw_chart(self, data: list, x: int, y: int, width: int, height: int, 
                   title: str, color: Tuple[int, int, int]):
        """Draw a simple line chart"""
        if len(data) < 2:
            return
            
        # Draw chart background
        pygame.draw.rect(self.screen, (240, 240, 240), (x, y, width, height))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, width, height), 2)
        
        # Draw title
        title_text = self.font.render(title, True, self.BLACK)
        self.screen.blit(title_text, (x, y - 25))
        
        # Calculate scaling
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1
            
        # Draw data points and lines
        points = []
        for i, value in enumerate(data):
            px = x + (i / (len(data) - 1)) * width
            py = y + height - ((value - min_val) / (max_val - min_val)) * height
            points.append((px, py))
        
        # Draw lines connecting points
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)
        
        # Draw data points
        for px, py in points:
            pygame.draw.circle(self.screen, color, (int(px), int(py)), 3)
    
    def handle_events(self) -> bool:
        """Handle Pygame events. Returns False if the window should be closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def close(self):
        """Clean up Pygame resources"""
        pygame.quit()

# Utility functions for GUI management
def create_dual_window_setup():
    """Create both CartPole visualizer and training progress visualizer"""
    cartpole_viz = CartPoleVisualizer()
    training_viz = TrainingVisualizer()
    return cartpole_viz, training_viz

def handle_dual_window_events(cartpole_viz: CartPoleVisualizer, training_viz: TrainingVisualizer) -> bool:
    """Handle events for both windows"""
    cartpole_continue = cartpole_viz.handle_events()
    training_continue = training_viz.handle_events()
    return cartpole_continue and training_continue

def close_dual_windows(cartpole_viz: CartPoleVisualizer, training_viz: TrainingVisualizer):
    """Close both visualization windows"""
    cartpole_viz.close()
    training_viz.close()
