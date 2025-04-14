import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Process, Queue, Manager
import matplotlib.pyplot as plt
import time
import os
import pygame.gfxdraw

# Enhanced Constants
WIDTH, HEIGHT = 800, 600  # Larger window
GRID_SIZE = 20
SPEED = 40
MAX_MEMORY = 100_000
LR = 0.001
GAMMA = 0.9
MAX_GENERATIONS = 500
MASTERY_THRESHOLD = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
BRIGHT_GREEN = (0, 255, 0)
RED = (200, 0, 0)
BRIGHT_RED = (255, 0, 0)
BLUE = (0, 100, 255)
GOLD = (255, 215, 0)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)

# Directions
LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3

class SnakeGameAI:
    # In the __init__ method of SnakeGameAI class
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI - Deep Q Learning")
        
        # Load fonts
        self.title_font = pygame.font.Font(None, 36)
        self.stats_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # Create a clock
        self.clock = pygame.time.Clock()
        
        # Game area dimensions - make it fill most of the window
        self.game_width = WIDTH - 200  # Reserve space for sidebar
        self.game_height = HEIGHT - 50  # Reserve space for status bar only
        
        # Game stats
        self.total_apples_eaten = 0
        self.scores_history = []
        self.skilled_games = 0
        
        # Debug - for checking positioning (moved up before reset() is called)
        self.debug_mode = False
        
        # Initialize game
        self.reset()
        
        # Create background grid
        self.grid_surface = self._create_grid()

    def _create_grid(self):
        """Create a background grid surface"""
        grid = pygame.Surface((self.game_width, self.game_height))
        grid.fill(BLACK)
        
        # Draw grid lines
        for x in range(0, self.game_width, GRID_SIZE):
            pygame.draw.line(grid, DARK_GRAY, (x, 0), (x, self.game_height))
        for y in range(0, self.game_height, GRID_SIZE):
            pygame.draw.line(grid, DARK_GRAY, (0, y), (self.game_width, y))
            
        return grid

    def reset(self):
        # Game area starts at position (0, 0) instead of (50, 50) to use more of the window
        self.game_area_x = 0
        self.game_area_y = 0
        
        # Snake properties
        self.direction = RIGHT
        self.head = [self.game_width // 2, self.game_height // 2]
        # Ensure head is aligned to grid
        self.head[0] = (self.head[0] // GRID_SIZE) * GRID_SIZE
        self.head[1] = (self.head[1] // GRID_SIZE) * GRID_SIZE
        
        self.snake = [
            self.head[:], 
            [self.head[0] - GRID_SIZE, self.head[1]], 
            [self.head[0] - 2 * GRID_SIZE, self.head[1]]
        ]
        
        # Game state
        self.score = 0
        self.frame = 0
        self.visited = set()
        self.last_positions = deque(maxlen=15)
        self.right_moves = 0
        self.wrong_moves = 0
        self.generation = 0
        self.step_counter = 0
        self.apple_skill_counter = 0
        self.is_snake_skilled = False
        self.prev_distance = None
        
        # Special effects
        self.flash_count = 0
        self.flash_food = False
        self.flash_snake = False
        
        # Place the first food
        self._place_food()

    def _place_food(self):
        # Place food within game area
        max_x = (self.game_width // GRID_SIZE) - 1
        max_y = (self.game_height // GRID_SIZE) - 1
        
        # Make sure food isn't placed where the snake currently is
        while True:
            food_position = [
                random.randint(0, max_x) * GRID_SIZE,
                random.randint(0, max_y) * GRID_SIZE
            ]
            if food_position not in self.snake:
                self.food = food_position
                break
                
        # Debug print the food position
        if self.debug_mode:
            print(f"Food placed at: {self.food}")
                
        self.step_counter = 0
        self.prev_distance = self._distance_to_food()
        self.flash_food = True  # Start flashing effect for food
        self.flash_count = 10

    def _distance_to_food(self):
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    def play_step(self, action):
        self.frame += 1
        self.step_counter += 1
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # For debugging - toggle debug mode with D key
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                if self.debug_mode:
                    print(f"Snake head: {self.head}, Food: {self.food}")

        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head[:])

        # Initialize reward
        reward = -0.1
        game_over = False

        # Calculate distance-based reward
        new_distance = self._distance_to_food()
        if new_distance < self.prev_distance:
            reward += 0.5
            self.right_moves += 1
        elif new_distance > self.prev_distance:
            reward -= 0.5
            self.wrong_moves += 1
        self.prev_distance = new_distance

        # Check for revisiting same state
        state_key = tuple(self.head + self.snake[1][0:2])
        if state_key in self.visited:
            reward -= 1
            self.wrong_moves += 1
        else:
            self.visited.add(state_key)

        # Check for getting stuck
        self.last_positions.append(tuple(self.head))
        if len(self.last_positions) == 15 and len(set(self.last_positions)) == 1:
            reward -= 5
            game_over = True
            self.wrong_moves += 5
            
        # Debug print positions
        if self.debug_mode:
            print(f"Snake head position: {self.head}")
            print(f"Food position: {self.food}")
            print(f"Distance to food: {new_distance}")
            print(f"Is position equal: {self.head == self.food}")

        # Check if food eaten - FIXED: Exact coordinate comparison for collision detection
        if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
            if self.debug_mode:
                print("FOOD EATEN!")
            self.score += 1
            self.total_apples_eaten += 1
            self.apple_skill_counter += 1
            reward = 10
            
            # Check if snake is skilled
            if self.frame < 200 and self.apple_skill_counter >= 5:
                self.is_snake_skilled = True
                self.flash_snake = True  # Flash the snake when it becomes skilled
                self.flash_count = 20
                
            self._place_food()  # Only place new food when current one is eaten
            self.right_moves += 5
        else:
            self.snake.pop()

        # Check for collision or timeout
        if self.is_collision() or self.frame > 100:
            reward = -10
            game_over = True
            self.wrong_moves += 10

        # Update the UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.total_apples_eaten, self.is_snake_skilled

    def is_collision(self):
        # Check if snake hits wall
        if (self.head[0] < 0 or self.head[0] >= self.game_width or 
            self.head[1] < 0 or self.head[1] >= self.game_height):
            return True
        
        # Check if snake hits itself
        if self.head in self.snake[1:]:
            return True
            
        return False

    def _move(self, action):
        clockwise = [RIGHT, DOWN, LEFT, UP]
        idx = clockwise.index(self.direction)
        
        # Interpret the action
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise[(idx + 1) % 4]  # Turn right
        else:  # [0, 0, 1]
            new_dir = clockwise[(idx - 1) % 4]  # Turn left
            
        self.direction = new_dir

        # Update head position
        x, y = self.head
        if self.direction == RIGHT: x += GRID_SIZE
        elif self.direction == LEFT: x -= GRID_SIZE
        elif self.direction == DOWN: y += GRID_SIZE
        elif self.direction == UP: y -= GRID_SIZE
        self.head = [x, y]

    def _update_ui(self):
        # Fill background
        self.display.fill(BLACK)
        
        # Draw panel backgrounds - use the full window
        pygame.draw.rect(self.display, DARK_GRAY, (0, 0, WIDTH, HEIGHT))
        pygame.draw.rect(self.display, BLACK, (self.game_area_x, self.game_area_y, self.game_width, self.game_height))
        
        # Draw grid in game area
        self.display.blit(self.grid_surface, (self.game_area_x, self.game_area_y))
        
        # Draw food with pulsing effect - Draw food first so snake appears on top
        food_x = self.game_area_x + self.food[0]
        food_y = self.game_area_y + self.food[1]
        
        if self.flash_food and self.flash_count > 0:
            # Flashing effect for new food
            color = BRIGHT_RED if self.flash_count % 2 == 0 else RED
            self.flash_count -= 1
        else:
            color = RED
            
        # Draw apple shape
        pygame.draw.circle(self.display, color, (food_x + GRID_SIZE//2, food_y + GRID_SIZE//2), GRID_SIZE//2)
        # Draw apple stem
        pygame.draw.line(self.display, DARK_GRAY, 
                         (food_x + GRID_SIZE//2, food_y + 2), 
                         (food_x + GRID_SIZE//2 + 2, food_y - 2), 3)
        
        # Draw snake with gradient effect and rounded corners
        for i, part in enumerate(self.snake):
            # Calculate color based on position in snake
            if self.flash_snake and i == 0 and self.flash_count % 2 == 0:
                # Flash the head when skilled
                color = GOLD
            else:
                # Gradient from head to tail
                intensity = max(50, 255 - (i * 10))
                color = (0, intensity, 0)
            
            # Draw rounded rectangle for snake parts
            x = self.game_area_x + part[0]
            y = self.game_area_y + part[1]
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            
            # Head is special
            if i == 0:
                pygame.draw.rect(self.display, color, rect, 0, 3)
                
                # Add eyes to head
                eye_size = GRID_SIZE // 4
                eye_offset = GRID_SIZE // 5
                
                # Different eye positions based on direction
                if self.direction == RIGHT:
                    eye_pos1 = (x + GRID_SIZE - eye_offset, y + eye_offset)
                    eye_pos2 = (x + GRID_SIZE - eye_offset, y + GRID_SIZE - eye_offset)
                elif self.direction == LEFT:
                    eye_pos1 = (x + eye_offset, y + eye_offset)
                    eye_pos2 = (x + eye_offset, y + GRID_SIZE - eye_offset)
                elif self.direction == UP:
                    eye_pos1 = (x + eye_offset, y + eye_offset)
                    eye_pos2 = (x + GRID_SIZE - eye_offset, y + eye_offset)
                else:  # DOWN
                    eye_pos1 = (x + eye_offset, y + GRID_SIZE - eye_offset)
                    eye_pos2 = (x + GRID_SIZE - eye_offset, y + GRID_SIZE - eye_offset)
                    
                pygame.draw.circle(self.display, WHITE, eye_pos1, eye_size)
                pygame.draw.circle(self.display, WHITE, eye_pos2, eye_size)
                pygame.draw.circle(self.display, BLACK, eye_pos1, eye_size // 2)
                pygame.draw.circle(self.display, BLACK, eye_pos2, eye_size // 2)
            else:
                pygame.draw.rect(self.display, color, rect, 0, 2)
        
        # Draw debug coordinates if in debug mode
        if self.debug_mode:
            head_text = self.small_font.render(f"Head: {self.head}", True, WHITE)
            food_text = self.small_font.render(f"Food: {self.food}", True, RED)
            self.display.blit(head_text, [10, 10])
            self.display.blit(food_text, [10, 30])
        
        # Draw sidebar background
        sidebar_x = self.game_area_x + self.game_width + 10
        sidebar_y = self.game_area_y
        sidebar_width = WIDTH - sidebar_x - 10
        sidebar_height = self.game_height
        pygame.draw.rect(self.display, DARK_GRAY, 
                         (sidebar_x, sidebar_y, sidebar_width, sidebar_height), 0, 5)
        
        # Draw title
        title = self.title_font.render("SNAKE AI", True, WHITE)
        self.display.blit(title, [sidebar_x + 20, sidebar_y + 20])
        
        # Draw stats in sidebar
        stats_y = sidebar_y + 70
        stats_x = sidebar_x + 15
        
        # Draw stats with colored backgrounds
        stats = [
            ("Generation", self.generation, BLUE),
            ("Score", self.score, GREEN),
            ("Total Apples", self.total_apples_eaten, RED),
            ("Right Moves", self.right_moves, GREEN),
            ("Wrong Moves", self.wrong_moves, RED),
            ("Frame", self.frame, BLUE)
        ]
        
        for i, (label, value, color) in enumerate(stats):
            # Draw stat background
            stat_rect = pygame.Rect(stats_x, stats_y + i*40, sidebar_width - 30, 30)
            pygame.draw.rect(self.display, BLACK, stat_rect, 0, 3)
            
            # Draw label
            label_text = self.stats_font.render(f"{label}:", True, WHITE)
            self.display.blit(label_text, [stats_x + 10, stats_y + i*40 + 5])
            
            # Draw value with color
            value_text = self.stats_font.render(f"{value}", True, color)
            self.display.blit(value_text, [stats_x + 120, stats_y + i*40 + 5])
        
        # Status indicator for skilled
        skilled_y = stats_y + len(stats)*40 + 20
        pygame.draw.rect(self.display, BLACK, (stats_x, skilled_y, sidebar_width - 30, 40), 0, 3)
        
        skilled_text = self.stats_font.render("SKILLED:", True, WHITE)
        self.display.blit(skilled_text, [stats_x + 10, skilled_y + 10])
        
        if self.is_snake_skilled:
            status_color = BRIGHT_GREEN
            status_text = "YES"
        else:
            status_color = RED
            status_text = "NO"
            
        skilled_status = self.stats_font.render(status_text, True, status_color)
        self.display.blit(skilled_status, [stats_x + 120, skilled_y + 10])
        
        # Draw footer with instructions
        footer_y = HEIGHT - 30
        footer_text = self.small_font.render("Training in progress... Press 'D' to toggle debug info", True, LIGHT_GRAY)
        self.display.blit(footer_text, [WIDTH//2 - footer_text.get_width()//2, footer_y])
        
        # Update the display
        pygame.display.flip()

    def get_state(self):
        head = self.snake[0]
        point_l = [head[0] - GRID_SIZE, head[1]]
        point_r = [head[0] + GRID_SIZE, head[1]]
        point_u = [head[0], head[1] - GRID_SIZE]
        point_d = [head[0], head[1] + GRID_SIZE]

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        state = [
            (dir_r and self._danger(point_r)) or (dir_l and self._danger(point_l)) or (dir_u and self._danger(point_u)) or (dir_d and self._danger(point_d)),
            (dir_u and self._danger(point_r)) or (dir_d and self._danger(point_l)) or (dir_l and self._danger(point_u)) or (dir_r and self._danger(point_d)),
            (dir_d and self._danger(point_r)) or (dir_u and self._danger(point_l)) or (dir_r and self._danger(point_u)) or (dir_l and self._danger(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            self.food[0] < head[0], self.food[0] > head[0],
            self.food[1] < head[1], self.food[1] > head[1]
        ]
        return np.array(state, dtype=int)

    def _danger(self, point):
        return (point in self.snake or 
                point[0] < 0 or point[0] >= self.game_width or 
                point[1] < 0 or point[1] >= self.game_height)

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

class DQNAgent:
    def __init__(self):
        self.model = LinearQNet(11, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.n_games = 0
        self.epsilon = 0
        self.total_score = 0
        self.scores = []  # Track scores for graphing

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            pred = self.model(state_tensor)
            move = torch.argmax(pred).item()
        action = [0, 0, 0]
        action[move] = 1
        return action

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        pred = self.model(state)
        target = pred.clone()
        Q_new = reward + GAMMA * torch.max(self.model(next_state)).item() if not done else reward
        target[torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        self.optimizer.step()

def train_instance(instance_id, results_queue, shared_data):
    # Initialize pygame separately in each process
    game = SnakeGameAI()
    agent = DQNAgent()
    
    consecutive_skilled_games = 0
    
    try:
        while agent.n_games < MAX_GENERATIONS:
            game.generation = agent.n_games
            state_old = game.get_state()
            action = agent.get_action(state_old)
            reward, done, apples_eaten, is_skilled = game.play_step(action)
            state_new = game.get_state()
            agent.train_step(state_old, action, reward, state_new, done)
            
            if done:
                agent.total_score = apples_eaten
                agent.scores.append(apples_eaten)
                
                # Update shared data
                shared_data[instance_id]["generations"].append(agent.n_games)
                shared_data[instance_id]["scores"].append(apples_eaten)
                shared_data[instance_id]["skilled"].append(1 if is_skilled else 0)
                
                # Check for consecutive skilled games
                if is_skilled:
                    consecutive_skilled_games += 1
                    print(f"[Instance {instance_id}] Gen {agent.n_games} | Apples: {apples_eaten} | Total: {agent.total_score} | Skilled: True | Consecutive: {consecutive_skilled_games}/{MASTERY_THRESHOLD}")
                else:
                    consecutive_skilled_games = 0
                    print(f"[Instance {instance_id}] Gen {agent.n_games} | Apples: {apples_eaten} | Total: {agent.total_score} | Skilled: False")
                
                # If snake is "really good", end training
                if consecutive_skilled_games >= MASTERY_THRESHOLD:
                    print(f"[Instance {instance_id}] Snake has become really good! Ending training.")
                    break
                    
                game.reset()
                agent.n_games += 1
    except Exception as e:
        print(f"Error in instance {instance_id}: {e}")
    finally:
        # Put results in queue
        results_queue.put({
            "instance_id": instance_id,
            "generations": agent.n_games,
            "scores": agent.scores,
            "final_score": agent.total_score
        })
        
        # Generate graph for this instance
        plot_performance(shared_data[instance_id]["generations"], 
                        shared_data[instance_id]["scores"], 
                        shared_data[instance_id]["skilled"],
                        instance_id)
    
    # Put results in queue
    results_queue.put({
        "instance_id": instance_id,
        "generations": agent.n_games,
        "scores": agent.scores,
        "final_score": agent.total_score
    })
    
    # Generate graph for this instance
    plot_performance(shared_data[instance_id]["generations"], 
                    shared_data[instance_id]["scores"], 
                    shared_data[instance_id]["skilled"],
                    instance_id)

def plot_performance(generations, scores, skilled, instance_id=None):
    # Create modern style for plots
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
    fig.patch.set_facecolor('#2D2D2D')
    
    # Plot scores
    ax1.set_facecolor('#1E1E1E')
    ax1.plot(generations, scores, color='#00FF00', linewidth=2, label='Score per Game')
    ax1.set_title(f'Snake AI Performance - Instance {instance_id}' if instance_id is not None else 'Snake AI Performance',
                 color='white', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Generation', color='white', fontsize=12)
    ax1.set_ylabel('Apples Eaten', color='white', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555555')
    
    # Plot moving average
    if len(scores) > 10:
        window_size = 10
        moving_avg = []
        for i in range(len(scores) - window_size + 1):
            avg = sum(scores[i:i+window_size]) / window_size
            moving_avg.append(avg)
        ax1.plot(generations[window_size-1:], moving_avg, color='#FF6600', 
                 linewidth=2, label='10-game Moving Average')
    
    ax1.legend(facecolor='#333333', edgecolor='#555555', framealpha=0.8)
    
    # Plot skilled status
    ax2.set_facecolor('#1E1E1E')
    ax2.scatter(generations, skilled, 
                c=['#00FF00' if s else '#FF0000' for s in skilled], 
                s=50, alpha=0.7, label='Skilled Status')
    ax2.set_title('Snake Skill Status by Generation', color='white', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Generation', color='white', fontsize=12)
    ax2.set_ylabel('Skilled', color='white', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555555')
    
    plt.tight_layout()
    filename = f'snake_performance_instance_{instance_id}.png' if instance_id is not None else 'snake_performance.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance graph saved as {filename}")

def combine_graphs(shared_data, num_instances):
    # Create modern style for plots
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), dpi=100)
    fig.patch.set_facecolor('#2D2D2D')
    
    # Define colors for different instances
    colors = ['#00FF00', '#FF6600', '#00FFFF', '#FF00FF', '#FFFF00', '#0000FF']
    
    # Plot scores for all instances
    ax1.set_facecolor('#1E1E1E')
    for i in range(num_instances):
        ax1.plot(shared_data[i]["generations"], shared_data[i]["scores"], 
                color=colors[i % len(colors)], linewidth=2, label=f'Instance {i}')
    
    ax1.set_title('Snake AI Performance Comparison', color='white', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Generation', color='white', fontsize=12)
    ax1.set_ylabel('Apples Eaten', color='white', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555555')
    ax1.legend(facecolor='#333333', edgecolor='#555555', framealpha=0.8)
    
    # Plot skilled status for all instances
    ax2.set_facecolor('#1E1E1E')
    for i in range(num_instances):
        # Calculate percentage of skilled games in a window
        window_size = 10
        skill_rates = []
        gens = []
        
        if len(shared_data[i]["skilled"]) > window_size:
            for j in range(0, len(shared_data[i]["skilled"]) - window_size + 1, window_size):
                skill_rate = sum(shared_data[i]["skilled"][j:j+window_size]) / window_size
                skill_rates.append(skill_rate)
                gens.append(shared_data[i]["generations"][j+window_size-1])
            
            ax2.plot(gens, skill_rates, color=colors[i % len(colors)], 
                     linewidth=2, label=f'Instance {i} Skill Rate')
    
    ax2.set_title('Snake Skill Rate by Generation', color='white', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Generation', color='white', fontsize=12)
    ax2.set_ylabel('Skill Rate (10-game window)', color='white', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555555')
    ax2.legend(facecolor='#333333', edgecolor='#555555', framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig('snake_performance_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Combined performance graph saved as snake_performance_combined.png")

if __name__ == "__main__":
    # Set window position to top-left corner for better visibility
    os.environ['SDL_VIDEO_WINDOW_POS'] = '50,50'
    
    # Initialize pygame once before creating processes
    pygame.init()
    
    num_instances = 2
    results_queue = Queue()
    manager = Manager()
    shared_data = manager.dict()
    
    # Initialize shared data structure
    for i in range(num_instances):
        shared_data[i] = {
            "generations": manager.list(),
            "scores": manager.list(),
            "skilled": manager.list()
        }
    
    processes = [Process(target=train_instance, args=(i, results_queue, shared_data)) for i in range(num_instances)]
    
    # Start all processes
    for p in processes:
        p.daemon = True  # Make processes daemon so they exit when main process exits
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    # Generate combined graph
    combine_graphs(shared_data, num_instances)
    
    # Display results summary
    print("\n====== TRAINING COMPLETE ======")
    print("\nResults Summary:")
    for result in results:
        print(f"Instance {result['instance_id']} - Trained for {result['generations']} generations, final score: {result['final_score']}")
    print("\nPerformance graphs have been saved.")
    
    # Quit pygame properly
    pygame.quit()