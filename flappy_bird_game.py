
import pygame
import random
import numpy as np
from collections import deque
import sys
import os

class FlappyBirdGame:
    def __init__(self):
        pygame.init()

        # Game constants
        self.WINDOW_WIDTH = 400
        self.WINDOW_HEIGHT = 600
        self.GROUND_HEIGHT = 100
        self.PIPE_WIDTH = 80
        self.PIPE_GAP = 200
        self.PIPE_SPEED = 5
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -8
        self.FPS = 60

        # Initialize display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird RL")
        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 0, 0)

        # Game state
        self.reset()

    def reset(self):
        # Bird properties
        self.bird_x = 50
        self.bird_y = self.WINDOW_HEIGHT // 2
        self.bird_velocity = 0

        # Pipes
        self.pipes = []
        self.pipe_timer = 0

        # Game state
        self.score = 0
        self.game_over = False

        # Add initial pipe
        self.add_pipe()

        return self.get_state()

    def add_pipe(self):
        # Random pipe height
        pipe_height = random.randint(50, self.WINDOW_HEIGHT - self.GROUND_HEIGHT - self.PIPE_GAP - 50)
        self.pipes.append({
            'x': self.WINDOW_WIDTH,
            'top_height': pipe_height,
            'bottom_y': pipe_height + self.PIPE_GAP,
            'passed': False
        })

    def step(self, action):
        # Action: 0 = do nothing, 1 = jump
        if action == 1:
            self.bird_velocity = self.JUMP_STRENGTH

        # Update bird physics
        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity

        # Update pipes
        for pipe in self.pipes[:]:
            pipe['x'] -= self.PIPE_SPEED

            # Check if bird passed the pipe (for scoring)
            if not pipe['passed'] and pipe['x'] + self.PIPE_WIDTH < self.bird_x:
                pipe['passed'] = True
                self.score += 1

            # Remove pipes that are off screen
            if pipe['x'] + self.PIPE_WIDTH < 0:
                self.pipes.remove(pipe)

        # Add new pipes
        self.pipe_timer += 1
        if self.pipe_timer > 90:  # Add pipe every 90 frames
            self.add_pipe()
            self.pipe_timer = 0

        # Check collisions
        reward = 0.1  # Small positive reward for staying alive

        if self.check_collision():
            self.game_over = True
            reward = -100  # Large negative reward for collision

        next_state = self.get_state()

        return next_state, reward, self.game_over

    def check_collision(self):
        # Bird collision with ground or ceiling
        if self.bird_y <= 0 or self.bird_y >= self.WINDOW_HEIGHT - self.GROUND_HEIGHT:
            return True

        # Bird collision with pipes
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, 30, 30)

        for pipe in self.pipes:
            # Top pipe collision
            top_pipe_rect = pygame.Rect(pipe['x'], 0, self.PIPE_WIDTH, pipe['top_height'])
            # Bottom pipe collision  
            bottom_pipe_rect = pygame.Rect(pipe['x'], pipe['bottom_y'], 
                                         self.PIPE_WIDTH, 
                                         self.WINDOW_HEIGHT - pipe['bottom_y'] - self.GROUND_HEIGHT)

            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                return True

        return False

    def get_state(self):
        # Get the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.PIPE_WIDTH > self.bird_x:
                next_pipe = pipe
                break

        if next_pipe is None:
            # No pipes ahead, use default values
            horizontal_distance = self.WINDOW_WIDTH
            vertical_distance = 0
            pipe_top = self.WINDOW_HEIGHT // 2 - self.PIPE_GAP // 2
            pipe_bottom = self.WINDOW_HEIGHT // 2 + self.PIPE_GAP // 2
        else:
            horizontal_distance = next_pipe['x'] - self.bird_x
            vertical_distance = self.bird_y - (next_pipe['top_height'] + self.PIPE_GAP // 2)
            pipe_top = next_pipe['top_height']
            pipe_bottom = next_pipe['bottom_y']

        # Normalize state values
        state = np.array([
            self.bird_y / self.WINDOW_HEIGHT,                    # Bird Y position (normalized)
            self.bird_velocity / 10.0,                          # Bird velocity (normalized)
            horizontal_distance / self.WINDOW_WIDTH,             # Horizontal distance to next pipe
            vertical_distance / self.WINDOW_HEIGHT,              # Vertical distance to pipe center
            pipe_top / self.WINDOW_HEIGHT,                       # Next pipe top position
            pipe_bottom / self.WINDOW_HEIGHT                     # Next pipe bottom position
        ], dtype=np.float32)

        return state

    def render(self):
        # Clear screen
        self.screen.fill(self.WHITE)

        # Draw bird
        pygame.draw.circle(self.screen, self.BLUE, 
                         (int(self.bird_x + 15), int(self.bird_y + 15)), 15)

        # Draw pipes
        for pipe in self.pipes:
            # Top pipe
            pygame.draw.rect(self.screen, self.GREEN, 
                           (pipe['x'], 0, self.PIPE_WIDTH, pipe['top_height']))
            # Bottom pipe
            pygame.draw.rect(self.screen, self.GREEN,
                           (pipe['x'], pipe['bottom_y'], self.PIPE_WIDTH, 
                            self.WINDOW_HEIGHT - pipe['bottom_y'] - self.GROUND_HEIGHT))

        # Draw ground
        pygame.draw.rect(self.screen, self.BLACK, 
                       (0, self.WINDOW_HEIGHT - self.GROUND_HEIGHT, 
                        self.WINDOW_WIDTH, self.GROUND_HEIGHT))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, self.BLACK)
        self.screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Test the game
    game = FlappyBirdGame()
    running = True

    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # Jump
                else:
                    action = 0  # Do nothing
            else:
                action = 0

        state, reward, done = game.step(action)
        game.render()

        if done:
            print(f"Game Over! Final Score: {game.score}")
            pygame.time.wait(2000)
            running = False

    game.close()
