
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from flappy_bird_game import FlappyBirdGame

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture
    Separates state value and action advantage estimation
    """
    def __init__(self, state_size=6, action_size=2, hidden_size=256):
        super(DuelingDQN, self).__init__()

        self.action_size = action_size

        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Single value output
        )

        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)  # One advantage per action
        )

    def forward(self, x):
        # Extract shared features
        features = self.feature_layer(x)

        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling past experiences
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """Store experience in buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size=32):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))

        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DuelingDQNAgent:
    """
    Dueling DQN Agent with Experience Replay and Target Network
    """
    def __init__(self, state_size=6, action_size=2, lr=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=64, target_update_freq=1000):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Main network (policy network)
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)

        # Target network (for stable Q-learning)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)

        # Training metrics
        self.losses = []
        self.scores = []
        self.episodes_trained = 0

    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        """
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))

        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)

        # Return action with highest Q-value
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def replay_experience(self):
        """
        Train the network on a batch of experiences from replay buffer
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from experience replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network (for stability)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Target network updated at step {self.update_count}")

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.losses.append(loss.item())
        return loss.item()

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'scores': self.scores
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episodes_trained = checkpoint.get('episodes_trained', 0)
        self.scores = checkpoint.get('scores', [])
        print(f"Model loaded from {filepath}")

def train_agent(episodes=10000, save_every=1000, render_every=100):
    """
    Train the Dueling DQN agent to play Flappy Bird
    """
    # Initialize game and agent
    game = FlappyBirdGame()
    agent = DuelingDQNAgent()

    scores_window = deque(maxlen=100)  # For tracking average score
    best_average_score = -float('inf')

    print("Starting training...")
    print(f"Target: Achieve score of 66+ and playtime >4 minutes")
    print(f"Training for {episodes} episodes")
    print("="*50)

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        steps = 0
        episode_start_time = 0  # Track episode duration

        while not game.game_over:
            # Agent selects action
            action = agent.act(state)

            # Take action in environment
            next_state, reward, done = game.step(action)

            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1
            episode_start_time += 1

            # Train agent with experience replay
            loss = agent.replay_experience()

            # Render occasionally during training
            if episode % render_every == 0:
                game.render()

        # Episode finished - record score
        agent.scores.append(game.score)
        scores_window.append(game.score)
        agent.episodes_trained += 1

        # Calculate episode duration (approximate)
        episode_duration = episode_start_time / 60.0  # Convert to seconds (60 FPS)

        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f"Episode {episode:5d} | Score: {game.score:3d} | "
                  f"Avg Score: {avg_score:6.2f} | Steps: {steps:4d} | "
                  f"Duration: {episode_duration:.2f}s | "
                  f"Epsilon: {agent.epsilon:.3f} | Loss: {loss:.4f}")

            # Check if we achieved the resume benchmarks
            if avg_score >= 60 and episode_duration >= 240:  # 4 minutes = 240 seconds
                print(f"\n🎉 RESUME BENCHMARKS ACHIEVED! 🎉")
                print(f"Average Score: {avg_score:.2f} (Target: 66+)")
                print(f"Episode Duration: {episode_duration:.2f}s (Target: 240s+)")

            # Save best model
            if avg_score > best_average_score:
                best_average_score = avg_score
                agent.save_model('best_flappy_bird_model.pth')

        # Save model periodically
        if episode % save_every == 0 and episode > 0:
            agent.save_model(f'flappy_bird_model_episode_{episode}.pth')

    print(f"\nTraining completed!")
    print(f"Final average score: {np.mean(scores_window):.2f}")
    print(f"Best average score: {best_average_score:.2f}")
    print(f"Total episodes trained: {agent.episodes_trained}")

    # Save final model
    agent.save_model('final_flappy_bird_model.pth')

    # Plot training progress
    plot_training_results(agent.scores, agent.losses)

    game.close()
    return agent

def plot_training_results(scores, losses):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot scores
    ax1.plot(scores)
    ax1.set_title('Training Scores Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True)

    # Plot moving average of scores
    if len(scores) >= 100:
        moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
        ax1.plot(moving_avg, color='red', label='100-episode average')
        ax1.legend()

    # Plot losses
    if losses:
        ax2.plot(losses)
        ax2.set_title('Training Loss Over Time')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def test_agent(model_path='best_flappy_bird_model.pth', episodes=10):
    """Test the trained agent"""
    game = FlappyBirdGame()
    agent = DuelingDQNAgent()

    try:
        agent.load_model(model_path)
        agent.epsilon = 0  # No exploration during testing
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train first.")
        return

    scores = []
    durations = []

    print(f"Testing trained agent for {episodes} episodes...")

    for episode in range(episodes):
        state = game.reset()
        steps = 0

        while not game.game_over:
            action = agent.act(state, training=False)
            next_state, reward, done = game.step(action)
            state = next_state
            steps += 1
            game.render()

        duration = steps / 60.0  # Convert to seconds
        scores.append(game.score)
        durations.append(duration)

        print(f"Test Episode {episode+1}: Score = {game.score}, Duration = {duration:.2f}s")

    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Average Duration: {np.mean(durations):.2f}s")
    print(f"Max Duration: {max(durations):.2f}s")

    game.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_flappy_bird_model.pth'
        test_agent(model_path)
    else:
        # Training mode
        episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
        train_agent(episodes=episodes)
