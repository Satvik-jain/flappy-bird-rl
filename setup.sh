#!/bin/bash
# Flappy Bird RL Setup Script

echo "Setting up Flappy Bird Reinforcement Learning Project..."

# Create virtual environment
python -m venv flappy_rl_env

# Activate virtual environment (Unix/Mac)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source flappy_rl_env/Scripts/activate
else
    source flappy_rl_env/bin/activate
fi

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! To run the project:"
echo "1. Activate environment: source flappy_rl_env/bin/activate (Unix/Mac) or flappy_rl_env\Scripts\activate (Windows)"
echo "2. Train agent: python flappy_bird_rl_agent.py"
echo "3. Test agent: python flappy_bird_rl_agent.py test"
