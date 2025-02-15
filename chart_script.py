import plotly.graph_objects as go
import json

# Parse the data
data = {
    "episodes": [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000], 
    "standard_dqn": [0, 2, 5, 8, 12, 15, 18, 22, 26, 30, 32, 35, 38, 42, 45, 48, 50, 52, 53, 54, 55], 
    "dueling_dqn": [0, 3, 7, 12, 16, 20, 24, 28, 33, 38, 42, 46, 50, 54, 58, 61, 64, 66, 67, 67, 66]
}

# Create the figure
fig = go.Figure()

# Add Standard DQN line
fig.add_trace(go.Scatter(
    x=data["episodes"],
    y=data["standard_dqn"],
    mode='lines',
    name='Standard DQN',
    line=dict(color='#1FB8CD', width=3),
    cliponaxis=False
))

# Add Dueling DQN line
fig.add_trace(go.Scatter(
    x=data["episodes"],
    y=data["dueling_dqn"],
    mode='lines',
    name='Dueling DQN',
    line=dict(color='#DB4545', width=3),
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Flappy Bird RL Training Progress',
    xaxis_title='Episodes',
    yaxis_title='Score',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_xaxes(range=[0, 10000])
fig.update_yaxes(range=[0, 80])

# Save the chart
fig.write_image("flappy_bird_training_progress.png")