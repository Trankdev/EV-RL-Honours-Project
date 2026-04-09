# src/agents/DQNagent.py
"""
DQN Agent Implementation
Supports a standard single-input Q-network (for Project 1)
"""

import os
import numpy as np

# Must be set before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import parl


# ============================================================================
# Standard Q-Network (aligned with Project 1)
# ============================================================================
class StandardQNetwork(parl.Model):
    """
    Standard Q-Network - Project 1 style
    
    Args:
        obs_dim: Observation dimension (Project 1: 68 = 8 (phase) + 12×5 (lanes))
        act_dim: Action dimension (Project 1: 8 phases)
        hidden_dim: Hidden layer dimension (Project 1: 20)
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=20):
        super(StandardQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs):
        """
        Forward pass
        
        Args:
            obs: torch.Tensor, shape (batch, obs_dim)
        
        Returns:
            Q: torch.Tensor, shape (batch, act_dim)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


# Alias: maintain compatibility
QNetwork = StandardQNetwork


# ============================================================================
# DQN Agent
# ============================================================================
class DQNAgent(parl.Agent):
    """
    DQN Agent - Project 1 style
    
    Args:
        algorithm: PARL DQN algorithm object
        obs_dim: Observation dimension
        act_dim: Action dimension
        epsilon: Initial exploration rate (Project 1: 1.0)
        epsilon_decay: Exploration decay (Project 1: exp(-0.0002) ≈ 0.9998)
        epsilon_min: Minimum exploration rate (Project 1: 0.01)
        grad_clip: Gradient clipping threshold
    """
    def __init__(self, algorithm, network_type='standard',
                 obs_dim=None, traffic_state_shape=None, phase_dim=None,
                 act_dim=8,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 grad_clip=10.0):
        super(DQNAgent, self).__init__(algorithm)
        
        self.network_type = network_type
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Exploration strategy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Gradient clipping
        self.grad_clip = grad_clip
    
    def sample(self, obs):
        """
        Epsilon-greedy sampling
        
        Args:
            obs: numpy array, shape (obs_dim,)
        
        Returns:
            action: int
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            device = next(self.alg.model.parameters()).device
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            pred_q = self.predict(obs)
            return int(pred_q.argmax())
    
    def predict(self, obs):
        """
        Predict Q-values
        
        Args:
            obs: torch.Tensor
        
        Returns:
            Q: torch.Tensor
        """
        return self.alg.predict(obs)
    
    def learn(self, obs, act, reward, next_obs, terminal):
        """
        Update model (based on Project 1 training process)
        
        Args:
            obs: numpy array, (batch, obs_dim)
            act: numpy array, (batch,)
            reward: numpy array, (batch,) or (batch, 1)
            next_obs: numpy array, (batch, obs_dim)
            terminal: numpy array, (batch,) or (batch, 1)
        
        Returns:
            loss: float
        """
        # Automatically get model device to ensure tensors match device
        device = next(self.alg.model.parameters()).device
        obs      = torch.FloatTensor(obs).to(device)
        act      = torch.LongTensor(act).to(device)
        reward   = torch.FloatTensor(reward).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        terminal = torch.FloatTensor(terminal).to(device)
        
        # Ensure correct shapes
        if reward.dim() == 1:
            reward = reward.unsqueeze(1)
        if terminal.dim() == 1:
            terminal = terminal.unsqueeze(1)
        if act.dim() == 2:
            act = act.squeeze(1)
        
        # 1. Compute target Q-values
        with torch.no_grad():
            next_q = self.alg.target_model(next_obs)
            max_next_q = torch.max(next_q, dim=1, keepdim=True)[0]
            target = reward + self.alg.gamma * max_next_q * (1 - terminal)
        
        # 2. Compute current Q-values
        current_q = self.alg.model(obs)
        current_q_acted = current_q.gather(1, act.unsqueeze(1))
        
        # 3. Compute MSE loss
        loss = F.mse_loss(current_q_acted, target)
        
        # 4. Backpropagation
        self.alg.optimizer.zero_grad()
        loss.backward()
        
        # 5. Gradient clipping (optional)
        if self.grad_clip > 0:
            clip_grad_norm_(self.alg.model.parameters(), self.grad_clip)
        
        # 6. Parameter update
        self.alg.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return float(loss.item())
    
    def decay_epsilon(self):
        """Manually decay epsilon (called per episode)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay