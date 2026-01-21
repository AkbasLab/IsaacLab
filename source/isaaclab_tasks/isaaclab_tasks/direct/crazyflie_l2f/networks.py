"""
Actor-Critic Network Matching Learning-to-Fly Architecture

This module provides PyTorch implementations of the actor and critic networks
that EXACTLY match the rl_tools architecture used in learning-to-fly:

Actor: 146 inputs -> 64 hidden (tanh) -> 64 hidden (tanh) -> 4 outputs (tanh)
Critic: 146 inputs -> 64 hidden (tanh) -> 64 hidden (tanh) -> 1 output (linear)

The architecture must match exactly for successful firmware deployment.

References:
- learning-to-fly/src/config/actor_and_critic.h
- learning-to-fly/training/ppo_gpu/ppo.py
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class L2FActorNetwork(nn.Module):
    """Actor network matching learning-to-fly architecture.
    
    Architecture:
        Input (146) -> Linear -> Tanh -> Linear -> Tanh -> Linear -> Tanh -> Output (4)
                       |  64  |        |  64  |        |   4   |
    
    The tanh activation on the output layer is critical as it bounds
    the actions to [-1, 1] which matches the firmware expectations.
    """
    
    # Network dimensions (MUST match firmware exactly)
    OBSERVATION_DIM = 146  # 18 core + 128 action history
    HIDDEN_DIM = 64
    ACTION_DIM = 4
    
    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        action_dim: int = ACTION_DIM,
        init_std: float = 0.3,
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Network layers (matching rl_tools naming: input_layer, hidden_layer_0, output_layer)
        self.fc1 = nn.Linear(observation_dim, hidden_dim)  # input_layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)       # hidden_layer_0
        self.fc3 = nn.Linear(hidden_dim, action_dim)       # output_layer
        
        # Learnable log standard deviation for PPO (per-action)
        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(init_std)))
        
        # Initialize weights using orthogonal initialization (common in RL)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
        
        # Smaller scale for output layer (helps with initial exploration)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning mean actions.
        
        Args:
            obs: Observations of shape (batch_size, observation_dim)
            
        Returns:
            Mean actions of shape (batch_size, action_dim) bounded to [-1, 1]
        """
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # Bounded to [-1, 1]
        return mean
    
    def forward_with_std(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and standard deviation.
        
        Args:
            obs: Observations of shape (batch_size, observation_dim)
            
        Returns:
            Tuple of (mean, std) for sampling actions
        """
        mean = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.
        
        Args:
            obs: Observations of shape (batch_size, observation_dim)
            
        Returns:
            Tuple of (actions, log_probs)
        """
        mean, std = self.forward_with_std(obs)
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        actions = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds (correction for tanh squashing)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return actions, log_prob
    
    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean of policy).
        
        Use this for evaluation/deployment.
        """
        return self.forward(obs)
    
    def export_weights(self) -> dict:
        """Export weights in a format suitable for firmware conversion.
        
        Returns weights as numpy arrays matching the layer naming expected
        by the firmware export pipeline.
        """
        return {
            'actor_l1_w': self.fc1.weight.detach().cpu().numpy(),
            'actor_l1_b': self.fc1.bias.detach().cpu().numpy(),
            'actor_l2_w': self.fc2.weight.detach().cpu().numpy(),
            'actor_l2_b': self.fc2.bias.detach().cpu().numpy(),
            'actor_l3_w': self.fc3.weight.detach().cpu().numpy(),
            'actor_l3_b': self.fc3.bias.detach().cpu().numpy(),
            'log_std': self.log_std.detach().cpu().numpy(),
        }


class L2FCriticNetwork(nn.Module):
    """Critic network matching learning-to-fly architecture.
    
    Architecture:
        Input (146) -> Linear -> Tanh -> Linear -> Tanh -> Linear -> Output (1)
                       |  64  |        |  64  |        |   1   |
    
    Note: The output has no activation (linear) for unbounded value estimation.
    """
    
    def __init__(
        self,
        observation_dim: int = L2FActorNetwork.OBSERVATION_DIM,
        hidden_dim: int = L2FActorNetwork.HIDDEN_DIM,
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate.
        
        Args:
            obs: Observations of shape (batch_size, observation_dim)
            
        Returns:
            Value estimates of shape (batch_size, 1)
        """
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)  # No activation (unbounded)
        return value


class L2FActorCritic(nn.Module):
    """Combined Actor-Critic network for PPO training.
    
    This class wraps both the actor and critic networks and provides
    a unified interface for PPO training.
    """
    
    def __init__(
        self,
        observation_dim: int = L2FActorNetwork.OBSERVATION_DIM,
        hidden_dim: int = L2FActorNetwork.HIDDEN_DIM,
        action_dim: int = L2FActorNetwork.ACTION_DIM,
        init_std: float = 0.3,
    ):
        super().__init__()
        
        self.actor = L2FActorNetwork(
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            init_std=init_std,
        )
        self.critic = L2FCriticNetwork(
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both action mean and value estimate.
        
        Args:
            obs: Observations of shape (batch_size, observation_dim)
            
        Returns:
            Tuple of (action_mean, value)
        """
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def get_action_and_value(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value for PPO.
        
        Args:
            obs: Observations
            deterministic: If True, return mean action instead of sampling
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        value = self.critic(obs)
        
        if deterministic:
            action = self.actor.get_deterministic_action(obs)
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
        else:
            action, log_prob = self.actor.sample(obs)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        mean, std = self.actor.forward_with_std(obs)
        value = self.critic(obs)
        
        # Create distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Compute log probability
        # Need to inverse tanh to get the pre-squashing value
        # Using atanh with clipping for numerical stability
        actions_clipped = torch.clamp(actions, -0.999, 0.999)
        x_t = torch.atanh(actions_clipped)
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Entropy for exploration bonus
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy
    
    def export_actor_weights(self) -> dict:
        """Export actor weights for firmware deployment."""
        return self.actor.export_weights()


class RunningMeanStd:
    """Running mean and standard deviation for observation normalization.
    
    This matches the observation normalization used in PPO training
    and is required for successful sim-to-real transfer.
    """
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: torch.Tensor):
        """Update running statistics with new batch of data."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ):
        """Update from pre-computed moments (Welford's algorithm)."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observation using running statistics."""
        return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + self.epsilon)
    
    def to(self, device: torch.device) -> 'RunningMeanStd':
        """Move statistics to device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self
    
    def export(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Export mean and inverse std for firmware."""
        std_inv = 1.0 / torch.sqrt(self.var + self.epsilon)
        return self.mean.clone(), std_inv.clone()
