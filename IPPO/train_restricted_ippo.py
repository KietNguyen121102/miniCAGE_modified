"""
IPPO training with 2 Blue agents having restricted action spaces.
Agent 0: Analyze, Remove, Restore (Defender)
Agent 1: Analyze, Decoy (Decoy Manager)
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from restricted_ippo_wrapper import RestrictedIPPOEnv

# Training Config
TOTAL_TIMESTEPS: int = 10_000_000
N_ENVS: int = 8
N_STEPS: int = 2048
BATCH_SIZE: int = 256
N_EPOCHS: int = 6
LEARNING_RATE: float = 0.0003
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
ENT_COEF: float = 0.05
VF_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5

USE_WANDB: bool = True
WANDB_PROJECT: str = "mini-cage-ippo"
GROUP_NAME: str = f"Restricted_IPPO_{TOTAL_TIMESTEPS}"

SAVE_DIR: Path = Path("ppo_models") / GROUP_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with variable action dimensions."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.features(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x, action=None):
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class RolloutBuffer:
    """Buffer for storing rollout data for one agent."""
    
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.reset()
    
    def reset(self):
        self.observations = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.pos = 0
    
    def add(self, obs, action, reward, done, value, log_prob):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
    
    def compute_returns_and_advantages(self, last_values, last_dones):
        """Compute GAE advantages and returns."""
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + GAMMA * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        
        returns = advantages + self.values
        return returns, advantages


def train_restricted_ippo():
    """Main IPPO training loop with restricted action spaces."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize W&B
    if USE_WANDB:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            group=GROUP_NAME,
            name=f"restricted_ippo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "total_timesteps": TOTAL_TIMESTEPS,
                "n_envs": N_ENVS,
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "lr": LEARNING_RATE,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE,
                "num_agents": 2,
                "agent_0_actions": "Analyze, Remove, Restore",
                "agent_1_actions": "Analyze, Decoy",
            },
            sync_tensorboard=False,
        )
    
    # Create environments
    envs = [RestrictedIPPOEnv(
        red_policy="bline", 
        max_steps=100, 
        remove_bugs=True, 
        seed=i,
    ) for i in range(N_ENVS)]
    
    # Get dimensions (different for each agent now!)
    num_agents = 2
    obs_dim = envs[0].observation_space.shape[0]  # 78
    action_dims = [envs[0].get_action_space(i).n for i in range(num_agents)]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Agent 0 action dim: {action_dims[0]} (Analyze, Remove, Restore)")
    print(f"Agent 1 action dim: {action_dims[1]} (Analyze, Decoy)")
    
    # Create 2 independent agent networks with DIFFERENT action dimensions
    agents = []
    optimizers = []
    for i in range(num_agents):
        agent = ActorCritic(obs_dim, action_dims[i], hidden_dim=256).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
        agents.append(agent)
        optimizers.append(optimizer)
    
    # Create rollout buffers for each agent
    buffers = [RolloutBuffer(N_STEPS, N_ENVS, obs_dim) for _ in range(num_agents)]
    
    # Initialize environments
    obs_list = [env.reset() for env in envs]
    current_obs = [
        np.array([obs_list[env_idx][agent_idx] for env_idx in range(N_ENVS)])
        for agent_idx in range(num_agents)
    ]
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (N_STEPS * N_ENVS)
    
    print(f"Starting training for {num_updates} updates...")
    start_time = time.time()
    
    episode_rewards = [[] for _ in range(num_agents)]
    episode_lengths = []
    current_episode_rewards = np.zeros((N_ENVS, num_agents))
    current_episode_lengths = np.zeros(N_ENVS)
    
    for update in range(1, num_updates + 1):
        # Reset buffers
        for buffer in buffers:
            buffer.reset()
        
        # Collect rollouts
        for step in range(N_STEPS):
            global_step += N_ENVS
            
            # Get actions from both agents
            actions_per_agent = []
            log_probs_per_agent = []
            values_per_agent = []
            
            for agent_idx in range(num_agents):
                obs_tensor = torch.FloatTensor(current_obs[agent_idx]).to(device)
                
                with torch.no_grad():
                    action, log_prob, _, value = agents[agent_idx].get_action_and_value(obs_tensor)
                
                actions_per_agent.append(action.cpu().numpy())
                log_probs_per_agent.append(log_prob.cpu().numpy())
                values_per_agent.append(value.cpu().numpy().flatten())
            
            # Step all environments
            next_obs_list = []
            rewards_list = []
            dones_list = []
            
            for env_idx in range(N_ENVS):
                env_actions = [actions_per_agent[agent_idx][env_idx] for agent_idx in range(num_agents)]
                obs, rewards, dones, infos = envs[env_idx].step(env_actions)
                
                next_obs_list.append(obs)
                rewards_list.append(rewards)
                dones_list.append(dones)
                
                # Track episode stats
                current_episode_rewards[env_idx] += rewards
                current_episode_lengths[env_idx] += 1
                
                if dones[0]:
                    for agent_idx in range(num_agents):
                        episode_rewards[agent_idx].append(current_episode_rewards[env_idx, agent_idx])
                    episode_lengths.append(current_episode_lengths[env_idx])
                    
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
                    
                    reset_obs = envs[env_idx].reset()
                    next_obs_list[-1] = reset_obs

            # Reshape observations
            next_obs = [
                np.array([next_obs_list[env_idx][agent_idx] for env_idx in range(N_ENVS)])
                for agent_idx in range(num_agents)
            ]
                        
            rewards = np.array([[rewards_list[env_idx][agent_idx] for env_idx in range(N_ENVS)] 
                               for agent_idx in range(num_agents)])
            
            dones = np.array([[dones_list[env_idx][agent_idx] for env_idx in range(N_ENVS)] 
                             for agent_idx in range(num_agents)])
            
            # Store in buffers
            for agent_idx in range(num_agents):
                buffers[agent_idx].add(
                    current_obs[agent_idx],
                    actions_per_agent[agent_idx],
                    rewards[agent_idx],
                    dones[agent_idx],
                    values_per_agent[agent_idx],
                    log_probs_per_agent[agent_idx]
                )
            
            current_obs = next_obs
        
        # Compute returns and advantages
        returns_and_advantages = []
        for agent_idx in range(num_agents):
            obs_tensor = torch.FloatTensor(current_obs[agent_idx]).to(device)
            with torch.no_grad():
                _, _, _, last_value = agents[agent_idx].get_action_and_value(obs_tensor)
            last_values = last_value.cpu().numpy().flatten()
            last_dones = dones[agent_idx]
            
            returns, advantages = buffers[agent_idx].compute_returns_and_advantages(last_values, last_dones)
            returns_and_advantages.append((returns, advantages))
        
        # Update each agent independently
        policy_losses_per_agent = [[] for _ in range(num_agents)]
        value_losses_per_agent = [[] for _ in range(num_agents)]
        entropy_losses_per_agent = [[] for _ in range(num_agents)]
        
        for agent_idx in range(num_agents):
            buffer = buffers[agent_idx]
            returns, advantages = returns_and_advantages[agent_idx]
            
            # Flatten batch
            b_obs = buffer.observations.reshape(-1, obs_dim)
            b_actions = buffer.actions.reshape(-1)
            b_log_probs = buffer.log_probs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = buffer.values.reshape(-1)
            
            # Normalize advantages
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # Training epochs
            batch_size = N_ENVS * N_STEPS
            indices = np.arange(batch_size)
            
            for epoch in range(N_EPOCHS):
                np.random.shuffle(indices)
                
                for start in range(0, batch_size, BATCH_SIZE):
                    end = start + BATCH_SIZE
                    mb_indices = indices[start:end]
                    
                    mb_obs = torch.FloatTensor(b_obs[mb_indices]).to(device)
                    mb_actions = torch.LongTensor(b_actions[mb_indices]).to(device)
                    mb_log_probs = torch.FloatTensor(b_log_probs[mb_indices]).to(device)
                    mb_advantages = torch.FloatTensor(b_advantages[mb_indices]).to(device)
                    mb_returns = torch.FloatTensor(b_returns[mb_indices]).to(device)
                    mb_values = torch.FloatTensor(b_values[mb_indices]).to(device)

                    # Forward pass
                    _, new_log_probs, entropy, new_values = agents[agent_idx].get_action_and_value(
                        mb_obs, mb_actions
                    )

                    # Policy loss
                    ratio = torch.exp(new_log_probs - mb_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss with clipping
                    new_values = new_values.flatten()
                    value_pred_clipped = mb_values + torch.clamp(
                        new_values - mb_values, -CLIP_RANGE, CLIP_RANGE
                    )
                    value_losses = (new_values - mb_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_loss
                    
                    policy_losses_per_agent[agent_idx].append(policy_loss.item())
                    value_losses_per_agent[agent_idx].append(value_loss.item())
                    entropy_losses_per_agent[agent_idx].append(entropy_loss.item())
                    
                    # Optimize
                    optimizers[agent_idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_idx].parameters(), MAX_GRAD_NORM)
                    optimizers[agent_idx].step()
        
        # Logging
        if update % 1 == 0:
            elapsed_time = time.time() - start_time
            fps = global_step / elapsed_time
            
            log_dict = {
                "time/fps": fps,
                "time/total_timesteps": global_step,
            }
            
            if len(episode_rewards[0]) > 0:
                recent_rewards_0 = episode_rewards[0][-100:]
                recent_rewards_1 = episode_rewards[1][-100:]
                recent_lengths = episode_lengths[-100:]
                
                log_dict["rollout/ep_rew_mean"] = np.mean(recent_rewards_0)
                log_dict["rollout/ep_len_mean"] = np.mean(recent_lengths)
                log_dict["rollout/agent_0_defender/ep_rew_mean"] = np.mean(recent_rewards_0)
                log_dict["rollout/agent_1_decoy/ep_rew_mean"] = np.mean(recent_rewards_1)
                
                print(f"Update {update} | Reward: {np.mean(recent_rewards_0):.2f} | FPS: {fps:.0f}")
            
            for agent_idx in range(num_agents):
                if len(policy_losses_per_agent[agent_idx]) > 0:
                    role = "defender" if agent_idx == 0 else "decoy"
                    log_dict[f"train/{role}/policy_loss"] = np.mean(policy_losses_per_agent[agent_idx])
                    log_dict[f"train/{role}/value_loss"] = np.mean(value_losses_per_agent[agent_idx])
                    log_dict[f"train/{role}/entropy_loss"] = np.mean(entropy_losses_per_agent[agent_idx])
            
            if USE_WANDB:
                wandb.log(log_dict, step=global_step)
        
        # Save models
        if update % 100 == 0:
            for agent_idx in range(num_agents):
                role = "defender" if agent_idx == 0 else "decoy"
                save_path = SAVE_DIR / f"agent_{agent_idx}_{role}_update_{update}.pt"
                torch.save({
                    'model_state_dict': agents[agent_idx].state_dict(),
                    'optimizer_state_dict': optimizers[agent_idx].state_dict(),
                    'update': update,
                }, save_path)
            print(f"Models saved at update {update}")
    
    # Final save
    for agent_idx in range(num_agents):
        role = "defender" if agent_idx == 0 else "decoy"
        save_path = SAVE_DIR / f"agent_{agent_idx}_{role}_final.pt"
        torch.save({
            'model_state_dict': agents[agent_idx].state_dict(),
            'update': num_updates,
        }, save_path)
    
    print("\nTraining completed!")
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    train_restricted_ippo()