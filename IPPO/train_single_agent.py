"""
Single Agent PPO training matching the IPPO implementation structure.
Uses the same hyperparameters and logging format for fair comparison.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
import random
from multiprocessing import Process, set_start_method
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Add parent directory to path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from single_agent_gym_wrapper import MiniCageBlue

# Training Config
NUM_RUNS: int = 1  # Number of parallel training runs
TOTAL_TIMESTEPS: int = 2000000
N_ENVS: int = 8  # Number of parallel environments (same as IPPO)
N_STEPS: int = 2048  # Steps per rollout
BATCH_SIZE: int = 256
N_EPOCHS: int = 6
LEARNING_RATE: float = 0.0003
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
CLIP_RANGE_VF: float = 0.2  # Value function clipping (set to None to disable)
ENT_COEF: float = 0.05
VF_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5
TARGET_KL: float = None  # Early stopping on KL divergence (None to disable)

USE_WANDB: bool = True
WANDB_PROJECT: str = "mini-cage-ippo"
GROUP_NAME: str = f"SingleAgent_PPO_{TOTAL_TIMESTEPS}"

# Save to parent directory (mini_CAGE/ppo_models)
SAVE_DIR: Path = Path(__file__).parent.parent / "ppo_models" / GROUP_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Policy head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Value head
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
    """Buffer for storing rollout data."""
    
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


def train_single_agent(run_idx: int):
    """Main single agent PPO training loop.
    
    Args:
        run_idx: Index for this training run, used as seed for reproducibility
    """
    
    # Set random seeds for reproducibility
    seed = run_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {run_idx}: Using device: {device}")
    print(f"Run {run_idx}: Global seed: {seed}")
    
    # Initialize W&B
    if USE_WANDB:
        import wandb
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            project=WANDB_PROJECT,
            group=GROUP_NAME,
            name=f"single_agent_ppo_{time_tag}_{run_idx}",
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
                "clip_range_vf": CLIP_RANGE_VF,
                "num_agents": 1,
                "algorithm": "PPO (Single Agent)",
                "seed": seed,
            },
            sync_tensorboard=False,
        )
    
    # Create environments
    envs = []
    for i in range(N_ENVS):
        env = MiniCageBlue(red_policy="bline", max_steps=100, remove_bugs=True)
        env_seed = seed + i
        env.action_space.seed(env_seed)
        env.observation_space.seed(env_seed)
        envs.append(env)
    
    # Get dimensions
    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create agent network (matching IPPO architecture: 256 hidden units)
    agent = ActorCritic(obs_dim, action_dim, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Create rollout buffer
    buffer = RolloutBuffer(N_STEPS, N_ENVS, obs_dim)
    
    # Initialize environments
    current_obs = np.array([envs[i].reset(seed=seed + i)[0] for i in range(N_ENVS)])
    
    # Diagnostic logging
    print(f"\n{'='*60}")
    print(f"INITIALIZATION DIAGNOSTICS:")
    print(f"{'='*60}")
    print(f"Initial obs shape: {current_obs.shape}")
    print(f"Initial obs mean: {current_obs.mean():.4f}, std: {current_obs.std():.4f}")
    print(f"Initial obs min: {current_obs.min():.4f}, max: {current_obs.max():.4f}")
    print(f"Adam epsilon: 1e-5 (matching IPPO)")
    print(f"Hidden dim: 256 (matching IPPO)")
    print(f"{'='*60}\n")
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (N_STEPS * N_ENVS)
    
    print(f"Starting training for {num_updates} updates...")
    start_time = time.time()
    
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = np.zeros(N_ENVS)
    current_episode_lengths = np.zeros(N_ENVS)
    
    for update in range(1, num_updates + 1):
        # Reset buffer
        buffer.reset()
        
        # Collect rollouts
        for step in range(N_STEPS):
            global_step += N_ENVS
            
            # Get action from agent
            obs_tensor = torch.FloatTensor(current_obs).to(device)
            
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)
            
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy().flatten()
            
            # Step all environments
            next_obs_list = []
            rewards_list = []
            dones_list = []
            
            for env_idx in range(N_ENVS):
                obs, reward, done, truncated, info = envs[env_idx].step(action_np[env_idx])
                
                next_obs_list.append(obs)
                rewards_list.append(reward)
                dones_list.append(done or truncated)
                
                # Track episode stats
                current_episode_rewards[env_idx] += reward
                current_episode_lengths[env_idx] += 1
                
                if done or truncated:
                    # Log completed episode stats
                    episode_rewards.append(current_episode_rewards[env_idx])
                    episode_lengths.append(current_episode_lengths[env_idx])
                    
                    # Reset episode tracking
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
                    
                    # Reset environment
                    reset_obs, _ = envs[env_idx].reset()
                    next_obs_list[-1] = reset_obs
            
            next_obs = np.array(next_obs_list)
            rewards = np.array(rewards_list)
            dones = np.array(dones_list)
            
            # Store in buffer
            buffer.add(
                current_obs,
                action_np,
                rewards,
                dones,
                value_np,
                log_prob_np
            )
            
            current_obs = next_obs
        
        # Compute returns and advantages
        obs_tensor = torch.FloatTensor(current_obs).to(device)
        with torch.no_grad():
            _, _, _, last_value = agent.get_action_and_value(obs_tensor)
        last_values = last_value.cpu().numpy().flatten()
        last_dones = dones
        
        returns, advantages = buffer.compute_returns_and_advantages(last_values, last_dones)
        
        # Initialize loss tracking
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        
        # Flatten batch
        b_obs = buffer.observations.reshape(-1, obs_dim)
        b_actions = buffer.actions.reshape(-1)
        b_log_probs = buffer.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer.values.reshape(-1)
        
        # Training epochs
        batch_size = N_ENVS * N_STEPS
        indices = np.arange(batch_size)
        
        continue_training = True
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
                
                # Normalize advantages per mini-batch (like SB3)
                if len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Forward pass
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    mb_obs, mb_actions
                )
                
                # Policy loss
                ratio = torch.exp(new_log_probs - mb_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with optional clipping (matching SB3 exactly)
                new_values = new_values.flatten()
                
                if CLIP_RANGE_VF is not None:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = mb_values + torch.clamp(
                        new_values - mb_values, -CLIP_RANGE_VF, CLIP_RANGE_VF
                    )
                else:
                    # No clipping
                    values_pred = new_values
                
                # Value loss using the TD(gae_lambda) target
                value_loss = ((mb_returns - values_pred) ** 2).mean()
                
                # Entropy loss favor exploration (matching SB3 sign convention)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + ENT_COEF * entropy_loss + VF_COEF * value_loss
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    log_ratio = new_log_probs - mb_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    clip_fraction = (torch.abs(ratio - 1) > CLIP_RANGE).float().mean().item()
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
                
                # Early stopping on KL divergence
                if TARGET_KL is not None and approx_kl > 1.5 * TARGET_KL:
                    continue_training = False
                    print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.2f}")
                    break
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            
            if not continue_training:
                break
        
        # Diagnostic logging (every 10 updates)
        if update % 10 == 0:
            print(f"\nUpdate {update} diagnostics:")
            print(f"  Obs stats - mean: {b_obs.mean():.4f}, std: {b_obs.std():.4f}")
            print(f"  Reward stats - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}")
            print(f"  Value estimates - mean: {b_values.mean():.4f}, std: {b_values.std():.4f}")
            print(f"  Advantages (normalized) - mean: {b_advantages.mean():.4f}, std: {b_advantages.std():.4f}")
            if len(policy_losses) > 0:
                print(f"  Policy loss: {np.mean(policy_losses):.4f}")
                print(f"  Value loss: {np.mean(value_losses):.4f}")
                print(f"  Approx KL: {np.mean(approx_kls):.4f}")
        
        # Logging (matching IPPO style)
        if update % 1 == 0:
            total_episodes = len(episode_lengths)
            
            elapsed_time = time.time() - start_time
            fps = global_step / elapsed_time
            
            log_dict = {
                "time/fps": fps,
                "time/time_elapsed": elapsed_time,
                "time/total_timesteps": global_step,
                "episodes/total": total_episodes,
            }
            
            # Rollout metrics
            if len(episode_rewards) > 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
                
                log_dict["rollout/ep_rew_mean"] = np.mean(recent_rewards)
                log_dict["rollout/ep_len_mean"] = np.mean(recent_lengths)
                log_dict["rollout/ep_rew_std"] = np.std(recent_rewards)
                
                # Print summary (SB3-style)
                mean_reward = np.mean(recent_rewards)
                mean_length = np.mean(recent_lengths)
                print(f"------------------------------------------")
                print(f"| rollout/                |              |")
                print(f"|    ep_len_mean          | {mean_length:<12.1f} |")
                print(f"|    ep_rew_mean          | {mean_reward:<12.2f} |")
                print(f"| time/                   |              |")
                print(f"|    fps                  | {fps:<12.0f} |")
                print(f"|    total_timesteps      | {global_step:<12} |")
                print(f"------------------------------------------")
            
            # Training metrics
            if len(policy_losses) > 0:
                log_dict["train/policy_loss"] = np.mean(policy_losses)
                log_dict["train/value_loss"] = np.mean(value_losses)
                log_dict["train/entropy_loss"] = np.mean(entropy_losses)
                log_dict["train/approx_kl"] = np.mean(approx_kls)
                log_dict["train/clip_fraction"] = np.mean(clip_fractions)
                log_dict["train/learning_rate"] = LEARNING_RATE
            
            if USE_WANDB:
                wandb.log(log_dict, step=global_step)
        
        # Save models
        if update % 100 == 0:
            save_path = SAVE_DIR / f"agent_update_{update}.pt"
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'global_step': global_step,
            }, save_path)
            print(f"Model saved at update {update}")
    
    # Final save
    save_path = SAVE_DIR / "agent_final.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': num_updates,
        'global_step': global_step,
    }, save_path)
    
    print(f"\nRun {run_idx}: Training completed!")
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    try:
        set_start_method("spawn")  # does nothing if already set
    except RuntimeError:
        pass

    START_IDX = 21
    processes: list[Process] = []
    for idx in range(START_IDX, START_IDX + NUM_RUNS):
        p = Process(target=train_single_agent, args=(idx,), daemon=False)
        p.start()
        processes.append(p)

    # Wait for all workers to complete
    for p in processes:
        p.join()

    print("\n All runs finished!")

