"""
Single Agent PPO training script with Stable Baselines 3 optimizations.
This includes vectorized environments, observation normalization, orthogonal
initialization, and a linear learning rate schedule.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
import random
from collections import deque
from multiprocessing import Process, set_start_method
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from single_agent_gym_wrapper import MiniCageBlue

# Training Config
NUM_RUNS: int = 1  # Number of parallel training runs
TOTAL_TIMESTEPS: int = 10000000
N_ENVS: int = 8  # Number of parallel environments (same as IPPO)
N_STEPS: int = 2048  # Steps per rollout
BATCH_SIZE: int = 256
N_EPOCHS: int = 6
LEARNING_RATE: float = 0.002
GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
CLIP_RANGE_VF: float = 0.2  # Value function clipping (set to None to disable)
ENT_COEF: float = 0.05
VF_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5
TARGET_KL: float = None  # Early stopping on KL divergence (None to disable)
ANNEAL_LR: bool = True  # Linearly anneal learning rate

USE_WANDB: bool = True
WANDB_PROJECT: str = "mini-cage-ippo"
GROUP_NAME: str = f"SB3_PPO_{TOTAL_TIMESTEPS}"

# Save to parent directory (mini_CAGE/ppo_models)
SAVE_DIR: Path = Path(__file__).parent.parent / "ppo_models" / GROUP_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO, inspired by SB3."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Shared feature extractor (SB3 default is 2 layers of 64)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Policy head
        self.action_net = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def get_value(self, x):
        return self.value_head(self.value_net(x))

    def get_action_and_value(self, x, action=None):
        policy_latent = self.policy_net(x)
        action_logits = self.action_net(policy_latent)
        value = self.value_head(self.value_net(x))

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


def make_env(seed, red_policy="bline", max_steps=100, remove_bugs=True):
    """Helper function to create a single environment."""
    def _init():
        env = MiniCageBlue(red_policy=red_policy, max_steps=max_steps, remove_bugs=remove_bugs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init


def train_sb3_ppo(run_idx: int):
    """Main single agent PPO training loop with SB3 optimizations.
    
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
            name=f"sb3_ppo_{time_tag}_{run_idx}",
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
                "ent_coef": ENT_COEF,
                "vf_coef": VF_COEF,
                "max_grad_norm": MAX_GRAD_NORM,
                "anneal_lr": ANNEAL_LR,
                "num_agents": 1,
                "algorithm": "PPO (SB3-style)",
                "seed": seed,
            },
            sync_tensorboard=False,
        )
    
    # Create vectorized environments
    vec_env = DummyVecEnv([make_env(seed + i) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, gamma=GAMMA)

    # Get dimensions
    obs_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create agent network (SB3 default: 2 layers of 64 hidden units)
    agent = ActorCritic(obs_dim, action_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Create rollout buffer
    buffer = RolloutBuffer(N_STEPS, N_ENVS, obs_dim)
    
    # Initialize environments
    current_obs = vec_env.reset()
    
    # Diagnostic logging
    print(f"\n{'='*60}")
    print(f"INITIALIZATION DIAGNOSTICS:")
    print(f"{'='*60}")
    print(f"Initial obs shape: {current_obs.shape}")
    print(f"Initial obs mean: {current_obs.mean():.4f}, std: {current_obs.std():.4f}")
    print(f"Initial obs min: {current_obs.min():.4f}, max: {current_obs.max():.4f}")
    print(f"Adam epsilon: 1e-5 (matching SB3)")
    print(f"Hidden dim: 64 (matching SB3)")
    print(f"{'='*60}\n")
    
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (N_STEPS * N_ENVS)
    
    print(f"Starting training for {num_updates} updates...")
    start_time = time.time()
    
    # For logging
    ep_rew_buffer = deque(maxlen=100)
    ep_len_buffer = deque(maxlen=100)
    total_episodes_completed = 0
    
    for update in range(1, num_updates + 1):
        # Linear learning rate annealing
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lr_now

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
            next_obs, rewards, dones, infos = vec_env.step(action_np)
            
            # Log episode stats from infos
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    ep_rew_buffer.append(info["episode"]["r"])
                    ep_len_buffer.append(info["episode"]["l"])
                    total_episodes_completed += 1

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
            last_value = agent.get_value(obs_tensor).cpu().numpy().flatten()
        last_dones = dones
        
        returns, advantages = buffer.compute_returns_and_advantages(last_value, last_dones)
        
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
            print(f"  Obs stats (from buffer) - mean: {b_obs.mean():.4f}, std: {b_obs.std():.4f}")
            print(f"  Reward stats (raw) - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}")
            print(f"  Value estimates - mean: {b_values.mean():.4f}, std: {b_values.std():.4f}")
            print(f"  Advantages (unnormalized) - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")
            if len(policy_losses) > 0:
                print(f"  Policy loss: {np.mean(policy_losses):.4f}")
                print(f"  Value loss: {np.mean(value_losses):.4f}")
                print(f"  Approx KL: {np.mean(approx_kls):.4f}")
        
        # Logging (matching IPPO style)
        if update % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = global_step / elapsed_time
            
            log_dict = {
                "time/fps": fps,
                "time/time_elapsed": elapsed_time,
                "time/total_timesteps": global_step,
                "episodes/total": total_episodes_completed,
            }
            
            # Rollout metrics
            if len(ep_rew_buffer) > 0:
                log_dict["rollout/ep_rew_mean"] = np.mean(ep_rew_buffer)
                log_dict["rollout/ep_len_mean"] = np.mean(ep_len_buffer)
                log_dict["rollout/ep_rew_std"] = np.std(ep_rew_buffer)

                # Print summary (SB3-style)
                mean_reward = np.mean(ep_rew_buffer)
                mean_length = np.mean(ep_len_buffer)
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
                log_dict["train/learning_rate"] = optimizer.param_groups[0]["lr"]
            
            if USE_WANDB:
                wandb.log(log_dict, step=global_step)
        
        # Save models and VecNormalize stats
        if update % 100 == 0:
            save_path = SAVE_DIR / f"agent_update_{update}.pt"
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'global_step': global_step,
            }, save_path)
            vec_env.save(SAVE_DIR / f"vec_normalize_{update}.pkl")
            print(f"Model and VecNormalize stats saved at update {update}")
    
    # Final save
    save_path = SAVE_DIR / "agent_final.pt"
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': num_updates,
        'global_step': global_step,
    }, save_path)
    vec_env.save(SAVE_DIR / "vec_normalize_final.pkl")
    
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
        p = Process(target=train_sb3_ppo, args=(idx,), daemon=False)
        p.start()
        processes.append(p)

    # Wait for all workers to complete
    for p in processes:
        p.join()

    print("\n All runs finished!")
