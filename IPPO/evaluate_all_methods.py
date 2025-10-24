"""
Comprehensive evaluation script for comparing:
1. Single Agent (SB3 PPO)
2. IPPO (2 agents with shared obs and full action space)
3. Restricted IPPO (2 agents with restricted action spaces)

This script evaluates performance and action distributions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the parent of mini_CAGE to path for imports
mini_cage_parent = os.path.dirname(parent_dir)
if mini_cage_parent not in sys.path:
    sys.path.insert(0, mini_cage_parent)

from stable_baselines3 import PPO

# Import from parent directory (mini_CAGE)
sys.path.insert(0, parent_dir)
from single_agent_gym_wrapper import MiniCageBlue
from minimal import HOSTS, BLUE_ACTIONS

# Import IPPO wrappers from current directory
from multi_agent_ippo_wrapper import IPPOSharedObsEnv
from restricted_ippo_wrapper import RestrictedIPPOEnv


# ═══════════════════════════════════════════════════════════════════════
# IPPO Model Definition (must match training)
# ═══════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
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
        
        # Note: Initialization is done in training, loaded models already have weights
    
    def forward(self, x):
        features = self.features(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x, action=None, deterministic=False):
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════

def evaluate_single_agent(
    model_path: str,
    n_episodes: int = 100,
    deterministic: bool = True,
    max_steps: int = 100,
) -> Dict:
    """Evaluate single agent (supports both .pt and .zip formats)."""
    print(f"\n{'='*60}")
    print("Evaluating Single Agent PPO")
    print(f"{'='*60}")
    
    # Create environment
    env = MiniCageBlue(red_policy="bline", max_steps=max_steps, remove_bugs=True)
    
    # Determine model format and load accordingly
    if model_path.endswith('.zip'):
        # SB3 format
        print("Loading SB3 model format...")
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        use_sb3 = True
    elif model_path.endswith('.pt'):
        # PyTorch format (matching IPPO)
        print("Loading PyTorch model format...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        model = ActorCritic(obs_dim, action_dim, hidden_dim=256).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        use_sb3 = False
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    action_counts = defaultdict(int)
    action_type_counts = defaultdict(int)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            if use_sb3:
                # SB3 model
                action, _states = model.predict(obs, deterministic=deterministic)
            else:
                # PyTorch model
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    if deterministic:
                        # Use argmax for deterministic actions
                        action_logits, _ = model.forward(obs_tensor)
                        action = torch.argmax(action_logits, dim=-1)
                    else:
                        action, _, _, _ = model.get_action_and_value(obs_tensor)
                action = action.item()
            
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track action distribution
            action_counts[int(action)] += 1
            
            # Get action type
            action_type = get_action_type_single(int(action))
            action_type_counts[action_type] += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes} - Mean Reward: {np.mean(episode_rewards[-20:]):.2f}")
    
    results = {
        "method": "Single Agent",
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "action_counts": dict(action_counts),
        "action_type_counts": dict(action_type_counts),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }
    
    print(f"\nSingle Agent Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.2f}")
    
    return results


def evaluate_ippo(
    model_paths: List[str],
    n_episodes: int = 100,
    deterministic: bool = True,
    max_steps: int = 100,
) -> Dict:
    """Evaluate IPPO (2 agents with shared obs)."""
    print(f"\n{'='*60}")
    print("Evaluating IPPO (Shared Observation)")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = IPPOSharedObsEnv(
        red_policy="bline",
        max_steps=max_steps,
        remove_bugs=True,
        action_resolution="sequential"
    )
    
    # Load models
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agents = []
    for model_path in model_paths:
        agent = ActorCritic(obs_dim, action_dim, hidden_dim=256).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        agents.append(agent)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    agent_action_counts = [defaultdict(int) for _ in range(2)]
    agent_action_type_counts = [defaultdict(int) for _ in range(2)]
    
    for ep in range(n_episodes):
        obs_list = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            actions = []
            for agent_idx, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs_list[agent_idx]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=deterministic)
                actions.append(action.item())
                
                # Track action distribution
                agent_action_counts[agent_idx][action.item()] += 1
                action_type = get_action_type_single(action.item())
                agent_action_type_counts[agent_idx][action_type] += 1
            
            obs_list, rewards, dones, infos = env.step(actions)
            
            episode_reward += rewards[0]  # Shared reward
            episode_length += 1
            done = dones[0]
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes} - Mean Reward: {np.mean(episode_rewards[-20:]):.2f}")
    
    results = {
        "method": "IPPO",
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "agent_0_action_counts": dict(agent_action_counts[0]),
        "agent_1_action_counts": dict(agent_action_counts[1]),
        "agent_0_action_type_counts": dict(agent_action_type_counts[0]),
        "agent_1_action_type_counts": dict(agent_action_type_counts[1]),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }
    
    print(f"\nIPPO Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.2f}")
    
    return results


def evaluate_restricted_ippo(
    model_paths: List[str],
    n_episodes: int = 100,
    deterministic: bool = True,
    max_steps: int = 100,
) -> Dict:
    """Evaluate Restricted IPPO (2 agents with restricted action spaces)."""
    print(f"\n{'='*60}")
    print("Evaluating Restricted IPPO")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = RestrictedIPPOEnv(
        red_policy="bline",
        max_steps=max_steps,
        remove_bugs=True,
    )
    
    # Load models
    obs_dim = env.observation_space.shape[0]
    action_dims = [env.get_action_space(i).n for i in range(2)]
    
    agents = []
    for agent_idx, model_path in enumerate(model_paths):
        agent = ActorCritic(obs_dim, action_dims[agent_idx], hidden_dim=256).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        agents.append(agent)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    agent_action_counts = [defaultdict(int) for _ in range(2)]
    agent_action_type_counts = [defaultdict(int) for _ in range(2)]
    
    for ep in range(n_episodes):
        obs_list = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            actions = []
            for agent_idx, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs_list[agent_idx]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=deterministic)
                actions.append(action.item())
                
                # Map to environment action
                if agent_idx == 0:
                    env_action = env.agent_0_actions[action.item()]
                else:
                    env_action = env.agent_1_actions[action.item()]
                
                # Track action distribution
                agent_action_counts[agent_idx][env_action] += 1
                action_type = get_action_type_single(env_action)
                agent_action_type_counts[agent_idx][action_type] += 1
            
            obs_list, rewards, dones, infos = env.step(actions)
            
            episode_reward += rewards[0]  # Shared reward
            episode_length += 1
            done = dones[0]
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes} - Mean Reward: {np.mean(episode_rewards[-20:]):.2f}")
    
    results = {
        "method": "Restricted IPPO",
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "agent_0_action_counts": dict(agent_action_counts[0]),
        "agent_1_action_counts": dict(agent_action_counts[1]),
        "agent_0_action_type_counts": dict(agent_action_type_counts[0]),
        "agent_1_action_type_counts": dict(agent_action_type_counts[1]),
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }
    
    print(f"\nRestricted IPPO Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:.2f}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def get_action_type_single(action: int) -> str:
    """Get action type from action index."""
    if action == 0:
        return "sleep"
    elif 1 <= action <= 13:
        return "analyse"
    elif 14 <= action <= 26:
        return "decoy"
    elif 27 <= action <= 39:
        return "remove"
    elif 40 <= action <= 52:
        return "restore"
    else:
        return "unknown"


def get_host_from_action(action: int) -> str:
    """Get host name from action index."""
    if action == 0:
        return "none"
    host_idx = (action - 1) % 13
    return HOSTS[host_idx]


# ═══════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════

def plot_performance_comparison(results_list: List[Dict], save_path: str = None):
    """Plot performance comparison across methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward Distribution
    ax = axes[0, 0]
    for results in results_list:
        ax.hist(results['episode_rewards'], alpha=0.6, label=results['method'], bins=30)
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mean Reward Comparison
    ax = axes[0, 1]
    methods = [r['method'] for r in results_list]
    means = [r['mean_reward'] for r in results_list]
    stds = [r['std_reward'] for r in results_list]
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Mean Reward Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Episode Length Distribution
    ax = axes[1, 0]
    for results in results_list:
        ax.hist(results['episode_lengths'], alpha=0.6, label=results['method'], bins=20)
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative Reward
    ax = axes[1, 1]
    for results in results_list:
        cumulative = np.cumsum(results['episode_rewards'])
        ax.plot(cumulative, label=results['method'], alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")
    plt.show()


def plot_action_distributions(results_list: List[Dict], save_path: str = None):
    """Plot action type distributions for each method."""
    n_methods = len(results_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    action_types = ['sleep', 'analyse', 'decoy', 'remove', 'restore']
    
    for idx, results in enumerate(results_list):
        ax = axes[idx]
        
        if results['method'] == "Single Agent":
            # Single agent case
            counts = results['action_type_counts']
            values = [counts.get(at, 0) for at in action_types]
            colors = plt.cm.Set3(range(len(action_types)))
            ax.pie(values, labels=action_types, autopct='%1.1f%%', colors=colors)
            ax.set_title(f"{results['method']}\nAction Distribution")
        
        else:
            # Multi-agent case - show both agents
            agent_0_counts = results['agent_0_action_type_counts']
            agent_1_counts = results['agent_1_action_type_counts']
            
            values_0 = [agent_0_counts.get(at, 0) for at in action_types]
            values_1 = [agent_1_counts.get(at, 0) for at in action_types]
            
            x = np.arange(len(action_types))
            width = 0.35
            
            ax.bar(x - width/2, values_0, width, label='Agent 0', alpha=0.8)
            ax.bar(x + width/2, values_1, width, label='Agent 1', alpha=0.8)
            
            ax.set_ylabel('Action Count')
            ax.set_title(f"{results['method']}\nAction Distribution")
            ax.set_xticks(x)
            ax.set_xticklabels(action_types, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved action distributions to {save_path}")
    plt.show()


def plot_host_action_heatmap(results: Dict, save_path: str = None):
    """Plot heatmap of actions taken on each host."""
    action_types = ['analyse', 'decoy', 'remove', 'restore']
    hosts = HOSTS
    
    if results['method'] == "Single Agent":
        # Create matrix for single agent
        matrix = np.zeros((len(action_types), len(hosts)))
        
        for action, count in results['action_counts'].items():
            if action > 0:  # Skip sleep
                action_type_idx = (action - 1) // 13
                host_idx = (action - 1) % 13
                if action_type_idx < len(action_types):
                    matrix[action_type_idx, host_idx] += count
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(matrix, xticklabels=hosts, yticklabels=action_types, 
                    annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
        ax.set_title(f"{results['method']} - Actions per Host")
        ax.set_xlabel('Host')
        ax.set_ylabel('Action Type')
        
    else:
        # Create matrices for both agents
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        for agent_idx in range(2):
            ax = axes[agent_idx]
            matrix = np.zeros((len(action_types), len(hosts)))
            
            action_counts = results[f'agent_{agent_idx}_action_counts']
            for action, count in action_counts.items():
                if action > 0:  # Skip sleep
                    action_type_idx = (action - 1) // 13
                    host_idx = (action - 1) % 13
                    if action_type_idx < len(action_types):
                        matrix[action_type_idx, host_idx] += count
            
            sns.heatmap(matrix, xticklabels=hosts, yticklabels=action_types,
                       annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
            ax.set_title(f"{results['method']} - Agent {agent_idx} - Actions per Host")
            ax.set_xlabel('Host')
            ax.set_ylabel('Action Type')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved host action heatmap to {save_path}")
    plt.show()


def save_summary_report(results_list: List[Dict], save_path: str = None):
    """Save a text summary report of all results."""
    if save_path is None:
        save_path = "evaluation_summary.txt"
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        for results in results_list:
            f.write(f"\n{results['method']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean Reward:    {results['mean_reward']:.4f} ± {results['std_reward']:.4f}\n")
            f.write(f"Mean Length:    {results['mean_length']:.2f}\n")
            f.write(f"Min Reward:     {np.min(results['episode_rewards']):.4f}\n")
            f.write(f"Max Reward:     {np.max(results['episode_rewards']):.4f}\n")
            f.write(f"Median Reward:  {np.median(results['episode_rewards']):.4f}\n")
            
            if results['method'] == "Single Agent":
                f.write(f"\nAction Type Distribution:\n")
                total_actions = sum(results['action_type_counts'].values())
                for action_type, count in sorted(results['action_type_counts'].items()):
                    pct = 100 * count / total_actions
                    f.write(f"  {action_type:10s}: {count:6d} ({pct:5.2f}%)\n")
            else:
                for agent_idx in range(2):
                    f.write(f"\nAgent {agent_idx} Action Type Distribution:\n")
                    counts = results[f'agent_{agent_idx}_action_type_counts']
                    total_actions = sum(counts.values())
                    for action_type, count in sorted(counts.items()):
                        pct = 100 * count / total_actions
                        f.write(f"  {action_type:10s}: {count:6d} ({pct:5.2f}%)\n")
            
            f.write("\n")
        
        # Statistical comparison
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        methods = [r['method'] for r in results_list]
        rewards_matrix = [r['episode_rewards'] for r in results_list]
        
        f.write("Pairwise Reward Differences:\n")
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                diff = np.mean(rewards_matrix[i]) - np.mean(rewards_matrix[j])
                f.write(f"  {methods[i]} vs {methods[j]}: {diff:+.4f}\n")
    
    print(f"Saved summary report to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Model Discovery Functions
# ═══════════════════════════════════════════════════════════════════════

def find_models(base_dir: str = None) -> Dict[str, any]:
    """
    Automatically find trained models in the standard directories.
    
    Returns:
        Dictionary with model paths for each method
    """
    if base_dir is None:
        # Default to parent directory of this script
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent / "ppo_models"
    else:
        base_dir = Path(base_dir)
    
    models = {
        "single_agent": None,
        "ippo": None,
        "restricted_ippo": None,
    }
    
    if not base_dir.exists():
        print(f"Warning: Model directory {base_dir} does not exist")
        return models
    
    # Find Single Agent models (both .pt from new training and .zip from SB3)
    single_agent_dirs = list(base_dir.glob("SingleAgent_PPO_*")) + list(base_dir.glob("SB3_*"))
    if single_agent_dirs:
        # Get most recent directory
        latest_dir = max(single_agent_dirs, key=lambda p: p.stat().st_mtime)
        
        # Try .pt files first (new training format)
        pt_files = [latest_dir / "agent_final.pt"]
        if not pt_files[0].exists():
            pt_files = list(latest_dir.glob("agent_update_*.pt"))
        
        # Try .zip files (SB3 format)
        zip_files = list(latest_dir.glob("*.zip"))
        
        if pt_files and pt_files[0].exists():
            latest_model = max([f for f in pt_files if f.exists()], key=lambda p: p.stat().st_mtime)
            models["single_agent"] = str(latest_model)
            print(f"Found Single Agent model: {latest_model}")
        elif zip_files:
            latest_model = max(zip_files, key=lambda p: p.stat().st_mtime)
            models["single_agent"] = str(latest_model)
            print(f"Found Single Agent model (SB3): {latest_model}")
    
    # Find IPPO models
    ippo_dirs = list(base_dir.glob("SharedObs_IPPO_*"))
    if ippo_dirs:
        latest_dir = max(ippo_dirs, key=lambda p: p.stat().st_mtime)
        agent_0 = latest_dir / "agent_0_final.pt"
        agent_1 = latest_dir / "agent_1_final.pt"
        if agent_0.exists() and agent_1.exists():
            models["ippo"] = [str(agent_0), str(agent_1)]
            print(f"Found IPPO models in: {latest_dir}")
        else:
            # Try to find latest checkpoint
            checkpoints_0 = list(latest_dir.glob("agent_0_update_*.pt"))
            checkpoints_1 = list(latest_dir.glob("agent_1_update_*.pt"))
            if checkpoints_0 and checkpoints_1:
                latest_0 = max(checkpoints_0, key=lambda p: p.stat().st_mtime)
                latest_1 = max(checkpoints_1, key=lambda p: p.stat().st_mtime)
                models["ippo"] = [str(latest_0), str(latest_1)]
                print(f"Found IPPO checkpoint models in: {latest_dir}")
    
    # Find Restricted IPPO models
    restricted_dirs = list(base_dir.glob("Restricted_IPPO_*"))
    if restricted_dirs:
        latest_dir = max(restricted_dirs, key=lambda p: p.stat().st_mtime)
        agent_0 = latest_dir / "agent_0_defender_final.pt"
        agent_1 = latest_dir / "agent_1_decoy_final.pt"
        if agent_0.exists() and agent_1.exists():
            models["restricted_ippo"] = [str(agent_0), str(agent_1)]
            print(f"Found Restricted IPPO models in: {latest_dir}")
        else:
            # Try to find latest checkpoint
            checkpoints_0 = list(latest_dir.glob("agent_0_defender_update_*.pt"))
            checkpoints_1 = list(latest_dir.glob("agent_1_decoy_update_*.pt"))
            if checkpoints_0 and checkpoints_1:
                latest_0 = max(checkpoints_0, key=lambda p: p.stat().st_mtime)
                latest_1 = max(checkpoints_1, key=lambda p: p.stat().st_mtime)
                models["restricted_ippo"] = [str(latest_0), str(latest_1)]
                print(f"Found Restricted IPPO checkpoint models in: {latest_dir}")
    
    return models


# ═══════════════════════════════════════════════════════════════════════
# Main Evaluation Script
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and evaluate all models
  python evaluate_all_methods.py --auto
  
  # Specify model directory
  python evaluate_all_methods.py --model-dir ../ppo_models
  
  # Manually specify models
  python evaluate_all_methods.py --single-agent-model path/to/model.zip
  
  # Evaluate only specific methods
  python evaluate_all_methods.py --auto --methods single_agent ippo
        """
    )
    
    # Model discovery options
    parser.add_argument("--auto", action="store_true", 
                       help="Automatically discover models in default directories")
    parser.add_argument("--model-dir", type=str, 
                       help="Base directory containing model subdirectories")
    
    # Manual model specification
    parser.add_argument("--single-agent-model", type=str, 
                       help="Path to single agent model (.zip)")
    parser.add_argument("--ippo-agent0-model", type=str, 
                       help="Path to IPPO agent 0 model (.pt)")
    parser.add_argument("--ippo-agent1-model", type=str, 
                       help="Path to IPPO agent 1 model (.pt)")
    parser.add_argument("--restricted-agent0-model", type=str, 
                       help="Path to restricted IPPO agent 0 model (.pt)")
    parser.add_argument("--restricted-agent1-model", type=str, 
                       help="Path to restricted IPPO agent 1 model (.pt)")
    
    # Evaluation options
    parser.add_argument("--methods", nargs="+", 
                       choices=["single_agent", "ippo", "restricted_ippo"],
                       help="Specific methods to evaluate (default: all found)")
    parser.add_argument("--n-episodes", type=int, default=100, 
                       help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", 
                       help="Use deterministic actions")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-discover models if requested
    if args.auto or args.model_dir:
        print(f"\n{'='*60}")
        print("Auto-discovering models...")
        print(f"{'='*60}")
        discovered_models = find_models(args.model_dir)
        
        # Override with discovered models if not manually specified
        if not args.single_agent_model and discovered_models["single_agent"]:
            args.single_agent_model = discovered_models["single_agent"]
        
        if not (args.ippo_agent0_model and args.ippo_agent1_model) and discovered_models["ippo"]:
            args.ippo_agent0_model, args.ippo_agent1_model = discovered_models["ippo"]
        
        if not (args.restricted_agent0_model and args.restricted_agent1_model) and discovered_models["restricted_ippo"]:
            args.restricted_agent0_model, args.restricted_agent1_model = discovered_models["restricted_ippo"]
    
    # Determine which methods to evaluate
    methods_to_eval = args.methods if args.methods else ["single_agent", "ippo", "restricted_ippo"]
    
    results_list = []
    
    # Evaluate Single Agent
    if "single_agent" in methods_to_eval and args.single_agent_model:
        try:
            results = evaluate_single_agent(
                args.single_agent_model,
                n_episodes=args.n_episodes,
                deterministic=args.deterministic
            )
            results_list.append(results)
        except Exception as e:
            print(f"Error evaluating Single Agent: {e}")
    elif "single_agent" in methods_to_eval:
        print("\nSkipping Single Agent: No model found")
    
    # Evaluate IPPO
    if "ippo" in methods_to_eval and args.ippo_agent0_model and args.ippo_agent1_model:
        try:
            results = evaluate_ippo(
                [args.ippo_agent0_model, args.ippo_agent1_model],
                n_episodes=args.n_episodes,
                deterministic=args.deterministic
            )
            results_list.append(results)
        except Exception as e:
            print(f"Error evaluating IPPO: {e}")
    elif "ippo" in methods_to_eval:
        print("\nSkipping IPPO: No models found")
    
    # Evaluate Restricted IPPO
    if "restricted_ippo" in methods_to_eval and args.restricted_agent0_model and args.restricted_agent1_model:
        try:
            results = evaluate_restricted_ippo(
                [args.restricted_agent0_model, args.restricted_agent1_model],
                n_episodes=args.n_episodes,
                deterministic=args.deterministic
            )
            results_list.append(results)
        except Exception as e:
            print(f"Error evaluating Restricted IPPO: {e}")
    elif "restricted_ippo" in methods_to_eval:
        print("\nSkipping Restricted IPPO: No models found")
    
    # Generate visualizations
    if results_list:
        print(f"\n{'='*60}")
        print("Generating Visualizations")
        print(f"{'='*60}")
        
        plot_performance_comparison(
            results_list, 
            save_path=output_dir / "performance_comparison.png"
        )
        
        plot_action_distributions(
            results_list,
            save_path=output_dir / "action_distributions.png"
        )
        
        # Plot heatmaps for each method
        for results in results_list:
            method_name = results['method'].replace(" ", "_").lower()
            plot_host_action_heatmap(
                results,
                save_path=output_dir / f"host_actions_{method_name}.png"
            )
        
        # Save summary report
        save_summary_report(
            results_list,
            save_path=output_dir / "evaluation_summary.txt"
        )
        
        print(f"\n{'='*60}")
        print(f"Evaluation complete! Results saved to {output_dir}")
        print(f"{'='*60}")
        print(f"\nGenerated files:")
        print(f"  - performance_comparison.png")
        print(f"  - action_distributions.png")
        for results in results_list:
            method_name = results['method'].replace(" ", "_").lower()
            print(f"  - host_actions_{method_name}.png")
        print(f"  - evaluation_summary.txt")
    else:
        print(f"\n{'='*60}")
        print("No models found or evaluated!")
        print(f"{'='*60}")
        print("\nPlease ensure you have trained models in one of these locations:")
        print("  - ../ppo_models/SingleAgent_PPO_*/agent_final.pt (Single Agent)")
        print("  - ../ppo_models/SharedObs_IPPO_*/agent_*_final.pt (IPPO)")
        print("  - ../ppo_models/Restricted_IPPO_*/agent_*_final.pt (Restricted IPPO)")
        print("\nOr specify model paths manually using:")
        print("  --single-agent-model, --ippo-agent0-model, --ippo-agent1-model, etc.")
        print("\nTo train models, run:")
        print("  - python train_single_agent.py (Single Agent)")
        print("  - python train_ippo.py (IPPO)")
        print("  - python train_restricted_ippo.py (Restricted IPPO)")


if __name__ == "__main__":
    import sys
    # If no arguments provided, default to auto mode
    if len(sys.argv) == 1:
        sys.argv.append("--auto")
    main()

