"""
Compare Action Distributions Across Multi-Exploit IPPO Environment Variants

This script evaluates trained agents across all 3 environment variants:
- SharedObs: Both agents have full action space (144 actions each)
- Restricted: Agent 0 = Defender (40 actions), Agent 1 = Decoy Manager (118 actions)
- DecoySpecialized: Agent 0 = Defender+Decoys1 (92 actions), Agent 1 = Support+Decoys2 (66 actions)

Generates comprehensive comparison visualizations showing how agent behavior
differs across environment designs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from multi_exploit_marl_wrapper import (
    MultiExploitSharedObsEnv,
    MultiExploitRestrictedEnv,
    MultiExploitDecoySpecializedEnv,
    NUM_HOSTS,
    ANALYSE_START, ANALYSE_END,
    DECOY_START, DECOY_END,
    REMOVE_START, REMOVE_END,
    RESTORE_START, RESTORE_END,
)
from multi_exploit_env import HOSTS, DECOYS, EXPLOITS


# ═══════════════════════════════════════════════════════════════════════
# Red Action Constants (147 total actions)
# ═══════════════════════════════════════════════════════════════════════
# Red action layout:
# - 0: sleep
# - 1-3: remote subnet scans (3 subnets)
# - 4-16: network scans (13 hosts)
# - 17-120: exploit actions (8 exploits × 13 hosts = 104)
# - 121-133: escalate (13 hosts)
# - 134-146: impact (13 hosts)

RED_SLEEP = 0
RED_SUBNET_SCAN_START = 1
RED_SUBNET_SCAN_END = 4
RED_NETWORK_SCAN_START = 4
RED_NETWORK_SCAN_END = 17
RED_EXPLOIT_START = 17
RED_EXPLOIT_END = 121
RED_ESCALATE_START = 121
RED_ESCALATE_END = 134
RED_IMPACT_START = 134
RED_IMPACT_END = 147

NUM_EXPLOITS = len(EXPLOITS)  # 8

RED_ACTION_TYPES = ['sleep', 'subnet_scan', 'network_scan', 'exploit', 'escalate', 'impact']
RED_ACTION_COLORS = {
    'sleep': '#95a5a6',
    'subnet_scan': '#3498db',
    'network_scan': '#9b59b6',
    'exploit': '#e74c3c',
    'escalate': '#f39c12',
    'impact': '#c0392b'
}


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = ["shared", "restricted", "decoy_specialized"]
VARIANT_LABELS = {
    "shared": "SharedObs\n(144, 144)",
    "restricted": "Restricted\n(40, 118)",
    "decoy_specialized": "DecoySpecialized\n(92, 66)"
}
VARIANT_COLORS = {
    "shared": "#3498db",
    "restricted": "#e74c3c", 
    "decoy_specialized": "#2ecc71"
}
ACTION_TYPES = ['sleep', 'analyse', 'decoy', 'remove', 'restore']
ACTION_TYPE_COLORS = {
    'sleep': '#95a5a6',
    'analyse': '#3498db',
    'decoy': '#2ecc71',
    'remove': '#e74c3c',
    'restore': '#9b59b6'
}


# ═══════════════════════════════════════════════════════════════════════
# Model Definition
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
    
    def forward(self, x):
        features = self.features(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, x, deterministic=False):
        action_logits, _ = self.forward(x)
        if deterministic:
            return torch.argmax(action_logits, dim=-1)
        return Categorical(logits=action_logits).sample()


# ═══════════════════════════════════════════════════════════════════════
# Action Classification
# ═══════════════════════════════════════════════════════════════════════

def classify_action(action: int) -> str:
    """Classify blue action into type."""
    if action == 0:
        return "sleep"
    elif ANALYSE_START <= action < ANALYSE_END:
        return "analyse"
    elif DECOY_START <= action < DECOY_END:
        return "decoy"
    elif REMOVE_START <= action < REMOVE_END:
        return "remove"
    elif RESTORE_START <= action < RESTORE_END:
        return "restore"
    return "unknown"


def classify_red_action(action: int) -> str:
    """Classify red action into type."""
    if action == RED_SLEEP:
        return "sleep"
    elif RED_SUBNET_SCAN_START <= action < RED_SUBNET_SCAN_END:
        return "subnet_scan"
    elif RED_NETWORK_SCAN_START <= action < RED_NETWORK_SCAN_END:
        return "network_scan"
    elif RED_EXPLOIT_START <= action < RED_EXPLOIT_END:
        return "exploit"
    elif RED_ESCALATE_START <= action < RED_ESCALATE_END:
        return "escalate"
    elif RED_IMPACT_START <= action < RED_IMPACT_END:
        return "impact"
    return "unknown"


def get_red_exploit_type(action: int) -> Optional[str]:
    """Get exploit type from red action (if it's an exploit action)."""
    if RED_EXPLOIT_START <= action < RED_EXPLOIT_END:
        exploit_offset = action - RED_EXPLOIT_START
        exploit_idx = exploit_offset // NUM_HOSTS
        return EXPLOITS[exploit_idx]
    return None


def get_red_host(action: int) -> Optional[str]:
    """Get target host from red action."""
    if action == RED_SLEEP:
        return None
    elif RED_SUBNET_SCAN_START <= action < RED_SUBNET_SCAN_END:
        return None  # Subnet scans don't target specific hosts
    elif RED_NETWORK_SCAN_START <= action < RED_NETWORK_SCAN_END:
        host_idx = action - RED_NETWORK_SCAN_START
    elif RED_EXPLOIT_START <= action < RED_EXPLOIT_END:
        exploit_offset = action - RED_EXPLOIT_START
        host_idx = exploit_offset % NUM_HOSTS
    elif RED_ESCALATE_START <= action < RED_ESCALATE_END:
        host_idx = action - RED_ESCALATE_START
    elif RED_IMPACT_START <= action < RED_IMPACT_END:
        host_idx = action - RED_IMPACT_START
    else:
        return None
    return HOSTS[host_idx] if 0 <= host_idx < len(HOSTS) else None


def get_decoy_type(action: int) -> Optional[str]:
    """Get decoy type from action."""
    if DECOY_START <= action < DECOY_END:
        decoy_idx = (action - DECOY_START) // NUM_HOSTS
        return DECOYS[decoy_idx]
    return None


def get_host(action: int) -> Optional[str]:
    """Get target host from action."""
    if action == 0:
        return None
    elif ANALYSE_START <= action < ANALYSE_END:
        host_idx = action - ANALYSE_START
    elif DECOY_START <= action < DECOY_END:
        host_idx = (action - DECOY_START) % NUM_HOSTS
    elif REMOVE_START <= action < REMOVE_END:
        host_idx = action - REMOVE_START
    elif RESTORE_START <= action < RESTORE_END:
        host_idx = action - RESTORE_START
    else:
        return None
    return HOSTS[host_idx]


# ═══════════════════════════════════════════════════════════════════════
# Environment & Model Setup
# ═══════════════════════════════════════════════════════════════════════

def create_env(variant: str, max_steps: int = 100):
    """Create environment for variant."""
    if variant == "shared":
        return MultiExploitSharedObsEnv(
            red_policy="high_success", max_steps=max_steps,
            remove_bugs=True, num_agents=2, action_resolution="sequential"
        )
    elif variant == "restricted":
        return MultiExploitRestrictedEnv(
            red_policy="high_success", max_steps=max_steps, remove_bugs=True
        )
    elif variant == "decoy_specialized":
        return MultiExploitDecoySpecializedEnv(
            red_policy="high_success", max_steps=max_steps, remove_bugs=True
        )
    raise ValueError(f"Unknown variant: {variant}")


def get_action_dims(env, variant: str) -> List[int]:
    """Get action dimensions for each agent."""
    if variant == "shared":
        return [env.action_space.n, env.action_space.n]
    return [env.get_action_space(i).n for i in range(2)]


def find_models(base_dir: Path, variant: str, total_timesteps: int) -> Optional[List[str]]:
    """Find trained models for variant."""
    group_name = f"MultiExploit_{variant.title()}_IPPO_2Blue_{total_timesteps}"
    model_dir = base_dir / group_name
    
    if not model_dir.exists():
        return None
    
    # Try final models
    agent_0 = model_dir / "agent_0_final.pt"
    agent_1 = model_dir / "agent_1_final.pt"
    if agent_0.exists() and agent_1.exists():
        return [str(agent_0), str(agent_1)]
    
    # Try latest checkpoints
    checkpoints_0 = list(model_dir.glob("agent_0_update_*.pt"))
    checkpoints_1 = list(model_dir.glob("agent_1_update_*.pt"))
    if checkpoints_0 and checkpoints_1:
        latest_0 = max(checkpoints_0, key=lambda p: p.stat().st_mtime)
        latest_1 = max(checkpoints_1, key=lambda p: p.stat().st_mtime)
        return [str(latest_0), str(latest_1)]
    
    return None


def load_agents(model_paths: List[str], obs_dim: int, action_dims: List[int], device) -> List[nn.Module]:
    """Load trained agents."""
    agents = []
    for idx, path in enumerate(model_paths):
        agent = ActorCritic(obs_dim, action_dims[idx], hidden_dim=256).to(device)
        checkpoint = torch.load(path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        agents.append(agent)
    return agents


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_variant(
    variant: str,
    model_paths: List[str],
    n_episodes: int = 100,
    deterministic: bool = True,
    device: torch.device = None
) -> Dict:
    """Evaluate a single variant and collect statistics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = create_env(variant)
    action_dims = get_action_dims(env, variant)
    obs_dim = env.observation_space.shape[0]
    agents = load_agents(model_paths, obs_dim, action_dims, device)
    
    # Statistics containers
    stats = {
        'variant': variant,
        'rewards': [],
        'lengths': [],
        'action_dims': action_dims,
        'agents': [{
            'action_type_counts': defaultdict(int),
            'decoy_type_counts': defaultdict(int),
            'host_counts': defaultdict(lambda: defaultdict(int)),
            'raw_action_counts': defaultdict(int),
        } for _ in range(2)],
        # Red agent statistics
        'red': {
            'action_type_counts': defaultdict(int),
            'exploit_type_counts': defaultdict(int),
            'host_counts': defaultdict(lambda: defaultdict(int)),
            'raw_action_counts': defaultdict(int),
        }
    }
    
    for ep in range(n_episodes):
        obs_list = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            actions = []
            for agent_idx, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs_list[agent_idx]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = agent.get_action(obs_tensor, deterministic=deterministic).item()
                actions.append(action)
                
                # Map to env action
                if variant == "shared":
                    env_action = action
                else:
                    env_action = env._map_action_to_env(agent_idx, action)
                
                # Record statistics
                stats['agents'][agent_idx]['raw_action_counts'][action] += 1
                
                action_type = classify_action(env_action)
                stats['agents'][agent_idx]['action_type_counts'][action_type] += 1
                
                if action_type == "decoy":
                    decoy = get_decoy_type(env_action)
                    if decoy:
                        stats['agents'][agent_idx]['decoy_type_counts'][decoy] += 1
                
                host = get_host(env_action)
                if host and action_type != "sleep":
                    stats['agents'][agent_idx]['host_counts'][host][action_type] += 1
            
            obs_list, rewards, dones, infos = env.step(actions)
            
            # Track red agent action
            red_action = infos[0].get('red_action', 0)
            stats['red']['raw_action_counts'][red_action] += 1
            
            red_action_type = classify_red_action(red_action)
            stats['red']['action_type_counts'][red_action_type] += 1
            
            if red_action_type == "exploit":
                exploit_type = get_red_exploit_type(red_action)
                if exploit_type:
                    stats['red']['exploit_type_counts'][exploit_type] += 1
            
            red_host = get_red_host(red_action)
            if red_host and red_action_type not in ["sleep", "subnet_scan"]:
                stats['red']['host_counts'][red_host][red_action_type] += 1
            
            episode_reward += rewards[0]
            steps += 1
            done = dones[0]
        
        stats['rewards'].append(episode_reward)
        stats['lengths'].append(steps)
    
    # Convert defaultdicts to regular dicts
    for agent_stats in stats['agents']:
        agent_stats['action_type_counts'] = dict(agent_stats['action_type_counts'])
        agent_stats['decoy_type_counts'] = dict(agent_stats['decoy_type_counts'])
        agent_stats['host_counts'] = {h: dict(v) for h, v in agent_stats['host_counts'].items()}
        agent_stats['raw_action_counts'] = dict(agent_stats['raw_action_counts'])
    
    # Convert red stats
    stats['red']['action_type_counts'] = dict(stats['red']['action_type_counts'])
    stats['red']['exploit_type_counts'] = dict(stats['red']['exploit_type_counts'])
    stats['red']['host_counts'] = {h: dict(v) for h, v in stats['red']['host_counts'].items()}
    stats['red']['raw_action_counts'] = dict(stats['red']['raw_action_counts'])
    
    return stats


# ═══════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════

def plot_action_type_comparison(all_stats: Dict[str, Dict], save_path: str):
    """Compare action type distributions across variants."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        stats = all_stats[variant]
        
        for row in range(2):
            ax = axes[row, col]
            counts = stats['agents'][row]['action_type_counts']
            
            values = [counts.get(at, 0) for at in ACTION_TYPES]
            total = sum(values)
            
            if total > 0:
                colors = [ACTION_TYPE_COLORS[at] for at in ACTION_TYPES]
                wedges, texts, autotexts = ax.pie(
                    values, 
                    labels=ACTION_TYPES,
                    colors=colors,
                    autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
                    startangle=90,
                    pctdistance=0.75
                )
                
                # Style
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight('bold')
            
            role = get_agent_role(variant, row)
            ax.set_title(f"Agent {row}: {role}\n(n={total})", fontsize=11, fontweight='bold')
    
    # Column titles
    for col, variant in enumerate(VARIANTS):
        axes[0, col].annotate(
            VARIANT_LABELS[variant], 
            xy=(0.5, 1.15), xycoords='axes fraction',
            ha='center', fontsize=13, fontweight='bold',
            color=VARIANT_COLORS[variant]
        )
    
    fig.suptitle("Action Type Distribution Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def get_agent_role(variant: str, agent_idx: int) -> str:
    """Get role description for agent."""
    if variant == "shared":
        return "Full Actions"
    elif variant == "restricted":
        return "Defender" if agent_idx == 0 else "Decoy Manager"
    else:  # decoy_specialized
        return "Defender + Decoys A" if agent_idx == 0 else "Support + Decoys B"


def plot_action_type_bars(all_stats: Dict[str, Dict], save_path: str):
    """Bar chart comparison of action types across all variants and agents."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for agent_idx in range(2):
        ax = axes[agent_idx]
        
        x = np.arange(len(ACTION_TYPES))
        width = 0.25
        
        for i, variant in enumerate(VARIANTS):
            if variant not in all_stats:
                continue
            
            counts = all_stats[variant]['agents'][agent_idx]['action_type_counts']
            total = sum(counts.values())
            percentages = [100 * counts.get(at, 0) / total if total > 0 else 0 for at in ACTION_TYPES]
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, percentages, width, 
                         label=variant, color=VARIANT_COLORS[variant], alpha=0.85)
            
            # Add value labels on bars
            for bar, pct in zip(bars, percentages):
                if pct > 3:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{pct:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Action Type', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Agent {agent_idx} Action Distribution', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_TYPES, fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
    
    fig.suptitle("Action Type Distribution by Agent", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_decoy_comparison(all_stats: Dict[str, Dict], save_path: str):
    """Compare decoy type usage across variants."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    all_decoys = DECOYS
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        stats = all_stats[variant]
        
        for row in range(2):
            ax = axes[row, col]
            counts = stats['agents'][row]['decoy_type_counts']
            
            values = [counts.get(d, 0) for d in all_decoys]
            total = sum(values)
            
            if total > 0:
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_decoys)))
                bars = ax.barh(all_decoys, values, color=colors)
                
                # Value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                               f'{val}', va='center', fontsize=9)
                
                ax.set_xlabel('Count')
            else:
                ax.text(0.5, 0.5, 'No decoys\ndeployed', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
            
            role = get_agent_role(variant, row)
            ax.set_title(f"Agent {row}: {role}\n(total={total})", fontsize=11, fontweight='bold')
    
    # Column titles
    for col, variant in enumerate(VARIANTS):
        axes[0, col].annotate(
            VARIANT_LABELS[variant],
            xy=(0.5, 1.15), xycoords='axes fraction',
            ha='center', fontsize=13, fontweight='bold',
            color=VARIANT_COLORS[variant]
        )
    
    fig.suptitle("Decoy Type Distribution Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_host_heatmaps(all_stats: Dict[str, Dict], save_path: str):
    """Heatmaps showing which hosts each agent targets."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    action_types_for_heatmap = ['analyse', 'decoy', 'remove', 'restore']
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        stats = all_stats[variant]
        
        for row in range(2):
            ax = axes[row, col]
            host_counts = stats['agents'][row]['host_counts']
            
            # Build matrix
            matrix = np.zeros((len(action_types_for_heatmap), len(HOSTS)))
            for host_idx, host in enumerate(HOSTS):
                if host in host_counts:
                    for action_idx, action_type in enumerate(action_types_for_heatmap):
                        matrix[action_idx, host_idx] = host_counts[host].get(action_type, 0)
            
            sns.heatmap(
                matrix, 
                xticklabels=HOSTS, 
                yticklabels=action_types_for_heatmap,
                annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'shrink': 0.8}
            )
            
            role = get_agent_role(variant, row)
            ax.set_title(f"Agent {row}: {role}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Host')
            ax.set_ylabel('Action')
    
    # Column titles
    for col, variant in enumerate(VARIANTS):
        axes[0, col].annotate(
            VARIANT_LABELS[variant],
            xy=(0.5, 1.15), xycoords='axes fraction',
            ha='center', fontsize=13, fontweight='bold',
            color=VARIANT_COLORS[variant]
        )
    
    fig.suptitle("Host Targeting Heatmaps", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_performance_summary(all_stats: Dict[str, Dict], save_path: str):
    """Summary performance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Mean Reward Comparison
    ax = axes[0, 0]
    variants_present = [v for v in VARIANTS if v in all_stats]
    means = [np.mean(all_stats[v]['rewards']) for v in variants_present]
    stds = [np.std(all_stats[v]['rewards']) for v in variants_present]
    colors = [VARIANT_COLORS[v] for v in variants_present]
    
    x = np.arange(len(variants_present))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants_present])
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Performance Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[bars.index(bar)] + 0.5,
               f'{mean:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # 2. Reward Distribution
    ax = axes[0, 1]
    for variant in variants_present:
        ax.hist(all_stats[variant]['rewards'], alpha=0.6, label=variant, 
               color=VARIANT_COLORS[variant], bins=25)
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Combined Action Type Distribution
    ax = axes[1, 0]
    x = np.arange(len(ACTION_TYPES))
    width = 0.25
    
    for i, variant in enumerate(variants_present):
        # Combine both agents
        combined = defaultdict(int)
        for agent_stats in all_stats[variant]['agents']:
            for at, count in agent_stats['action_type_counts'].items():
                combined[at] += count
        
        total = sum(combined.values())
        pcts = [100 * combined.get(at, 0) / total if total > 0 else 0 for at in ACTION_TYPES]
        
        offset = (i - len(variants_present)/2 + 0.5) * width
        ax.bar(x + offset, pcts, width, label=variant, color=VARIANT_COLORS[variant], alpha=0.85)
    
    ax.set_xlabel('Action Type')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Combined Action Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_TYPES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Decoy Usage Summary
    ax = axes[1, 1]
    decoy_totals = {}
    for variant in variants_present:
        total = sum(
            sum(agent['decoy_type_counts'].values())
            for agent in all_stats[variant]['agents']
        )
        decoy_totals[variant] = total
    
    bars = ax.bar(
        range(len(variants_present)),
        [decoy_totals[v] for v in variants_present],
        color=[VARIANT_COLORS[v] for v in variants_present],
        alpha=0.85
    )
    ax.set_xticks(range(len(variants_present)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants_present])
    ax.set_ylabel('Total Decoy Deployments')
    ax.set_title('Decoy Usage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [decoy_totals[v] for v in variants_present]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{val}', ha='center', fontsize=10, fontweight='bold')
    
    fig.suptitle("Multi-Exploit IPPO - Variant Comparison Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_agent_specialization(all_stats: Dict[str, Dict], save_path: str):
    """Visualize how agents specialize in different variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        
        ax = axes[col]
        stats = all_stats[variant]
        
        # Calculate specialization: what % of each action type does each agent do
        combined = {at: [0, 0] for at in ACTION_TYPES}
        
        for agent_idx in range(2):
            for at, count in stats['agents'][agent_idx]['action_type_counts'].items():
                combined[at][agent_idx] += count
        
        # Stacked bar chart
        x = np.arange(len(ACTION_TYPES))
        width = 0.6
        
        agent0_pcts = []
        agent1_pcts = []
        
        for at in ACTION_TYPES:
            total = combined[at][0] + combined[at][1]
            if total > 0:
                agent0_pcts.append(100 * combined[at][0] / total)
                agent1_pcts.append(100 * combined[at][1] / total)
            else:
                agent0_pcts.append(0)
                agent1_pcts.append(0)
        
        ax.bar(x, agent0_pcts, width, label='Agent 0', color='#3498db', alpha=0.85)
        ax.bar(x, agent1_pcts, width, bottom=agent0_pcts, label='Agent 1', color='#e74c3c', alpha=0.85)
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Action Type')
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_TYPES, rotation=15, ha='right')
        ax.set_title(f"{VARIANT_LABELS[variant]}", fontsize=13, fontweight='bold',
                    color=VARIANT_COLORS[variant])
        ax.legend(loc='upper right')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Agent Specialization: Who Does What?", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_red_action_distribution(all_stats: Dict[str, Dict], save_path: str):
    """Plot red agent action type distribution across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        
        ax = axes[col]
        counts = all_stats[variant]['red']['action_type_counts']
        
        values = [counts.get(at, 0) for at in RED_ACTION_TYPES]
        total = sum(values)
        
        if total > 0:
            colors = [RED_ACTION_COLORS[at] for at in RED_ACTION_TYPES]
            non_zero = [(at, v, c) for at, v, c in zip(RED_ACTION_TYPES, values, colors) if v > 0]
            
            if non_zero:
                labels = [f"{at}\n({v}, {100*v/total:.1f}%)" for at, v, _ in non_zero]
                sizes = [v for _, v, _ in non_zero]
                colors_used = [c for _, _, c in non_zero]
                
                ax.pie(sizes, labels=labels, colors=colors_used, startangle=90)
        
        ax.set_title(f"{VARIANT_LABELS[variant]}\n(Total: {total})", fontsize=12, fontweight='bold',
                    color=VARIANT_COLORS[variant])
    
    fig.suptitle("Red Agent Action Type Distribution", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_red_exploit_distribution(all_stats: Dict[str, Dict], save_path: str):
    """Plot red agent exploit type usage across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        
        ax = axes[col]
        counts = all_stats[variant]['red']['exploit_type_counts']
        
        if counts:
            exploits = list(counts.keys())
            values = list(counts.values())
            
            # Sort by count
            sorted_pairs = sorted(zip(exploits, values), key=lambda x: -x[1])
            exploits, values = zip(*sorted_pairs)
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(exploits)))
            bars = ax.barh(exploits, values, color=colors)
            
            ax.set_xlabel('Count')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       str(val), va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No exploits', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_title(f"{VARIANT_LABELS[variant]}", fontsize=12, fontweight='bold',
                    color=VARIANT_COLORS[variant])
    
    fig.suptitle("Red Agent Exploit Type Distribution", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_red_host_heatmap(all_stats: Dict[str, Dict], save_path: str):
    """Plot heatmap of red agent actions per host across variants."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    red_action_types_for_heatmap = ['network_scan', 'exploit', 'escalate', 'impact']
    
    for col, variant in enumerate(VARIANTS):
        if variant not in all_stats:
            continue
        
        ax = axes[col]
        host_counts = all_stats[variant]['red']['host_counts']
        
        # Build matrix
        matrix = np.zeros((len(red_action_types_for_heatmap), len(HOSTS)))
        for host_idx, host in enumerate(HOSTS):
            if host in host_counts:
                for action_idx, action_type in enumerate(red_action_types_for_heatmap):
                    matrix[action_idx, host_idx] = host_counts[host].get(action_type, 0)
        
        sns.heatmap(
            matrix,
            xticklabels=HOSTS,
            yticklabels=red_action_types_for_heatmap,
            annot=True, fmt='.0f', cmap='Reds', ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_title(f"{VARIANT_LABELS[variant]}", fontsize=12, fontweight='bold',
                    color=VARIANT_COLORS[variant])
        ax.set_xlabel('Host')
        ax.set_ylabel('Action')
    
    fig.suptitle("Red Agent Actions per Host", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_red_action_bars(all_stats: Dict[str, Dict], save_path: str):
    """Bar chart comparing red action types across variants."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    variants_present = [v for v in VARIANTS if v in all_stats]
    x = np.arange(len(RED_ACTION_TYPES))
    width = 0.25
    
    for i, variant in enumerate(variants_present):
        counts = all_stats[variant]['red']['action_type_counts']
        total = sum(counts.values())
        percentages = [100 * counts.get(at, 0) / total if total > 0 else 0 for at in RED_ACTION_TYPES]
        
        offset = (i - len(variants_present)/2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width,
                     label=variant, color=VARIANT_COLORS[variant], alpha=0.85)
        
        # Add value labels
        for bar, pct in zip(bars, percentages):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Action Type', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Red Agent Action Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(RED_ACTION_TYPES, fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def save_detailed_report(all_stats: Dict[str, Dict], save_path: str):
    """Save detailed text report."""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-EXPLOIT IPPO - ACTION DISTRIBUTION COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Variant':<25} {'Mean Reward':>12} {'Std':>10} {'Actions':>15}\n")
        f.write("-" * 60 + "\n")
        
        for variant in VARIANTS:
            if variant not in all_stats:
                continue
            stats = all_stats[variant]
            mean = np.mean(stats['rewards'])
            std = np.std(stats['rewards'])
            dims = stats['action_dims']
            f.write(f"{variant:<25} {mean:>12.2f} {std:>10.2f} {str(dims):>15}\n")
        
        f.write("\n")
        
        # Per-variant details
        for variant in VARIANTS:
            if variant not in all_stats:
                continue
            
            stats = all_stats[variant]
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"{variant.upper()}\n")
            f.write("=" * 60 + "\n")
            
            f.write(f"\nAction Dimensions: {stats['action_dims']}\n")
            f.write(f"Mean Reward: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}\n")
            f.write(f"Mean Episode Length: {np.mean(stats['lengths']):.2f}\n")
            
            for agent_idx in range(2):
                role = get_agent_role(variant, agent_idx)
                f.write(f"\n--- Agent {agent_idx} ({role}) ---\n")
                
                f.write("\nAction Types:\n")
                counts = stats['agents'][agent_idx]['action_type_counts']
                total = sum(counts.values())
                for at in ACTION_TYPES:
                    c = counts.get(at, 0)
                    pct = 100 * c / total if total > 0 else 0
                    f.write(f"  {at:10s}: {c:6d} ({pct:5.1f}%)\n")
                
                f.write("\nDecoy Types:\n")
                decoy_counts = stats['agents'][agent_idx]['decoy_type_counts']
                if decoy_counts:
                    total_decoys = sum(decoy_counts.values())
                    for decoy in sorted(decoy_counts.keys(), key=lambda x: -decoy_counts[x]):
                        c = decoy_counts[decoy]
                        pct = 100 * c / total_decoys if total_decoys > 0 else 0
                        f.write(f"  {decoy:10s}: {c:6d} ({pct:5.1f}%)\n")
                else:
                    f.write("  (no decoys)\n")
            
            # Red agent statistics
            f.write(f"\n--- Red Agent ---\n")
            
            f.write("\nAction Types:\n")
            red_counts = stats['red']['action_type_counts']
            red_total = sum(red_counts.values())
            for at in RED_ACTION_TYPES:
                c = red_counts.get(at, 0)
                pct = 100 * c / red_total if red_total > 0 else 0
                f.write(f"  {at:15s}: {c:6d} ({pct:5.1f}%)\n")
            
            f.write("\nExploit Types:\n")
            exploit_counts = stats['red']['exploit_type_counts']
            if exploit_counts:
                total_exploits = sum(exploit_counts.values())
                for exploit in sorted(exploit_counts.keys(), key=lambda x: -exploit_counts[x]):
                    c = exploit_counts[exploit]
                    pct = 100 * c / total_exploits if total_exploits > 0 else 0
                    f.write(f"  {exploit:10s}: {c:6d} ({pct:5.1f}%)\n")
            else:
                f.write("  (no exploits)\n")
            
            f.write("\nTop Targeted Hosts:\n")
            host_totals = {}
            for host, actions in stats['red']['host_counts'].items():
                host_totals[host] = sum(actions.values())
            
            if host_totals:
                for host in sorted(host_totals.keys(), key=lambda x: -host_totals[x])[:5]:
                    f.write(f"  {host:15s}: {host_totals[host]:6d}\n")
            else:
                f.write("  (no host targets)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare action distributions across IPPO variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_action_distributions.py --auto
  python compare_action_distributions.py --auto --n-episodes 200
  python compare_action_distributions.py --auto --total-timesteps 10000000
        """
    )
    
    parser.add_argument("--auto", action="store_true", default=True, 
                       help="Auto-discover models (default)")
    parser.add_argument("--model-dir", type=str, default="ppo_models",
                       help="Base directory for models")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
                       help="Training timesteps (for model discovery)")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="Number of evaluation episodes per variant")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic actions")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Multi-Exploit IPPO - Action Distribution Comparison")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Episodes per variant: {args.n_episodes}")
    print(f"Device: {device}")
    print()
    
    # Discover and evaluate all variants
    all_stats = {}
    
    for variant in VARIANTS:
        print(f"\n{'='*40}")
        print(f"Processing: {variant.upper()}")
        print(f"{'='*40}")
        
        models = find_models(base_dir, variant, args.total_timesteps)
        
        if models is None:
            print(f"  No models found for {variant}")
            continue
        
        print(f"  Found models:")
        for m in models:
            print(f"    - {m}")
        
        stats = evaluate_variant(
            variant=variant,
            model_paths=models,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            device=device
        )
        
        all_stats[variant] = stats
        print(f"  Mean Reward: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
    
    if not all_stats:
        print("\nNo models found! Train models first using train_multi_exploit_ippo.py")
        return
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating Comparison Visualizations")
    print(f"{'='*60}")
    
    # Blue agent visualizations
    print("\nBlue Agent Visualizations:")
    plot_action_type_comparison(all_stats, output_dir / "action_type_comparison.png")
    plot_action_type_bars(all_stats, output_dir / "action_type_bars.png")
    plot_decoy_comparison(all_stats, output_dir / "decoy_comparison.png")
    plot_host_heatmaps(all_stats, output_dir / "host_heatmaps.png")
    plot_performance_summary(all_stats, output_dir / "performance_summary.png")
    plot_agent_specialization(all_stats, output_dir / "agent_specialization.png")
    
    # Red agent visualizations
    print("\nRed Agent Visualizations:")
    plot_red_action_distribution(all_stats, output_dir / "red_action_distribution.png")
    plot_red_action_bars(all_stats, output_dir / "red_action_bars.png")
    plot_red_exploit_distribution(all_stats, output_dir / "red_exploit_distribution.png")
    plot_red_host_heatmap(all_stats, output_dir / "red_host_heatmap.png")
    
    # Summary report
    print("\nGenerating report:")
    save_detailed_report(all_stats, output_dir / "comparison_report.txt")
    
    print(f"\n{'='*60}")
    print("Comparison Complete!")
    print(f"{'='*60}")
    print(f"\nGenerated files in {output_dir}/:")
    print("\nBlue Agent:")
    print("  - action_type_comparison.png   (Pie charts per agent)")
    print("  - action_type_bars.png         (Bar chart comparison)")
    print("  - decoy_comparison.png         (Decoy usage by agent)")
    print("  - host_heatmaps.png            (Blue target host heatmaps)")
    print("  - performance_summary.png      (Overall performance)")
    print("  - agent_specialization.png     (Who does what %)")
    print("\nRed Agent:")
    print("  - red_action_distribution.png  (Red action type pie charts)")
    print("  - red_action_bars.png          (Red action type bars)")
    print("  - red_exploit_distribution.png (Red exploit usage)")
    print("  - red_host_heatmap.png         (Red target host heatmaps)")
    print("\nReport:")
    print("  - comparison_report.txt        (Detailed text report)")


if __name__ == "__main__":
    main()

