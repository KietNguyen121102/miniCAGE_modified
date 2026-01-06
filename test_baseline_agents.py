"""
Test script for baseline agents in the Multi-Exploit CAGE environment.

Demonstrates:
1. Red baseline agent following strategic attack path
2. Blue baseline agents with different strategies
3. Performance comparison across strategies
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multi_exploit_env import MultiExploitCAGE, action_mapping_multi_exploit
from multi_exploit_bline_agents import RedBaselineAgent, BlueBaselineAgent


def test_red_agent():
    """Test the red baseline agent against a fixed blue agent."""
    print("=" * 80)
    print("TESTING RED BASELINE AGENT")
    print("=" * 80)
    
    strategies = ['high_success', 'random', 'brute_only']
    results = {}
    num_episodes = 100
    fixed_blue_strategy = 'proactive'

    for red_strategy in strategies:
        print(f"\nRed Strategy: {red_strategy} (vs Blue: {fixed_blue_strategy})")
        print("-" * 80)
        
        episode_rewards = []
        episode_steps = []
        
        for _ in range(num_episodes):
            env = MultiExploitCAGE(num_envs=1)
            obs, info = env.reset()

            red_agent = RedBaselineAgent(strategy=red_strategy)
            blue_agent = BlueBaselineAgent(strategy=fixed_blue_strategy)
            
            total_red_reward = 0
            step_count = 0
            
            for step in range(100):
                red_obs = obs['Red']
                red_action = red_agent.get_action(red_obs)
                
                blue_obs = obs['Blue']
                masks = env.get_mask(blue_obs, env.current_decoys)
                blue_action = blue_agent.get_action(blue_obs[0], masks['Blue'][0])
                
                obs, reward, done, info = env.step(red_action, blue_action)
                
                total_red_reward += reward['Red'][0, 0]
                step_count += 1

                if np.all(done):
                    break
            
            episode_rewards.append(total_red_reward)
            episode_steps.append(step_count)
            
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_steps = np.mean(episode_steps)
        
        results[red_strategy] = (avg_reward, std_reward, avg_steps)
        
        print(f"  Average reward: {avg_reward: >7.2f} ± {std_reward:.2f}")
        print(f"  Average steps:   {avg_steps:.2f}")

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<15} {'Avg Reward':<20} {'Avg Steps':<10}")
    print("-" * 80)
    for strategy, (mean, std, steps) in results.items():
        reward_str = f"{mean:.2f} ± {std:.2f}"
        print(f"{strategy:<15} {reward_str:<20} {steps:<10.2f}")
    print()


def test_blue_agent():
    """Test blue baseline agent with different strategies."""
    print("\n" + "=" * 80)
    print("TESTING BLUE BASELINE AGENTS")
    print("=" * 80)
    
    strategies = ['sleep', 'reactive', 'proactive', 'smarter_proactive', 'mixed', 'random']
    results = {}
    action_space = action_mapping_multi_exploit()
    
    num_episodes = 100
    fixed_red_strategy = 'high_success'
    
    for blue_strategy in strategies:
        print(f"\nBlue Strategy: {blue_strategy} (vs Red: {fixed_red_strategy})")
        print("-" * 80)
        
        episode_rewards = []
        episode_steps = []
        
        for episode in tqdm(range(num_episodes), desc="  Simulating episodes"):
            env = MultiExploitCAGE(num_envs=1)
            
            red_agent = RedBaselineAgent(strategy=fixed_red_strategy)
            blue_agent = BlueBaselineAgent(strategy=blue_strategy)
            
            obs, info = env.reset()
            red_agent.reset()
            blue_agent.reset()
            
            total_blue_reward = 0
            step_count = 0
            max_steps = 100
            
            for step in range(max_steps):
                red_obs = obs['Red']
                blue_obs = obs['Blue']
                
                masks = env.get_mask(blue_obs, env.current_decoys)
                
                red_action = red_agent.get_action(red_obs)
                blue_action = blue_agent.get_action(blue_obs[0], masks['Blue'][0])
                
                obs, reward, done, info = env.step(red_action, blue_action)
                
                total_blue_reward += reward['Blue'][0, 0]
                step_count += 1

                if np.all(done):
                    break
            
            episode_rewards.append(total_blue_reward)
            episode_steps.append(step_count)
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_steps = np.mean(episode_steps)
        
        results[blue_strategy] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_steps': avg_steps
        }
        
        print(f"  Average reward: {avg_reward: >7.2f} ± {std_reward:.2f}")
        print(f"  Average steps:   {avg_steps:.2f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<15} {'Avg Reward':<20} {'Avg Steps':<10}")
    print("-" * 80)
    for strategy, res in results.items():
        reward_str = f"{res['avg_reward']:.2f} ± {res['std_reward']:.2f}"
        print(f"{strategy:<15} {reward_str:<20} {res['avg_steps']:<10.2f}")
    print()


def test_action_selection():
    """Test specific action selection for red agent."""
    print("\n" + "=" * 80)
    print("RED AGENT ACTION SELECTION DEMONSTRATION")
    print("=" * 80)
    
    env = MultiExploitCAGE(num_envs=1, remove_bugs=False)
    
    strategies = ['high_success', 'random', 'brute_only']
    
    print("\nTesting exploit selection on user hosts:")
    print("-" * 80)
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        red_agent = RedBaselineAgent(strategy=strategy)
        
        # Simulate state where user1 is scanned
        obs, info = env.reset()
        red_agent.reset()
        
        # Progress to exploit phase
        # Manually set agent to exploit state
        red_agent.action = 2
        red_agent.first_user_host = 9  # user1
        red_agent.last_host = 9
        
        # Simulate scanned state for user1
        # Set obs so user1 shows [1, 0, 0] (scanned, not exploited)
        obs_state = obs['Red'][0, 1:].reshape(13, 3)
        obs_state[9] = [1, 0, 0]  # user1 scanned
        obs['Red'][0, 1:] = obs_state.reshape(-1)
        
        # Get exploit action
        action = red_agent.exploit_remote_service_user(9, obs['Red'][0])
        if action:
            action_name = env.action_mapping['Red'][action]
            print(f"  Selected: {action_name}")


def demonstrate_full_episode(red_strategy='default', blue_strategy='random'):
    """
    Runs and prints a single, full episode to demonstrate agent interactions.
    """
    print("=" * 80)
    print("FULL EPISODE DEMONSTRATION")
    print("=" * 80)

    env = MultiExploitCAGE(num_envs=1)
    obs, info = env.reset()

    red_agent = RedBaselineAgent(strategy=red_strategy)
    blue_agent = BlueBaselineAgent(strategy=blue_strategy)
    
    action_space = action_mapping_multi_exploit()

    for step in range(100):
        # Red Agent's turn
        red_obs = obs['Red']
        red_action = red_agent.get_action(red_obs)
        red_action_name = action_space['Red'][red_action[0, 0]]

        # Blue Agent's turn
        blue_obs = obs['Blue']
        masks = env.get_mask(blue_obs, env.current_decoys)
        blue_action = blue_agent.get_action(blue_obs[0], masks['Blue'][0])
        blue_action_name = action_space['Blue'][blue_action[0, 0]]

        print(f"Step {step+1: >3} | Red action: {red_action_name:<25} | Blue action: {blue_action_name:<25}")

        obs, reward, done, info = env.step(red_action, blue_action)

        if np.all(done):
            print("\nEpisode finished.")
            break


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-EXPLOIT CAGE - BASELINE AGENTS TEST")
    print("=" * 80)
    
    # Test red agent
    test_red_agent()
        
    # Test blue agents
    test_blue_agent()
    
    # Full episode demonstration
    demonstrate_full_episode()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\nBaseline agents are ready for training comparisons!")
    print("\nUsage:")
    print("  from multi_exploit_bline_agents import RedBaselineAgent, BlueBaselineAgent")
    print("  red_agent = RedBaselineAgent(strategy='high_success')")
    print("  blue_agent = BlueBaselineAgent(strategy='mixed')")
    print("=" * 80 + "\n")

