"""
Generates a summary table comparing the performance of all red agent
strategies against all blue agent strategies.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from multi_exploit_env import MultiExploitCAGE
from multi_exploit_bline_agents import RedBaselineAgent, BlueBaselineAgent

def run_simulation(red_strategy, blue_strategy, num_episodes=100):
    """
    Runs a simulation for a given pair of strategies and returns the
    average red agent reward.
    """
    episode_rewards = []
    
    for _ in range(num_episodes):
        env = MultiExploitCAGE(num_envs=1)
        obs, info = env.reset()

        red_agent = RedBaselineAgent(strategy=red_strategy)
        blue_agent = BlueBaselineAgent(strategy=blue_strategy)
        
        total_red_reward = 0
        
        for step in range(100):
            red_obs = obs['Red']
            red_action = red_agent.get_action(red_obs)
            
            blue_obs = obs['Blue']
            masks = env.get_mask(blue_obs, env.current_decoys)
            blue_action = blue_agent.get_action(blue_obs[0], masks['Blue'][0])
            
            obs, reward, done, info = env.step(red_action, blue_action)
            
            total_red_reward += reward['Red'][0, 0]

            if np.all(done):
                break
        
        episode_rewards.append(total_red_reward)
        
    return np.mean(episode_rewards)

def generate_performance_matrix():
    """
    Generates and prints a table of red agent rewards for every
    red vs. blue strategy matchup.
    """
    red_strategies = ['high_success', 'random', 'brute_only']
    blue_strategies = ['sleep', 'reactive', 'proactive', 'smarter_proactive', 'mixed', 'random']
    
    # Use a smaller number of episodes for a quicker result, can be increased for accuracy
    num_episodes_per_matchup = 50 

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(index=red_strategies, columns=blue_strategies, dtype=float)
    
    print("=" * 80)
    print("GENERATING RED VS. BLUE PERFORMANCE MATRIX")
    print(f"(Running {num_episodes_per_matchup} episodes per matchup)")
    print("=" * 80)

    # Use tqdm for a single, overarching progress bar
    total_simulations = len(red_strategies) * len(blue_strategies)
    with tqdm(total=total_simulations, desc="  Simulating all matchups") as pbar:
        for red_strat in red_strategies:
            for blue_strat in blue_strategies:
                avg_reward = run_simulation(red_strat, blue_strat, num_episodes_per_matchup)
                results_df.loc[red_strat, blue_strat] = avg_reward
                pbar.update(1)

    print("\n" + "=" * 80)
    print("Red Agent Average Reward Matrix")
    print("-" * 80)
    print("Rows: Red Agent Strategy | Columns: Blue Agent Strategy")
    print("Higher scores are better for the Red Agent.")
    print("-" * 80)
    
    # Configure pandas to display all columns
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 20)
    
    # Format the DataFrame for better readability
    formatted_df = results_df.map(lambda x: f"{x: >8.2f}")
    print(formatted_df)
    print("=" * 80)


if __name__ == '__main__':
    generate_performance_matrix()
