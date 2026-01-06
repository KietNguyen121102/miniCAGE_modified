"""
Compares the actions of all blue agent baseline strategies
in response to a series of specific, strategic red agent actions.
"""

import numpy as np
import copy
from multi_exploit_env import MultiExploitCAGE, action_mapping_multi_exploit
from multi_exploit_bline_agents import BlueBaselineAgent

def run_comparison(env, scenario_name, red_action_name, blue_strategies, action_space):
    """
    For a given environment state, executes a single red action and records
    the response of each blue agent strategy.
    """
    print("-" * 80)
    print(f"{scenario_name}: Red's action is '{red_action_name}'")
    print("-" * 80)

    # Create a reverse mapping to get action index from name
    red_action_to_idx = {name: i for i, name in enumerate(action_space['Red'])}
    red_action_idx = red_action_to_idx[red_action_name]
    red_action = np.array([[red_action_idx]])

    results = {}
    for strategy in blue_strategies:
        # Each strategy gets an identical copy of the environment state
        env_copy = copy.deepcopy(env)
        blue_agent = BlueBaselineAgent(strategy=strategy)

        # Apply the red action. Blue agent will respond to the outcome.
        obs, _, _, _ = env_copy.step(red_action, np.array([[0]])) # Blue sleeps this turn

        # Get the blue agent's response to the new state
        blue_obs = obs['Blue']
        masks = env_copy.get_mask(blue_obs, env_copy.current_decoys)
        blue_action_idx = blue_agent.get_action(blue_obs[0], masks['Blue'][0])[0, 0]
        action_name = action_space['Blue'][blue_action_idx]

        results[strategy] = action_name

    # Print results for this scenario
    for strategy, action in results.items():
        print(f"{strategy:<20}: {action}")
    print("=" * 80 + "\n")


def compare_strategic_scenarios():
    """
    Sets up and runs multiple strategic scenarios to compare blue agent responses.
    """
    print("=" * 80)
    print("BLUE AGENT STRATEGY COMPARISON")
    print("=" * 80)

    strategies = [
        'sleep',
        'reactive',
        'proactive',
        'smarter_proactive',
        'mixed',
        'random'
    ]
    action_space = action_mapping_multi_exploit()
    
    # --- Scenario 1: Initial State, Red performs a network scan ---
    env_s1 = MultiExploitCAGE(num_envs=1)
    env_s1.reset()
    run_comparison(
        env=env_s1,
        scenario_name="Scenario 1: Initial Scan",
        red_action_name="network_user4",
        blue_strategies=strategies,
        action_space=action_space
    )
    
    # --- Scenario 2: Red has scanned user4 and now attempts an exploit ---
    env_s2 = MultiExploitCAGE(num_envs=1)
    env_s2.reset()
    # Setup: Manually perform the initial scan to prepare the state
    red_action_to_idx = {name: i for i, name in enumerate(action_space['Red'])}
    scan_action_idx = red_action_to_idx["network_user4"]
    env_s2.step(np.array([[scan_action_idx]]), np.array([[0]])) # Red scans, blue sleeps
    
    run_comparison(
        env=env_s2,
        scenario_name="Scenario 2: Red Exploits Scanned Host",
        red_action_name="exploit_Brute_ent2",
        blue_strategies=strategies,
        action_space=action_space
    )


if __name__ == '__main__':
    compare_strategic_scenarios()
