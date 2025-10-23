"""
Multi-agent environment where agents have restricted action spaces:
- Agent 0: Analyze, Remove, Restore (and Sleep)
- Agent 1: Analyze, Decoy (and Sleep)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Add parent directory (mini_CAGE) to path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from minimal import SimplifiedCAGE, HOSTS
from red_bline_agent import B_line_minimal

class RestrictedIPPOEnv:
    """
    Multi-agent environment with restricted action spaces:
    - Agent 0 (Defender): Analyze, Remove, Restore
    - Agent 1 (Decoy Manager): Analyze, Decoy
    """
    
    def __init__(
        self,
        red_policy: str = "bline",
        remove_bugs: bool = True,
        max_steps: int = 100,
        seed: int = None,
    ):
        self.sim = SimplifiedCAGE(num_envs=1, remove_bugs=remove_bugs)
        self.max_steps = max_steps
        self.steps_done = 0
        self.num_agents = 2
        
        # Shared observation space (both agents see full state)
        obs_dim = 6 * len(HOSTS)  # 78 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define action mappings for each agent
        # Full action space: 0 (sleep), 1-13 (analyze), 14-26 (decoy), 27-39 (remove), 40-52 (restore)
        
        # Agent 0: Sleep + Analyze + Remove + Restore = 1 + 13 + 13 + 13 = 40 actions
        self.agent_0_actions = [0]  # sleep
        self.agent_0_actions.extend(range(1, 14))   # analyze (1-13)
        self.agent_0_actions.extend(range(27, 40))  # remove (27-39)
        self.agent_0_actions.extend(range(40, 53))  # restore (40-52)
        
        # Agent 1: Sleep + Analyze + Decoy = 1 + 13 + 13 = 27 actions
        self.agent_1_actions = [0]  # sleep
        self.agent_1_actions.extend(range(1, 14))   # analyze (1-13)
        self.agent_1_actions.extend(range(14, 27))  # decoy (14-26)
        
        # Create action spaces
        self.agent_0_action_space = spaces.Discrete(len(self.agent_0_actions))
        self.agent_1_action_space = spaces.Discrete(len(self.agent_1_actions))
        
        # For compatibility, also provide a list of action spaces
        self.action_spaces = [self.agent_0_action_space, self.agent_1_action_space]
        
        # Red agent
        if red_policy.lower() in {"bline", "b_line", "b_line_minimal"}:
            self.red_agent = B_line_minimal()
        else:
            raise ValueError(f"Unknown red agent '{red_policy}'")
        
        self._red_obs = None
        self.seed_value = seed
        
        print(f"Agent 0 action space: {len(self.agent_0_actions)} actions (Analyze, Remove, Restore)")
        print(f"Agent 1 action space: {len(self.agent_1_actions)} actions (Analyze, Decoy)")
    
    def _map_action_to_env(self, agent_idx, action_idx):
        """
        Map agent's action index to the environment's full action space.
        
        Args:
            agent_idx: 0 or 1
            action_idx: Action index from agent's restricted action space
            
        Returns:
            Full environment action index
        """
        if agent_idx == 0:
            return self.agent_0_actions[action_idx]
        else:
            return self.agent_1_actions[action_idx]
    
    def get_action_space(self, agent_idx):
        """Get action space for specific agent."""
        return self.action_spaces[agent_idx]
    
    def reset(self, seed=None):
        """Reset environment and return initial observation for both agents."""
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
        
        self.red_agent.reset()
        self.steps_done = 0
        
        obs_dict, info = self.sim.reset()
        self._red_obs = obs_dict["Red"][0]
        
        full_blue_obs = obs_dict["Blue"][0].astype(np.float32)
        
        # Both agents get the same observation
        agent_observations = [full_blue_obs.copy(), full_blue_obs.copy()]
        
        return agent_observations
    
    def step(self, agent_actions):
        """
        Take a step with actions from both agents.
        
        Args:
            agent_actions: [action_agent_0, action_agent_1] (in their restricted spaces)
        
        Returns:
            observations: [obs_agent_0, obs_agent_1]
            rewards: [reward_agent_0, reward_agent_1]
            dones: [done_agent_0, done_agent_1]
            infos: [info_agent_0, info_agent_1]
        """
        self.steps_done += 1
        
        # Red acts
        red_action = self.red_agent.get_action(self._red_obs)
        red_action = red_action.astype(np.int32)
        
        # Map agent actions to full environment action space
        env_action_0 = self._map_action_to_env(0, agent_actions[0])
        env_action_1 = self._map_action_to_env(1, agent_actions[1])
        
        # Execute both actions sequentially (if not sleep)
        cumulative_reward = 0.0
        actions_executed = []
        
        for agent_idx, env_action_int in enumerate([env_action_0, env_action_1]):
            if env_action_int > 0:  # Skip sleep actions
                blue_action = np.array([[env_action_int]], dtype=np.int32)
                
                obs_dict, reward_dict, terminated, info = self.sim.step(
                    red_action=red_action,
                    blue_action=blue_action,
                    red_agent=self.red_agent
                )
                
                cumulative_reward += float(reward_dict["Blue"][0][0])
                actions_executed.append((agent_idx, env_action_int))
                
                # Red doesn't act again in the same timestep
                red_action = np.array([[0]], dtype=np.int32)
        
        # If both agents slept, still need to step the environment
        if not actions_executed:
            blue_action = np.array([[0]], dtype=np.int32)
            obs_dict, reward_dict, terminated, info = self.sim.step(
                red_action=red_action,
                blue_action=blue_action,
                red_agent=self.red_agent
            )
            cumulative_reward = float(reward_dict["Blue"][0][0])
        
        self._red_obs = obs_dict["Red"][0]
        full_blue_obs = obs_dict["Blue"][0].astype(np.float32)
        
        # Both agents observe the same state
        agent_observations = [full_blue_obs.copy(), full_blue_obs.copy()]
        
        # Shared team reward
        agent_rewards = [cumulative_reward, cumulative_reward]
        
        done = self.steps_done >= self.max_steps
        agent_dones = [done, done]
        
        # Info for each agent
        info["agent_0_action_idx"] = int(agent_actions[0])
        info["agent_1_action_idx"] = int(agent_actions[1])
        info["agent_0_env_action"] = int(env_action_0)
        info["agent_1_env_action"] = int(env_action_1)
        info["actions_executed"] = actions_executed
        agent_infos = [info.copy(), info.copy()]
        
        return agent_observations, agent_rewards, agent_dones, agent_infos