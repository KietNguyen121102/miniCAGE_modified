"""
Multi-agent environment wrapper where both agents:
- Observe the full network state
- Can take actions on any host
- Learn to coordinate through training
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys, os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from minimal import SimplifiedCAGE, HOSTS
from red_bline_agent import B_line_minimal


class IPPOSharedObsEnv:
    """
    Multi-agent environment for 2 Blue agents with:
    - Shared full observation of the network
    - Same action space for both agents
    - Simultaneous action execution
    """
    
    def __init__(
        self,
        red_policy: str = "bline",
        remove_bugs: bool = True,
        max_steps: int = 100,
        seed: int = None,
        action_resolution: str = "sequential"  # "sequential", "first_valid", or "both"
    ):
        self.sim = SimplifiedCAGE(num_envs=1, remove_bugs=remove_bugs)
        self.max_steps = max_steps
        self.steps_done = 0
        self.num_agents = 2
        self.action_resolution = action_resolution
        
        # Both agents share the same observation and action spaces
        obs_dim = 6 * len(HOSTS)  # 78 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 0 (sleep) + 4 actions * 13 hosts = 53 actions
        # 0: sleep
        # 1-13: analyse hosts 0-12
        # 14-26: decoy hosts 0-12
        # 27-39: remove hosts 0-12
        # 40-52: restore hosts 0-12
        num_actions = 1 + 4 * len(HOSTS)
        self.action_space = spaces.Discrete(num_actions)
        
        # Red agent
        if red_policy.lower() in {"bline", "b_line", "b_line_minimal"}:
            self.red_agent = B_line_minimal()
        else:
            raise ValueError(f"Unknown red agent '{red_policy}'")
        
        self._red_obs = None
        self.seed_value = seed
    
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
    
    def _resolve_actions(self, agent_actions):
        """
        Resolve conflicts when both agents want to act.
        
        Args:
            agent_actions: [action_agent_0, action_agent_1]
        
        Returns:
            List of actions to execute in the environment
        """
        action_0, action_1 = agent_actions
        
        if self.action_resolution == "sequential":
            # Execute both actions in sequence (agent 0, then agent 1)
            # Filter out sleep actions
            actions_to_execute = []
            if action_0 > 0:
                actions_to_execute.append(action_0)
            if action_1 > 0:
                actions_to_execute.append(action_1)
            
            return actions_to_execute if actions_to_execute else [0]
        
        elif self.action_resolution == "first_valid":
            # Execute first non-sleep action
            if action_0 > 0:
                return [action_0]
            elif action_1 > 0:
                return [action_1]
            else:
                return [0]
        
        elif self.action_resolution == "both":
            # Always try to execute both (even if they conflict)
            if action_0 > 0 and action_1 > 0:
                return [action_0, action_1]
            elif action_0 > 0:
                return [action_0]
            elif action_1 > 0:
                return [action_1]
            else:
                return [0]
        
        else:
            raise ValueError(f"Unknown action resolution: {self.action_resolution}")
    
    def step(self, agent_actions):
        """
        Take a step with actions from both agents.
        
        Args:
            agent_actions: [action_agent_0, action_agent_1]
        
        Returns:
            observations: [obs_agent_0, obs_agent_1] (same observation)
            rewards: [reward_agent_0, reward_agent_1] (shared team reward)
            dones: [done_agent_0, done_agent_1]
            infos: [info_agent_0, info_agent_1]
        """
        self.steps_done += 1
        
        # Red acts
        red_action = self.red_agent.get_action(self._red_obs)
        red_action = red_action.astype(np.int32)
        
        # Resolve multi-agent actions
        actions_to_execute = self._resolve_actions(agent_actions)
        
        # Execute all actions sequentially
        cumulative_reward = 0.0
        for blue_action_int in actions_to_execute:
            blue_action = np.array([[blue_action_int]], dtype=np.int32)
            
            obs_dict, reward_dict, terminated, info = self.sim.step(
                red_action=red_action,
                blue_action=blue_action,
                red_agent=self.red_agent
            )
            
            cumulative_reward += float(reward_dict["Blue"][0][0])
            
            # Red doesn't act again in the same timestep
            # Red action already happened, so set to sleep for subsequent blue actions
            red_action = np.array([[0]], dtype=np.int32)
        
        self._red_obs = obs_dict["Red"][0]
        full_blue_obs = obs_dict["Blue"][0].astype(np.float32)
        
        # Both agents observe the same state
        agent_observations = [full_blue_obs.copy(), full_blue_obs.copy()]
        
        # Shared team reward
        agent_rewards = [cumulative_reward, cumulative_reward]
        
        done = self.steps_done >= self.max_steps
        agent_dones = [done, done]
        
        # Info for each agent
        info["agent_0_action"] = int(agent_actions[0])
        info["agent_1_action"] = int(agent_actions[1])
        info["actions_executed"] = [int(a) for a in actions_to_execute]
        agent_infos = [info.copy(), info.copy()]
        
        return agent_observations, agent_rewards, agent_dones, agent_infos