"""
Multi-agent environment wrapper where:
- Agent 0 acts first
- Agent 1 observes intermediate state and then acts
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys, os

sys.path.append(os.path.expanduser("~/Documents/Coding/CybORG_plus_plus"))
from minimal import SimplifiedCAGE, HOSTS
from red_bline_agent import B_line_minimal


class IPPOSequentialObsEnv:
    """
    Multi-agent environment where agent 1 observes the state after agent 0's action.
    This requires a two-stage step process.
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
        
        # Both agents share the same observation and action spaces
        obs_dim = 6 * len(HOSTS)  # 78 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 0 (sleep) + 4 actions * 13 hosts = 53 actions
        num_actions = 1 + 4 * len(HOSTS)
        self.action_space = spaces.Discrete(num_actions)
        
        # Red agent
        if red_policy.lower() in {"bline", "b_line", "b_line_minimal"}:
            self.red_agent = B_line_minimal()
        else:
            raise ValueError(f"Unknown red agent '{red_policy}'")
        
        self._red_obs = None
        self.seed_value = seed
        
        # Track intermediate state
        self._intermediate_obs = None
        self._intermediate_reward = None
        self._intermediate_info = None
    
    def reset(self, seed=None):
        """Reset environment and return initial observation for agent 0 only."""
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
        
        self.red_agent.reset()
        self.steps_done = 0
        self._intermediate_obs = None
        
        obs_dict, info = self.sim.reset()
        self._red_obs = obs_dict["Red"][0]
        
        full_blue_obs = obs_dict["Blue"][0].astype(np.float32)
        
        # Return observation for agent 0
        return full_blue_obs
    
    def step_agent_0(self, action_0):
        """
        Execute agent 0's action and return intermediate observation for agent 1.
        
        Args:
            action_0: Action chosen by agent 0
            
        Returns:
            intermediate_obs: Observation for agent 1 after agent 0's action
            reward_0: Reward for agent 0's action
            done: Whether episode is done
            info: Information dict
        """
        self.steps_done += 1
        
        # Red acts first
        red_action = self.red_agent.get_action(self._red_obs)
        red_action = red_action.astype(np.int32)
        
        # Agent 0 acts
        blue_action_0 = np.array([[action_0]], dtype=np.int32)
        
        obs_dict, reward_dict, terminated, info = self.sim.step(
            red_action=red_action,
            blue_action=blue_action_0,
            red_agent=self.red_agent
        )
        
        # Store intermediate state
        self._red_obs = obs_dict["Red"][0]
        self._intermediate_obs = obs_dict["Blue"][0].astype(np.float32)
        self._intermediate_reward = float(reward_dict["Blue"][0][0])
        self._intermediate_info = info.copy()
        
        done = self.steps_done >= self.max_steps
        
        info["agent_0_action"] = int(action_0)
        
        # Return intermediate observation for agent 1
        return self._intermediate_obs.copy(), self._intermediate_reward, done, info
    
    def step_agent_1(self, action_1):
        """
        Execute agent 1's action based on intermediate state.
        
        Args:
            action_1: Action chosen by agent 1 (based on intermediate obs)
            
        Returns:
            observations: [obs_agent_0, obs_agent_1] - final state
            rewards: [reward_agent_0, reward_agent_1]
            dones: [done_agent_0, done_agent_1]
            infos: [info_agent_0, info_agent_1]
        """
        if self._intermediate_obs is None:
            raise RuntimeError("Must call step_agent_0() before step_agent_1()")
        
        # Red sleeps, agent 1 acts
        red_action_sleep = np.array([[0]], dtype=np.int32)
        blue_action_1 = np.array([[action_1]], dtype=np.int32)
        
        obs_dict, reward_dict, terminated, info = self.sim.step(
            red_action=red_action_sleep,
            blue_action=blue_action_1,
            red_agent=self.red_agent
        )
        
        reward_agent_1 = float(reward_dict["Blue"][0][0])
        
        # Update final state
        self._red_obs = obs_dict["Red"][0]
        final_blue_obs = obs_dict["Blue"][0].astype(np.float32)
        
        # Both agents get final observation for next timestep
        agent_observations = [final_blue_obs.copy(), final_blue_obs.copy()]
        
        # Return rewards from each agent's action
        agent_rewards = [self._intermediate_reward, reward_agent_1]
        
        done = self.steps_done >= self.max_steps
        agent_dones = [done, done]
        
        # Combine info
        info["agent_1_action"] = int(action_1)
        info["agent_0_reward"] = self._intermediate_reward
        info["agent_1_reward"] = reward_agent_1
        agent_infos = [info.copy(), info.copy()]
        
        # Clear intermediate state
        self._intermediate_obs = None
        self._intermediate_reward = None
        
        return agent_observations, agent_rewards, agent_dones, agent_infos
    
    def step(self, agent_actions):
        """
        Convenience method for compatibility.
        WARNING: This still requires both actions simultaneously,
        so agent 1 can't actually observe intermediate state here.
        Use step_agent_0() and step_agent_1() for true sequential observation.
        """
        action_0, action_1 = agent_actions
        
        # Execute agent 0
        intermediate_obs, reward_0, done_partial, info_partial = self.step_agent_0(action_0)
        
        # Execute agent 1
        final_obs, rewards, dones, infos = self.step_agent_1(action_1)
        
        return final_obs, rewards, dones, infos