# parlenv.py
# from parl.env import MAEnv
from .env import World
import numpy as np
import gymnasium as gym

class PARLSumoEnv:
    def __init__(self, config):
        self.env = World(
            config["sumo_config"], 
            interface=config.get("interface", "traci"),
            seed=config.get("seed", None),
            sync_mode=config.get("sync_mode", False),
            obs_to_subscribe=config["obs_to_subscribe"],
            reward_to_subscribe=config["reward_to_subscribe"],
            algorithm_name=config.get("algorithm_name", None),
            normalize_observation=config.get("normalize_observation", False),
            norm_params=config.get("norm_params", {}),
            # ✅ 添加这3行：传递奖励配置
            reward_weights = config.get("reward_weights", [1.0]),
            reward_scale = config.get("reward_scale", 1.0),
            reward_clip_range = config.get("reward_clip_range", None),
            sim_max_steps=config.get("sim_max_steps", 1000)
        )
        self.agent_ids = list(self.env.traffic_light_ids)
        
        self.closed_edges = config.get("closed_edges", [])


        # Build observation and action spaces for each agent
        self._observation_spaces = {}
        self._action_spaces = {}
        
        for agent_id in self.agent_ids:
            # Get spaces from World
            obs_space = self.env.observation_spaces(agent_id)
            act_space = self.env.action_spaces(agent_id)
            
            # Ensure float32 for observations
            self._observation_spaces[agent_id] = gym.spaces.Box(
                low=obs_space.low,
                high=obs_space.high,
                shape=obs_space.shape,
                dtype=np.float32
            )
            self._action_spaces[agent_id] = act_space
    
    def reset(self):
        """Returns: obs_dict"""
        obs = self.env.reset()
        # ✅ 在reset之后封闭指定道路
        if self.closed_edges:
            print(f"\n🚧 正在封闭道路并重新规划受影响车辆的路线...")
            self.env.close_edges(self.closed_edges)
            print()
        # 处理空字典情况
        if not obs:
            # 如果 reset 返回空字典，执行一次空 step
            obs, _, _, _ = self.env.step({})
        # return {agent_id: observation.astype(np.float32) 
        #         for agent_id, observation in obs.items()}
        # ✅ FIX: Safer numpy conversion with explicit copy
        result = {}
        for agent_id, observation in obs.items():
            if isinstance(observation, np.ndarray):
                result[agent_id] = observation.astype(np.float32, copy=True)
            else:
                result[agent_id] = np.array(observation, dtype=np.float32)
        return result
    
    def step(self, action_dict):
        """
        Returns: obs_dict, reward_dict, done, info_dict
        """
        obs, rewards, dones, infos = self.env.step(action_dict)
        
        # Convert to float32
        # obs = {agent_id: observation.astype(np.float32) 
        #        for agent_id, observation in obs.items()}
        # ✅ FIX: Safer numpy conversion
        result_obs = {}
        for agent_id, observation in obs.items():
            if isinstance(observation, np.ndarray):
                result_obs[agent_id] = observation.astype(np.float32, copy=True)
            else:
                result_obs[agent_id] = np.array(observation, dtype=np.float32)
        # PARL expects scalar rewards
        rewards = {agent_id: float(reward.item()) if hasattr(reward, 'item') else float(reward)
                for agent_id, reward in rewards.items()}
        
        # Extract episode done
        done = dones.get("__all__", False)
        
        return result_obs, rewards, done, infos
    
    def get_agent_ids(self):
        """Return list of agent IDs"""
        return self.agent_ids
    
    def observation_space(self, agent_id):
        """Return observation space for a specific agent"""
        return self._observation_spaces[agent_id]
    
    def action_space(self, agent_id):
        """Return action space for a specific agent"""
        return self._action_spaces[agent_id]
    
    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()
    # ============additional functions===================
    def get_closable_edges(self):
        """返回可以安全封闭的道路ID列表"""
        return self.env.get_closable_edges()

    def get_all_edges(self):
        """返回所有道路ID（包括不能封闭的）"""
        return self.env.all_roads
        
    def set_traffic_scale(self, scale: float):
        """
        设置流量缩放因子（便捷接口）
        
        Args:
            scale: 流量缩放因子
        """
        self.env.set_traffic_scale(scale)
    
    def get_traffic_scale(self):
        """获取当前流量缩放因子"""
        return self.env.get_traffic_scale()