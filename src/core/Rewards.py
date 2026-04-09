"""rewards for traffic signals."""


import numpy as np
from .Observations import ObservationFunction
# class ObservationFunction:
#     """Abstract base class for observation functions."""

#     def __init__(self, ts: Intersection, world: World):
#         """Initialize observation function."""
#         self.ts = ts
#         self.world = world

    # @abstractmethod
    # def __call__(self):
    #     """Subclasses must override this method."""
    #     pass


class GetRewards(ObservationFunction):
    """Default observation function for traffic signals."""
    # ✅ 定义不同奖励类型的配置
    REWARD_CONFIGS = {
        'lane_waiting_count': {
            'description': '车道等待车辆数',
            'level': 'lane',  # ✅ 添加level
            'use_difference': False,
            'negative': True,
            'scale': 1.0,
            'clip_range': None
        },
        'lane_waiting_time_count': {
            'description': '车道等待时间总和',
            'level': 'lane',  # ✅ 添加
            'use_difference': False,
            'negative': True,
            'scale': 1.0,
            'clip_range': None
        },
        'lane_vehicle_count': {
            'description': '车道车辆总数',
            'use_difference': False,
            'negative': False,
            'scale': 1.0,
            'clip_range': None
        },
        'avg_waiting_time_per_vehicle': {
            'description': '平均每车等待时间',
            'use_difference': False,
            'negative': True,
            'scale': 1.0,
            'clip_range': (-10.0, 0.0)
        },
        # ✅ 新增：pressure（PressLight 使用）
        'pressure': {
            'description': '交叉口压力（入口-出口车辆数）',
            'level': 'intersection',  # ✅ 标记为 intersection-level
            'negative': True,  # 最小化压力
            'scale': 1.0,
        },
        # ✅ 新增：救护车优先奖励
        'emergency_vehicle_priority': {
            'description': 'Priority rewards for ambulances (civilian vehicles wait + ambulance penalty)',
            'level': 'intersection',
            'negative': True,
            'civilian_weight': 0.7,      # 民用车权重
            'ambulance_penalty': 5000,   # 救护车停车惩罚
            'ambulance_speed_threshold': 1.0,  # 速度阈值(m/s)
            'ambulance_type_id': 'ambulance_type',  # 救护车类型ID
            'scale': 1.0,
            'clip_range': None
        },
        'project1_std_reward': {
            'description': 'project1_std_reward: 50 - (reg_mean + K*reg_std + Z*(emg_mean + K*emg_std))',
            'level': 'intersection',
            'negative': False,  # 奖励公式已包含负号逻辑
            'K': 0.5,  # 标准差权重（可调参数）
            'Z': 1.0,  # 应急车辆惩罚倍数（可调参数）
            'base_reward': 50.0,  # 基础奖励值
            'scale': 1.0,
            'clip_range': None,
            'ambulance_type_ids': ['ambulance_type', 'emergency']  # 应急车辆类型列表
        },
    }
    
    def __init__(self, ts, world, reward_to_subscribe, in_only=True, negative=True):
        super().__init__(ts, world)
        self.lanes_road_observed = [] # two-dimensional list, each element is a list of lanes on a road
        if in_only:
            roads = self.ts.in_roads
        else:
            roads = self.ts.roads
        # ---------------------------------------------------------------------------------------------------------------
        # TODO: register it in Registry
        for r in roads:
            if not self.world.RIGHT:
                tmp = sorted(self.ts.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
            else:
                tmp = sorted(self.ts.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
            self.lanes_road_observed.append(tmp)
            # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        # subscribe functions
        # ✅ 特殊处理：emergency_vehicle_priority 需要订阅底层信息
        if 'emergency_vehicle_priority' in reward_to_subscribe:
            # 救护车奖励需要 lane_waiting_time_count
            self.world.subscribe(['lane_waiting_time_count'])
        else:
            # 其他奖励类型直接订阅
            self.world.subscribe(reward_to_subscribe)
        # self.world.subscribe(reward_to_subscribe) # obs_to_subscribe have been obs_subscribed here
        self.fns_subscribed = reward_to_subscribe # a list
        self.negative = negative
        # ✅ 新增：从world获取算法名称（用于特殊算法的奖励计算）
        self.algorithm_name = getattr(world, '_algorithm_name', 'default')
        # 为每个奖励类型维护历史值（用于差分）
        self.last_rewards = {fn: None for fn in self.fns_subscribed}
    def _compute_emergency_priority_reward(self):
        """
        计算救护车优先奖励
        对齐原项目: reward = -1 × [(civilian_penalty × 0.7) + ambulance_penalty]
        """
        config = self.REWARD_CONFIGS['emergency_vehicle_priority']
        
        # 1. 计算民用车等待时间惩罚
        waiting_time_result = self.world.info_dynamics_real_time.get('lane_waiting_time_count', {})
        civilian_penalty = 0
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                civilian_penalty += waiting_time_result.get(lane_id, 0)
        
        # 2. 计算救护车惩罚
        ambulance_penalty = 0
        try:
            eng = self.world.eng
            vehicle_list = eng.vehicle.getIDList()
            
            ambulance_type = config['ambulance_type_id']
            speed_threshold = config['ambulance_speed_threshold']
            penalty_value = config['ambulance_penalty']
            
            for veh_id in vehicle_list:
                try:
                    veh_type = eng.vehicle.getTypeID(veh_id)
                    if veh_type == ambulance_type:
                        speed = eng.vehicle.getSpeed(veh_id)
                        if speed < speed_threshold:
                            ambulance_penalty += penalty_value
                except:
                    pass
        except:
            pass
        
        # 3. 组合奖励（对齐原项目公式）
        civilian_weight = config['civilian_weight']
        raw_value = (civilian_penalty * civilian_weight) + ambulance_penalty
        
        # 4. 取负值并缩放
        reward = -raw_value / config.get('scale', 1.0)
        
        return float(reward)
    def compute_reward(self) -> np.ndarray:
        """计算奖励，支持多种奖励类型的不同处理策略"""
        if 'project1' in self.algorithm_name.lower() or 'std_dqn' in self.algorithm_name.lower():
            return self._compute_project1_std_reward()
        # 特殊处理 emergency_vehicle_priority
        if 'emergency_vehicle_priority' in self.fns_subscribed:
            return self._compute_emergency_priority_reward()
        subscribed_results = [self.world.info_dynamics_real_time[fn] for fn in self.fns_subscribed]
        # ========== MA2C 特殊处理 ==========
        if 'ma2c' in self.algorithm_name.lower():
            # ✅ 从world获取MA2C特定参数
            reward_weights = getattr(self.world, '_reward_weights', [1.0, 0.2])
            reward_scale = getattr(self.world, '_reward_scale', 2000.0)
            reward_clip_range = getattr(self.world, '_reward_clip_range', [-2.0, 2.0])
            
            # 计算各个奖励分量
            reward_components = []
            for i, fn_name in enumerate(self.fns_subscribed):
                result = subscribed_results[i]
                
                # 聚合lane级别数据（MA2C使用总和）
                fn_result = []
                for road_lanes in self.lanes_road_observed:
                    road_result = [result[lane_id] for lane_id in road_lanes]
                    fn_result.extend(road_result)
                raw_value = np.sum(fn_result)
                
                # 取负值（最小化）
                reward_components.append(-raw_value)
            
            # ✅ 加权组合: reward = -queue - 0.2*wait
            weighted_reward = sum(w * r for w, r in zip(reward_weights, reward_components))
            
            # ✅ 归一化: reward /= 2000.0
            normalized_reward = weighted_reward / reward_scale
            
            # ✅ 裁剪: clip(reward, -2, 2)
            if reward_clip_range:
                clipped_reward = np.clip(normalized_reward, 
                                        reward_clip_range[0], 
                                        reward_clip_range[1])
            else:
                clipped_reward = normalized_reward
            
            return float(clipped_reward)
        
        # ========== 默认模式：其他算法（保持原有逻辑）==========
        else:
            reward_components = []
            
            for i, fn_name in enumerate(self.fns_subscribed):
                result = subscribed_results[i]
                config = self.REWARD_CONFIGS[fn_name]
                
                if fn_name == 'pressure':
                    raw_value = result[self.ts.id]
                elif fn_name == 'lane_waiting_count':
                    # ===================聚合lane级别的数据====================
                    # lane_waiting_count 和 lane_waiting_time_count 都是lane级别数据
                    fn_result = []
                    for road_lanes in self.lanes_road_observed:
                        road_result = [result[lane_id] for lane_id in road_lanes]
                        fn_result.append(np.sum(road_result))
                    
                    # 所有路段的平均值
                    # raw_value = np.mean(fn_result)
                    # ✅ 改为总和
                    raw_value = np.sum(fn_result)
                # ✅ 新增：救护车优先奖励
                elif fn_name == 'emergency_vehicle_priority':
                    # 1. 计算民用车等待时间（从 lane_waiting_time_count 获取）
                    waiting_time_result = self.world.info_dynamics_real_time.get('lane_waiting_time_count', {})
                    civilian_penalty = 0
                    for road_lanes in self.lanes_road_observed:
                        for lane_id in road_lanes:
                            civilian_penalty += waiting_time_result.get(lane_id, 0)
                    
                    # 2. 计算救护车惩罚（检查所有车辆）
                    ambulance_penalty = 0
                    # 获取当前仿真中的所有车辆
                    eng = self.world.eng
                    vehicle_list = eng.vehicle.getIDList()
                    
                    ambulance_type = config['ambulance_type_id']
                    speed_threshold = config['ambulance_speed_threshold']
                    penalty_value = config['ambulance_penalty']
                    
                    for veh_id in vehicle_list:
                        try:
                            veh_type = eng.vehicle.getTypeID(veh_id)
                            if veh_type == ambulance_type:
                                speed = eng.vehicle.getSpeed(veh_id)
                                if speed < speed_threshold:
                                    ambulance_penalty += penalty_value
                                    # print(f"⚠️ 救护车 {veh_id} 速度 {speed:.2f} m/s < {speed_threshold} m/s")
                        except:
                            pass
                    # 3. 组合奖励
                    civilian_weight = config['civilian_weight']
                    raw_value = (civilian_penalty * civilian_weight) + ambulance_penalty
                else:
                    raise ValueError(f"Unknown reward level: {config['level']}")
                
                # 处理奖励：取负 -> 缩放 -> 裁剪
                value = -raw_value if config['negative'] else raw_value
                value = value / config['scale']
                # if config['clip_range']:
                #     value = np.clip(value, config['clip_range'][0], config['clip_range'][1])
                reward_components.append(value)
            
            return float(reward_components[0])

    def _compute_project1_std_reward(self) -> float:
        """
        计算项目1标准差感知奖励
        
        奖励公式 (对齐项目1 Agent.ipynb):
            reward = 50 - ((reg_mean + K*reg_std) + Z*(emg_mean + K*emg_std))
        
        其中:
            - reg_mean: 普通车辆平均等待时间
            - reg_std: 普通车辆等待时间标准差 ⭐核心创新
            - emg_mean: 应急车辆平均等待时间
            - emg_std: 应急车辆等待时间标准差
            - K: 标准差权重系数 (默认0.5)
            - Z: 应急车辆惩罚倍数 (默认1.0)
        
        Returns:
            reward: float, 奖励值
                - 范围约为 [-150, 50]
                - 正值表示交通状态良好
                - 负值表示拥堵严重
        """
        config = self.REWARD_CONFIGS['project1_std_reward']
        K = config['K']
        Z = config['Z']
        base_reward = config['base_reward']
        ambulance_type_ids = config['ambulance_type_ids']
        
        # ========== 1. 收集所有观测车道的等待时间分布 ==========
        regular_waiting_times = []  # 普通车辆
        emergency_waiting_times = []  # 应急车辆
        
        eng = self.world.eng
        
        # 遍历所有观测车道
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    # 获取车道上的所有车辆
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    
                    for veh_id in vehicle_ids:
                        try:
                            # 获取等待时间（累计等待时间）
                            waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id)
                            
                            # 判断车辆类型
                            veh_type = eng.vehicle.getTypeID(veh_id)
                            
                            if veh_type in ambulance_type_ids:
                                # 应急车辆
                                emergency_waiting_times.append(waiting_time)
                            else:
                                # 普通车辆
                                regular_waiting_times.append(waiting_time)
                        
                        except Exception as e:
                            # 单个车辆信息获取失败，跳过
                            continue
                
                except Exception as e:
                    # 车道信息获取失败，跳过
                    continue
        
        # ========== 2. 计算普通车辆的统计量 ==========
        if len(regular_waiting_times) > 0:
            reg_mean = float(np.mean(regular_waiting_times))
            reg_std = float(np.std(regular_waiting_times))
        else:
            # 没有普通车辆，设为0（理想状态）
            reg_mean = 0.0
            reg_std = 0.0
        
        # ========== 3. 计算应急车辆的统计量 ==========
        if len(emergency_waiting_times) > 0:
            emg_mean = float(np.mean(emergency_waiting_times))
            emg_std = float(np.std(emergency_waiting_times))
        else:
            # 没有应急车辆，设为0
            emg_mean = 0.0
            emg_std = 0.0
        
        # ========== 4. 计算奖励（完全对齐项目1公式）==========
        reward = base_reward - (
            (reg_mean + K * reg_std) + 
            Z * (emg_mean + K * emg_std)
        )
        
        # ========== 5. 可选：记录调试信息 ==========
        # 如果需要调试，可以取消注释以下代码
        # if hasattr(self.world, '_debug_reward_stats'):
        #     self.world._debug_reward_stats = {
        #         'reg_mean': reg_mean,
        #         'reg_std': reg_std,
        #         'emg_mean': emg_mean,
        #         'emg_std': emg_std,
        #         'reward': reward,
        #         'num_regular': len(regular_waiting_times),
        #         'num_emergency': len(emergency_waiting_times)
        #     }
        
        return float(reward)


    def _compute_project1_std_reward_with_configurable_params(self, K=None, Z=None) -> float:
        """
        项目1标准差奖励 - 支持外部传入K和Z参数（用于参数扫描实验）
        
        Args:
            K: 标准差权重系数（如果为None，使用配置中的默认值）
            Z: 应急车辆惩罚倍数（如果为None，使用配置中的默认值）
        
        Returns:
            reward: float
        """
        config = self.REWARD_CONFIGS['project1_std_reward']
        
        # 使用传入参数或默认值
        K = K if K is not None else config['K']
        Z = Z if Z is not None else config['Z']
        
        base_reward = config['base_reward']
        ambulance_type_ids = config['ambulance_type_ids']
        
        # 收集等待时间
        regular_waiting_times = []
        emergency_waiting_times = []
        
        eng = self.world.eng
        
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    
                    for veh_id in vehicle_ids:
                        try:
                            waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id)
                            veh_type = eng.vehicle.getTypeID(veh_id)
                            
                            if veh_type in ambulance_type_ids:
                                emergency_waiting_times.append(waiting_time)
                            else:
                                regular_waiting_times.append(waiting_time)
                        except:
                            continue
                except:
                    continue
        
        # 计算统计量
        reg_mean = float(np.mean(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0
        reg_std = float(np.std(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0
        emg_mean = float(np.mean(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0
        emg_std = float(np.std(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0
        
        # 计算奖励
        reward = base_reward - (
            (reg_mean + K * reg_std) + 
            Z * (emg_mean + K * emg_std)
        )
        
        return float(reward)
    
    def get_reward_statistics(self) -> dict:
        """
        获取当前时刻的奖励统计信息（用于分析）
        
        Returns:
            stats: dict
                {
                    'regular_vehicles': {
                        'count': int,
                        'mean_waiting': float,
                        'std_waiting': float,
                        'max_waiting': float,
                        'min_waiting': float
                    },
                    'emergency_vehicles': {
                        'count': int,
                        'mean_waiting': float,
                        'std_waiting': float,
                        'max_waiting': float,
                        'min_waiting': float
                    },
                    'reward_components': {
                        'base': float,
                        'regular_penalty': float,
                        'emergency_penalty': float,
                        'total_reward': float
                    }
                }
        """
        config = self.REWARD_CONFIGS.get('project1_std_reward', {})
        K = config.get('K', 0.5)
        Z = config.get('Z', 1.0)
        base_reward = config.get('base_reward', 50.0)
        ambulance_type_ids = config.get('ambulance_type_ids', ['ambulance_type', 'emergency'])
        
        # 收集等待时间
        regular_waiting_times = []
        emergency_waiting_times = []
        
        eng = self.world.eng
        
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    
                    for veh_id in vehicle_ids:
                        try:
                            waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id)
                            veh_type = eng.vehicle.getTypeID(veh_id)
                            
                            if veh_type in ambulance_type_ids:
                                emergency_waiting_times.append(waiting_time)
                            else:
                                regular_waiting_times.append(waiting_time)
                        except:
                            continue
                except:
                    continue
        
        # 计算统计信息
        stats = {
            'regular_vehicles': {
                'count': len(regular_waiting_times),
                'mean_waiting': float(np.mean(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0,
                'std_waiting': float(np.std(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0,
                'max_waiting': float(np.max(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0,
                'min_waiting': float(np.min(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0,
            },
            'emergency_vehicles': {
                'count': len(emergency_waiting_times),
                'mean_waiting': float(np.mean(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
                'std_waiting': float(np.std(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
                'max_waiting': float(np.max(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
                'min_waiting': float(np.min(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
            }
        }
        
        # 计算奖励组件
        reg_mean = stats['regular_vehicles']['mean_waiting']
        reg_std = stats['regular_vehicles']['std_waiting']
        emg_mean = stats['emergency_vehicles']['mean_waiting']
        emg_std = stats['emergency_vehicles']['std_waiting']
        
        regular_penalty = reg_mean + K * reg_std
        emergency_penalty = Z * (emg_mean + K * emg_std)
        total_reward = base_reward - regular_penalty - emergency_penalty
        
        stats['reward_components'] = {
            'base': base_reward,
            'regular_penalty': regular_penalty,
            'emergency_penalty': emergency_penalty,
            'total_reward': total_reward,
            'K': K,
            'Z': Z
        }
        
        return stats








    
    # def compute_reward(self) -> np.ndarray:
    #     subscribed_results = [self.world.info_dynamics_real_time[fn] for fn in self.fns_subscribed]
    #     reward_array = np.array([])
    #     for i in range(len(self.fns_subscribed)):
    #         # result can be a dict or list, represents certain information cover all lanes
    #         result = subscribed_results[i] # result can be a dictionary{lane_id/inter_id: value}, fetch an information data

    #         #===================pressure information====================
    #         # pressure returns result of each intersections, so return directly
    #         if self.ts.id in result:
    #             reward_array = np.append(reward_array, result[self.ts.id])
    #             continue
    #         #====================================================
    #         fn_result = np.array([])

    #         for road_lanes in self.lanes_road_observed:
    #             road_result = []
    #             for lane_id in road_lanes:
    #                 road_result.append(result[lane_id])
    #             # if self.average == "road" or self.average == "all":
    #             road_result = np.mean(road_result)
    #             fn_result = np.append(fn_result, road_result)
    #         #====================================================
    #         # if self.average == "all":
    #         fn_result = np.mean(fn_result)
    #         reward_array = np.append(reward_array, fn_result)
        
    #     if self.negative:
    #         reward_array = reward_array * (-1)
        
    #     # ✅ 计算差值奖励
    #     if self.last_reward is not None:
    #         # 差值：上次 - 当前（改善为正）
    #         reward = self.last_reward - reward_array
    #     else:
    #         # 第一次调用
    #         reward = np.zeros_like(reward_array)
    #      # 更新历史值
    #     self.last_reward = reward_array.copy()
    #     # ✅ 添加奖励归一化/缩放
    #     # 方案1: 除以一个常数缩放（推荐）
    #     # reward_array = reward_array / 1000.0  # 根据你的实际奖励范围调整
        
    #     # 方案2: Clip到合理范围
    #     # reward_array = np.clip(reward_array, -10.0, 10.0)
        
    #     # 方案3: 使用tanh压缩（如果奖励范围很大）
    #     # reward_array = np.tanh(reward_array / 1000.0) * 10.0
    #     return reward