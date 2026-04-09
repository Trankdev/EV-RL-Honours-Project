"""Observation functions for traffic signals."""

import numpy as np

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts, world):
        """Initialize observation function."""
        self.ts = ts
        self.world = world

class Observation(ObservationFunction):
    """Default observation function for traffic signals."""
    def __init__(self, ts, world, obs_to_subscribe, in_only=True):
        super().__init__(ts, world)

        self.obs_to_subscribe = obs_to_subscribe
        self.algorithm_name = getattr(world, '_algorithm_name', '')
        self.normalize = getattr(world, '_normalize_observation', False)

        if not hasattr(world, 'traffic_light_info'):
            raise RuntimeError(
                "traffic_light_info not found in World object. "
                "Static preprocessing must be completed in parse_sumo_config.__init__()"
            )

        if ts.id not in world.traffic_light_info:
            raise ValueError(
                f"Traffic light '{ts.id}' not found in traffic_light_info. "
                f"Available IDs: {list(world.traffic_light_info.keys())}"
            )

        tl_info = world.traffic_light_info[ts.id]
        if in_only:
            self.lanes_road_observed = tl_info['lanes_road_observed_in_only']
        else:
            self.lanes_road_observed = tl_info['lanes_road_observed']

        self._vehicle_size_min_gap = 7.5  # 车身长度 + 最小间距
        self._flat_lanes_observed = [lane_id
                                     for road_lanes in self.lanes_road_observed
                                     for lane_id in road_lanes]
        self._lane_capacities = None
        self._build_lane_capacity_cache()

    def _build_lane_capacity_cache(self):
        """Cache lane capacities aligned with self._flat_lanes_observed."""
        try:
            eng = getattr(self.world, 'eng', None)
            if eng is None:
                raise RuntimeError(
                    f"world.eng is None when building lane capacity cache (ts={self.ts.id}). "
                    "SUMO should be started (libsumo/traci) before creating Observation."
                )

            lane_lengths = np.array(
                [float(eng.lane.getLength(lane_id)) for lane_id in self._flat_lanes_observed],
                dtype=np.float32
            )

        except Exception as e:
            interface = getattr(self.world, 'interface_flag', None)
            raise RuntimeError(
                f"Failed to build lane capacity cache for ts={self.ts.id}. "
                f"interface_flag={interface}. "
                f"Example lane_id={self._flat_lanes_observed[0] if self._flat_lanes_observed else 'N/A'}. "
                f"Total lanes={len(self._flat_lanes_observed)}. "
                f"Original error: {e}"
            ) from e

        capacities = lane_lengths / np.float32(self._vehicle_size_min_gap)
        capacities = np.maximum(capacities, np.float32(1e-6))
        self._lane_capacities = capacities

    def compute_observation(self) -> np.ndarray:
        observation_array = np.array([], dtype=np.float32)

        # project1 / std_dqn 模式
        if 'project1' in self.algorithm_name.lower() or 'std_dqn' in self.algorithm_name.lower():
            return self._compute_project1_std_observation()

        # 默认模式（MAPPO 等）：phase + min_green + 订阅特征
        num_phases = len(self.ts.green_phases)
        phase_id = [1.0 if self.ts.green_phase == i else 0.0 for i in range(num_phases)]
        observation_array = np.append(observation_array, phase_id)

        # 0 = 不能切换（未满足 min_green），1 = 可以切换
        min_green_satisfied = [
            0.0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_phase_time
            else 1.0
        ]
        observation_array = np.append(observation_array, min_green_satisfied)

        subscribed_results = [self.world.info_dynamics_real_time[fn] for fn in self.obs_to_subscribe]

        for i in range(len(self.obs_to_subscribe)):
            fn_name = self.obs_to_subscribe[i]
            result = subscribed_results[i]
            fn_result = np.array([], dtype=np.float32)

            for road_lanes in self.lanes_road_observed:
                road_result = []
                for lane_id in road_lanes:
                    road_result.append(result[lane_id])
                road_result = np.array(road_result, dtype=np.float32)
                fn_result = np.append(fn_result, road_result)

            if self.normalize:
                fn_result = self._normalize_features(fn_name, fn_result)

            observation_array = np.append(observation_array, fn_result)

        return observation_array

    def _normalize_features(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        """根据特征类型对值进行归一化。"""
        # 优先使用 YAML 中配置的归一化参数
        norm_params = getattr(self.world, '_norm_params', {})

        if feature_name in norm_params:
            params = norm_params[feature_name]
            norm_value = params.get('norm', 1.0)
            clip_value = params.get('clip', None)
            normalized = values / norm_value
            if clip_value is not None:
                normalized = np.clip(normalized, 0.0, clip_value)
            return normalized.astype(np.float32)

        # 基于车道容量归一化
        if feature_name in ('lane_waiting_count', 'lane_count'):
            if (self._lane_capacities is None) or (len(self._lane_capacities) != len(values)):
                self._build_lane_capacity_cache()
            return np.clip(values / self._lane_capacities, 0.0, 1.0).astype(np.float32)

        if feature_name == 'lane_waiting_time_count':
            return np.clip(values / 300.0, 0.0, 1.0).astype(np.float32)

        elif feature_name == 'lane_delay':
            return np.clip(values / 1.0, 0.0, 1.0).astype(np.float32)

        elif feature_name == 'lane_queue_length':
            return np.clip(values / 200.0, 0.0, 1.0).astype(np.float32)

        elif feature_name == 'lane_speed':
            return np.clip(values / 15.0, 0.0, 1.0).astype(np.float32)

        elif feature_name == 'lane_occupancy':
            return values.astype(np.float32)

        elif feature_name == 'phase':
            return values.astype(np.float32)

        elif feature_name == 'current_phase_duration':
            return np.clip(values / 120.0, 0.0, 1.0).astype(np.float32)

        else:
            print(f"Warning: No normalization method defined for '{feature_name}', using default max=50.0")
            return np.clip(values / 50.0, 0.0, 1.0).astype(np.float32)

    def _compute_project1_std_observation(self) -> np.ndarray:
        """
        项目1风格观测：包含等待时间标准差。

        Returns:
            observation: numpy array, shape (num_phases + num_lanes * 5,)
                = [phase_onehot(N_phases), lane_features(N_lanes × 5)]

            车道数由路网自动决定（不硬编码为 12）。
            每个车道 5 个特征:
            1. 车辆数 / 17
            2. 普通车辆平均等待时间 / 100
            3. 普通车辆等待时间标准差 / 100  ⭐
            4. 应急车辆最大等待时间 / 100
            5. 出口道路拥堵状态 (0=堵塞, 1=畅通)
        """
        obs = []

        # 1. 相位 one-hot 编码
        num_phases = len(self.ts.green_phases)
        phase_onehot = [1.0 if self.ts.green_phase == i else 0.0
                        for i in range(num_phases)]
        obs.extend(phase_onehot)

        # 2. 获取入口车道列表（自动适应路网，不填充/截断）
        flat_lanes = [lane_id for road_lanes in self.lanes_road_observed
                      for lane_id in road_lanes]

        # 3. 计算每个车道的5个特征
        for lane_idx, lane_id in enumerate(flat_lanes):
            if lane_id == 'dummy':
                obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
                continue

            try:
                vehicles_info = self._get_lane_vehicles_detailed(lane_id)
            except Exception as e:
                print(f"Warning: 获取车道{lane_id}信息失败: {e}")
                obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
                continue

            regular_vehicles = [v for v in vehicles_info if not v['is_emergency']]
            emergency_vehicles = [v for v in vehicles_info if v['is_emergency']]

            # 特征1: 车辆数 / 17
            feature1 = min(len(vehicles_info) / 17.0, 1.0)

            # 特征2: 平均等待时间 / 100（仅普通车辆）
            if len(regular_vehicles) > 0:
                waiting_times = [v['waiting_time'] for v in regular_vehicles]
                feature2 = min(np.mean(waiting_times) / 100.0, 1.0)
            else:
                waiting_times = []
                feature2 = 0.0

            # 特征3: 等待时间标准差 / 100
            if len(waiting_times) > 1:
                feature3 = min(np.std(waiting_times) / 100.0, 1.0)
            else:
                feature3 = 0.0

            # 特征4: 应急车辆最大等待时间 / 100
            if len(emergency_vehicles) > 0:
                emg_waiting_times = [v['waiting_time'] for v in emergency_vehicles]
                feature4 = min(max(emg_waiting_times) / 100.0, 1.0)
            else:
                feature4 = 0.0

            # 特征5: 出口道路拥堵状态
            feature5 = self._compute_outgoing_attention_for_lane(lane_idx)

            obs.extend([feature1, feature2, feature3, feature4, feature5])

        return np.array(obs, dtype=np.float32)

    def _get_lane_vehicles_detailed(self, lane_id: str) -> list:
        """
        获取车道上所有车辆的详细信息。

        Returns:
            list of dict: [{'id', 'waiting_time', 'is_emergency', 'position'}, ...]
        """
        vehicles_info = []

        try:
            eng = self.world.eng
            vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)

            for veh_id in vehicle_ids:
                try:
                    waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id)
                    veh_type = eng.vehicle.getTypeID(veh_id)
                    is_emergency = (veh_type in ['ambulance_type', 'emergency'])
                    position = eng.vehicle.getLanePosition(veh_id)

                    vehicles_info.append({
                        'id': veh_id,
                        'waiting_time': waiting_time,
                        'is_emergency': is_emergency,
                        'position': position
                    })
                except Exception:
                    continue

        except Exception:
            pass

        return vehicles_info

    def _compute_outgoing_attention_for_lane(self, lane_idx: int) -> float:
        """
        计算出口道路拥堵状态。

        每3个入口车道对应一个出口道路：
            lane_idx 0-2 → 出口0，lane_idx 3-5 → 出口1，...

        Returns:
            float [0, 1]：1.0 = 畅通，0.0 = 完全拥堵
        """
        outgoing_road_idx = lane_idx // 3

        try:
            if outgoing_road_idx >= len(self.ts.out_roads):
                return 1.0

            out_road = self.ts.out_roads[outgoing_road_idx]
            out_lanes = self.ts.road_lane_mapping.get(out_road, [])

            if not out_lanes:
                return 1.0

            eng = self.world.eng
            total_vehicles = 0

            for out_lane_id in out_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(out_lane_id)
                    lane_length = eng.lane.getLength(out_lane_id)

                    for veh_id in vehicle_ids:
                        position = eng.vehicle.getLanePosition(veh_id)
                        distance_to_end = lane_length - position
                        # 只统计靠近出口路口（终点100米内）的车辆
                        if distance_to_end <= 100:
                            total_vehicles += 1
                except Exception:
                    continue

            # attention值：1 - (车辆数 / 容量)，越拥堵值越小
            capacity = 17 * len(out_lanes)
            congestion = 1.0 - min(total_vehicles / max(capacity, 1), 1.0)
            return float(congestion)

        except Exception:
            return 1.0
