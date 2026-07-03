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

        self._vehicle_size_min_gap = 7.5 # Vehicle length + minimum gap
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

        # Project1 / std_DQN mode
        if 'project1' in self.algorithm_name.lower() or 'std_dqn' in self.algorithm_name.lower():
            return self._compute_project1_std_observation()
        
        # fyp mode - LANE FEATURES MODE
        if 'final_year_project_lane_mode' in self.algorithm_name.lower() or 'fyp_lane' in self.algorithm_name.lower():
            return self._compute_fyp_observation_lane_version() 
        
        # fyp mode
        if 'final_year_project' in self.algorithm_name.lower() or 'fyp' in self.algorithm_name.lower():
            return self._compute_fyp_observation() 
        # Default mode (e.g., MAPPO): phase + min_green + subscribed features
        num_phases = len(self.ts.green_phases)
        phase_id = [1.0 if self.ts.green_phase == i else 0.0 for i in range(num_phases)]
        observation_array = np.append(observation_array, phase_id)

        # 0 = cannot switch (min green not satisfied), 1 = can switch
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
        """Normalize feature values based on type."""
        # Prefer normalization parameters defined in YAML
        norm_params = getattr(self.world, '_norm_params', {})

        if feature_name in norm_params:
            params = norm_params[feature_name]
            norm_value = params.get('norm', 1.0)
            clip_value = params.get('clip', None)
            normalized = values / norm_value
            if clip_value is not None:
                normalized = np.clip(normalized, 0.0, clip_value)
            return normalized.astype(np.float32)

        # Capacity-based normalization
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


###############################################################################
##################### OLD OBSERVATION SPACE (Baseline) ########################
###############################################################################

    def _compute_project1_std_observation(self) -> np.ndarray:
        """
        Project1-style observation including waiting time standard deviation.

        Returns:
            observation: numpy array
                = [phase_onehot(N_phases), lane_features(N_lanes × 5)]

        Each lane has 5 features:
        1. vehicle count / 17
        2. regular vehicle mean waiting time / 100
        3. regular vehicle waiting time std / 100
        4. emergency vehicle max waiting time / 100
        5. outgoing congestion state (0 = blocked, 1 = free)
        """
        obs = []

        # 1. Phase one-hot encoding
        num_phases = len(self.ts.green_phases)
        phase_onehot = [1.0 if self.ts.green_phase == i else 0.0
                        for i in range(num_phases)]
        obs.extend(phase_onehot)

        # 2. Flatten lane list (auto adapts to network size)
        flat_lanes = [lane_id for road_lanes in self.lanes_road_observed
                      for lane_id in road_lanes]

        # 3. Compute lane features
        for lane_idx, lane_id in enumerate(flat_lanes):
            if lane_id == 'dummy':
                obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
                continue

            try:
                vehicles_info = self._get_lane_vehicles_detailed(lane_id)
            except Exception as e:
                print(f"Warning: Failed to retrieve information for lane {lane_id}: {e}")
                obs.extend([0.0, 0.0, 0.0, 0.0, 1.0])
                continue

            regular_vehicles = [v for v in vehicles_info if not v['is_emergency']]
            emergency_vehicles = [v for v in vehicles_info if v['is_emergency']]

            # Feature 1: vehicle count / 17
            feature1 = min(len(vehicles_info) / 17.0, 1.0)

            # Feature 2: mean waiting time / 100 (regular only)
            if len(regular_vehicles) > 0:
                waiting_times = [v['waiting_time'] for v in regular_vehicles]
                feature2 = min(np.mean(waiting_times) / 100.0, 1.0)
            else:
                waiting_times = []
                feature2 = 0.0

            # Feature 3: waiting time std / 100
            if len(waiting_times) > 1:
                feature3 = min(np.std(waiting_times) / 100.0, 1.0)
            else:
                feature3 = 0.0

            # Feature 4: emergency max waiting time / 100
            if len(emergency_vehicles) > 0:
                emg_waiting_times = [v['waiting_time'] for v in emergency_vehicles]
                feature4 = min(max(emg_waiting_times) / 100.0, 1.0)
            else:
                feature4 = 0.0

            # Feature 5: outgoing road congestion state
            feature5 = self._compute_outgoing_attention_for_lane(lane_idx)

            obs.extend([feature1, feature2, feature3, feature4, feature5])

        return np.array(obs, dtype=np.float32)
    
    ###########################################################################
    ################## NEW OBSERVATION SPACE FOR FYP PROJECT ##################
    ###########################################################################
    
    # This is the LANE FEATURE version - where new additions are done as lane features
    # TODO: need to edit line 564 in env.py whenever switching to/from this function - ob_length    = num_phases + num_in_lanes * 6 # TODO: will need to adjust to match observation space length
    def _compute_fyp_observation_lane_version(self) -> np.ndarray:
        """
        FYP-style observation space (per-lane EV features)
    
        Returns:
            observation: numpy array
                = [phase_onehot(N_phases),
                   lane_features(N_lanes × K)]
    
        Each lane has K features (K depends on EV_URGENCY_MODE):
            1. vehicle count / 17  OR  lane occupancy (toggle: FEATURE1_MODE)
            2. regular vehicle mean waiting time / 100
            3. regular vehicle waiting time std / 100
            4. EV delay metric (toggle: EV_DELAY_METRIC_MODE -> waiting_time or delay_ratio)
            5. EV urgency (toggle: EV_URGENCY_MODE -> tta(1) / speed(1) / distance(1) / speed_distance(2) / none(0))
            6. outgoing congestion state (0 = blocked, 1 = free)
    
        K = 4 + urgency_dims + 1 = 5 (none), 6 (tta/speed/distance), or 7 (speed_distance)
        """
        obs = []
        eng = self.world.eng
    
        # 1. Phase one-hot encoding (intersection-level, unavoidable)
        num_phases = len(self.ts.green_phases)
        phase_onehot = [1.0 if self.ts.green_phase == i else 0.0
                        for i in range(num_phases)]
        obs.extend(phase_onehot)
    
        # 2. Flatten lane list
        flat_lanes = [lane_id for road_lanes in self.lanes_road_observed
                      for lane_id in road_lanes]
    
        # Toggle which lane-fullness metric feeds Feature 1.
        FEATURE1_MODE = 'fixed_17_baseline'  # options: 'fixed_17_baseline', 'lane_capacity_occupancy' # TODO
    
        # Toggle which EV congestion metric feeds the observation.
        EV_DELAY_METRIC_MODE = 'waiting_time'  # options: 'delay_ratio', 'waiting_time' # TODO
    
        # Toggle which EV urgency representation feeds the observation.
        # Obs dims per lane vary by mode: tta(1), speed(1), distance(1), speed_distance(2), none(0)
        EV_URGENCY_MODE = 'none'  # options: 'tta', 'speed', 'distance', 'speed_distance', 'none' # TODO
    
        for lane_idx, lane_id in enumerate(flat_lanes):
            if lane_id == 'dummy':
                base = [0.0, 0.0, 0.0, 0.0]
                urgency_pad = [0.0, 0.0] if EV_URGENCY_MODE == 'speed_distance' else \
                              ([0.0] if EV_URGENCY_MODE != 'none' else [])
                obs.extend(base + urgency_pad + [1.0])
                continue
    
            try:
                vehicles_info = self._get_lane_vehicles_detailed(lane_id)
            except Exception as e:
                print(f"Warning: Failed to retrieve information for lane {lane_id}: {e}")
                base = [0.0, 0.0, 0.0, 0.0]
                urgency_pad = [0.0, 0.0] if EV_URGENCY_MODE == 'speed_distance' else \
                              ([0.0] if EV_URGENCY_MODE != 'none' else [])
                obs.extend(base + urgency_pad + [1.0])
                continue
    
            regular_vehicles = [v for v in vehicles_info if not v['is_emergency']]
            emergency_vehicles = [v for v in vehicles_info if v['is_emergency']]
    
            # Feature 1: vehicle count / 17  OR  lane capacity occupancy
            if FEATURE1_MODE == 'lane_capacity_occupancy':
                feature1 = min(len(vehicles_info) / self._lane_capacities[lane_idx], 1.0)
            else:
                feature1 = min(len(vehicles_info) / 17.0, 1.0)
    
            # Feature 2: regular mean waiting time / 100
            if len(regular_vehicles) > 0:
                waiting_times = [v['waiting_time'] for v in regular_vehicles]
                feature2 = min(np.mean(waiting_times) / 100.0, 1.0)
            else:
                waiting_times = []
                feature2 = 0.0
    
            # Feature 3: waiting time std / 100
            if len(waiting_times) > 1:
                feature3 = min(np.std(waiting_times) / 100.0, 1.0)
            else:
                feature3 = 0.0
    
            # Find the closest EV in this lane (by distance to the intersection)
            ev_found_id = None
            ev_found_position = None
            ev_found_waiting_time = None
            best_distance = float('inf')
            for ev in emergency_vehicles:
                try:
                    veh_id = ev['id']
                    next_tls = eng.vehicle.getNextTLS(veh_id)
                    distance = next_tls[0][2] if next_tls else float('inf')
                    if distance < best_distance:
                        best_distance = distance
                        ev_found_id = veh_id
                        ev_found_position = ev['position']
                        ev_found_waiting_time = ev['waiting_time']
                except Exception:
                    continue
    
            ev_delay_metric = 0.0
            ev_eta = 0.0
            ev_speed_norm = 0.0
            ev_distance_norm = 0.0
    
            if ev_found_id is not None:
                # Feature 4: EV delay metric
                if EV_DELAY_METRIC_MODE == 'waiting_time':
                    try:
                        ev_delay_metric = min(ev_found_waiting_time / 100.0, 1.0)
                    except Exception:
                        ev_delay_metric = 0.0
                elif EV_DELAY_METRIC_MODE == 'delay_ratio':
                    try:
                        current_time = self.world.get_current_time()
                        if ev_found_id in self.world.ev_lane_entry_time:
                            stored_lane, entry_time = self.world.ev_lane_entry_time[ev_found_id]
                            actual_travel_time = current_time - entry_time
                        else:
                            actual_travel_time = 0.0
                        lane_max_speed = eng.lane.getMaxSpeed(lane_id)
                        if lane_max_speed > 0 and ev_found_position > 0 and actual_travel_time > 0:
                            free_flow_time = ev_found_position / lane_max_speed
                            delay_ratio = 1.0 - (free_flow_time / max(actual_travel_time, free_flow_time))
                            ev_delay_metric = max(0.0, min(delay_ratio, 1.0))
                    except Exception:
                        ev_delay_metric = 0.0
    
                # Feature 5: EV urgency
                if EV_URGENCY_MODE == 'tta':
                    try:
                        next_tls = eng.vehicle.getNextTLS(ev_found_id)
                        if next_tls:
                            distance = next_tls[0][2]
                            current_speed = eng.vehicle.getSpeed(ev_found_id)
                            if current_speed > 0.5:
                                eta = distance / current_speed
                                ev_eta = 1.0 - min(eta / 60.0, 1.0)
                            else:
                                ev_eta = 1.0
                        else:
                            ev_eta = 0.0
                    except Exception:
                        ev_eta = 0.0
    
                if EV_URGENCY_MODE in ('speed', 'speed_distance'):
                    try:
                        current_speed = eng.vehicle.getSpeed(ev_found_id)
                        lane_max_speed = eng.lane.getMaxSpeed(lane_id)
                        if lane_max_speed > 0:
                            ev_speed_norm = 1.0 - max(0.0, min(current_speed / lane_max_speed, 1.0))
                        else:
                            ev_speed_norm = 0.0
                    except Exception:
                        ev_speed_norm = 0.0
    
                if EV_URGENCY_MODE in ('distance', 'speed_distance'):
                    try:
                        lane_length = eng.lane.getLength(lane_id)
                        if lane_length > 0:
                            dist_to_stop = lane_length - ev_found_position
                            ev_distance_norm = 1.0 - max(0.0, min(dist_to_stop / lane_length, 1.0))
                        else:
                            ev_distance_norm = 0.0
                    except Exception:
                        ev_distance_norm = 0.0
    
            # Feature 6: outgoing road congestion state
            feature_congestion = self._compute_outgoing_attention_for_lane_new(lane_id)
    
            obs.extend([feature1, feature2, feature3, ev_delay_metric])
    
            if EV_URGENCY_MODE == 'tta':
                obs.append(ev_eta)
            elif EV_URGENCY_MODE == 'speed':
                obs.append(ev_speed_norm)
            elif EV_URGENCY_MODE == 'distance':
                obs.append(ev_distance_norm)
            elif EV_URGENCY_MODE == 'speed_distance':
                obs.append(ev_speed_norm)
                obs.append(ev_distance_norm)
            # 'none' -> nothing appended
    
            obs.append(feature_congestion)
    
        return np.array(obs, dtype=np.float32)

    ###########################################################################
    
    # This is the INTERSECTION LEVEL FEATURE version - where certain features are switched to become intersection level features (instead of lane level)
    # TODO: need to edit line 577 in env.py whenever switching to/from this function - ob_length    = num_phases + num_in_lanes * 6 # TODO: will need to adjust to match observation space length
    def _compute_fyp_observation(self) -> np.ndarray:
        """
        FYP-style observation space
        Returns:
            observation: numpy array
                = [phase_onehot(N_phases), 
                   EV Delay Ratio (0-1), ← intersection level
                   EV ETA urgency (0-1), ← intersection level
                   lane_features(N_lanes × 4)]
        Each lane has 4 features:
            1. vehicle count / 17
            2. regular vehicle mean waiting time / 100
            3. regular vehicle waiting time std / 100
            4. outgoing congestion state (0 = blocked, 1 = free)
        """
        obs = []
        eng = self.world.eng
    
        # 1. Phase one-hot encoding
        num_phases = len(self.ts.green_phases)
        phase_onehot = [1.0 if self.ts.green_phase == i else 0.0
                        for i in range(num_phases)]
        obs.extend(phase_onehot)
    
        # 2. Flatten lane list
        flat_lanes = [lane_id for road_lanes in self.lanes_road_observed
                      for lane_id in road_lanes]
    
        # 3. Intersection-level EV features — scan all lanes once to find EV
        ev_delay_ratio = 0.0
        ev_eta = 0.0
        ev_found_id = None
        ev_found_lane = None
        ev_found_position = None
        ev_found_waiting_time = None

        # Toggle which EV congestion metric feeds the observation.
        # Switch this one line to flip between the two — obs length is unchanged either way.
        EV_DELAY_METRIC_MODE = 'waiting_time'  # options: 'delay_ratio', 'waiting_time' # TODO: pick if you want EV Delay Ratio or EV Waiting time used
        
        # Toggle which EV urgency representation feeds the observation.
        # Mutually exclusive by design — TTA is itself derived from speed+distance,
        # so mixing TTA with speed/distance would be redundant rather than a clean ablation.
        # Obs length varies by mode:
        #   'tta'            -> +1 dim (EV ETA/urgency)
        #   'speed'          -> +1 dim (EV speed / free-flow speed)
        #   'distance'       -> +1 dim (distance remaining to intersection / lane length)
        #   'speed_distance' -> +2 dims (both of the above)
        #   'none'           -> +0 dims
        # Remember to update ob_length in env.py to match whichever mode you pick.
        EV_URGENCY_MODE = 'speed_distance'  # options: 'tta', 'speed', 'distance', 'speed_distance', 'none' # TODO: pick what you want to use to inform how close an EV is to an intersection / measure of urgency

        ev_delay_metric = 0.0
        
        ev_eta = 0.0
        ev_speed_norm = 0.0
        ev_distance_norm = 0.0

        # Scan all lanes once to find the closest EV (by distance to this intersection)
        best_distance = float('inf')
        for lane_id in flat_lanes:
            if lane_id == 'dummy':
                continue
            try:
                vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
            except Exception:
                continue
            for veh_id in vehicle_ids:
                try:
                    veh_type = eng.vehicle.getTypeID(veh_id)
                    if veh_type not in ('ambulance_type', 'emergency'):
                        continue
                    next_tls = eng.vehicle.getNextTLS(veh_id)
                    distance = next_tls[0][2] if next_tls else float('inf')
                    if distance < best_distance:
                        best_distance = distance
                        ev_found_id = veh_id
                        ev_found_lane = lane_id
                        ev_found_position = eng.vehicle.getLanePosition(veh_id)
                        ev_found_waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id)
                except Exception:
                    continue
        
        if ev_found_id is not None:
            if EV_DELAY_METRIC_MODE == 'waiting_time':
                try:
                    ev_delay_metric = min(ev_found_waiting_time / 100.0, 1.0)
                except Exception:
                    ev_delay_metric = 0.0
            elif EV_DELAY_METRIC_MODE == 'delay_ratio':
                try:
                    current_time = self.world.get_current_time()
                    if ev_found_id in self.world.ev_lane_entry_time:
                        stored_lane, entry_time = self.world.ev_lane_entry_time[ev_found_id]
                        actual_travel_time = current_time - entry_time
                    else:
                        actual_travel_time = 0.0
                    lane_max_speed = eng.lane.getMaxSpeed(ev_found_lane)
                    if lane_max_speed > 0 and ev_found_position > 0 and actual_travel_time > 0:
                        free_flow_time = ev_found_position / lane_max_speed
                        ev_delay_ratio = 1.0 - (free_flow_time / max(actual_travel_time, free_flow_time))
                        ev_delay_metric = max(0.0, min(ev_delay_ratio, 1.0))
                except Exception:
                    ev_delay_metric = 0.0
            
            if EV_URGENCY_MODE == 'tta':
                try:
                    next_tls = eng.vehicle.getNextTLS(ev_found_id)
                    if next_tls:
                        distance = next_tls[0][2]
                        current_speed = eng.vehicle.getSpeed(ev_found_id)
                        if current_speed > 0.5:
                            eta = distance / current_speed
                            ev_eta = 1.0 - min(eta / 60.0, 1.0)
                        else:
                            ev_eta = 1.0
                    else:
                        ev_eta = 0.0
                except Exception:
                    ev_eta = 0.0

            if EV_URGENCY_MODE in ('speed', 'speed_distance'):
                try:
                    current_speed = eng.vehicle.getSpeed(ev_found_id)
                    lane_max_speed = eng.lane.getMaxSpeed(ev_found_lane)
                    if lane_max_speed > 0:
                        # 1 = stopped/crawling (urgent), 0 = free-flowing (fine) — matches "no EV" default
                        ev_speed_norm = 1.0 - max(0.0, min(current_speed / lane_max_speed, 1.0))
                    else:
                        ev_speed_norm = 0.0
                except Exception:
                    ev_speed_norm = 0.0

            if EV_URGENCY_MODE in ('distance', 'speed_distance'):
                try:
                    lane_length = eng.lane.getLength(ev_found_lane)
                    if lane_length > 0:
                        dist_to_stop = lane_length - ev_found_position
                        # 1 = at the stop line (urgent), 0 = just entered / far away (fine) — matches "no EV" default
                        ev_distance_norm = 1.0 - max(0.0, min(dist_to_stop / lane_length, 1.0))
                    else:
                        ev_distance_norm = 0.0
                 
                except Exception:
                    ev_distance_norm = 0.0
        
        obs.append(ev_delay_metric)
        
        if EV_URGENCY_MODE == 'tta':
            obs.append(ev_eta)
        elif EV_URGENCY_MODE == 'speed':
            obs.append(ev_speed_norm)
        elif EV_URGENCY_MODE == 'distance':
            obs.append(ev_distance_norm)
        elif EV_URGENCY_MODE == 'speed_distance':
            obs.append(ev_speed_norm)
            obs.append(ev_distance_norm)
        # 'none' -> nothing appended
    
        # 4. Per-lane features (4 each)
        for lane_idx, lane_id in enumerate(flat_lanes):
            if lane_id == 'dummy':
                obs.extend([0.0, 0.0, 0.0, 1.0])
                continue
    
            try:
                vehicles_info = self._get_lane_vehicles_detailed(lane_id)
            except Exception as e:
                print(f"Warning: Failed to retrieve information for lane {lane_id}: {e}")
                obs.extend([0.0, 0.0, 0.0, 1.0])
                continue
    
            regular_vehicles = [v for v in vehicles_info if not v['is_emergency']]
    
            # Toggle which lane-fullness metric feeds Feature 1.
            # Switch this one line to flip between the two — obs length is unchanged either way.
            FEATURE1_MODE = 'fixed_17_baseline'  # options: 'fixed_17_baseline', 'lane_capacity_occupancy' # TODO: pick if you want the fixed /17 count or per-lane capacity ('lane_capacity_occupancy') occupancy used

            if FEATURE1_MODE == 'lane_capacity_occupancy':
                # Lane Occupancy - how full a lane is from [0, 1] - vehicle count / lane capacity (capacity = lane_length / vehicle_size_min_gap)
                feature1 = min(len(vehicles_info) / self._lane_capacities[lane_idx], 1.0)
            else:
                # Fixed vehicle count / 17
                feature1 = min(len(vehicles_info) / 17.0, 1.0)
    
            # Feature 2: regular mean waiting time / 100
            if len(regular_vehicles) > 0:
                waiting_times = [v['waiting_time'] for v in regular_vehicles]
                feature2 = min(np.mean(waiting_times) / 100.0, 1.0)
            else:
                waiting_times = []
                feature2 = 0.0
    
            # Feature 3: waiting time std / 100
            if len(waiting_times) > 1:
                feature3 = min(np.std(waiting_times) / 100.0, 1.0)
            else:
                feature3 = 0.0
    
            # Feature 4: outgoing road congestion state
            feature4 = self._compute_outgoing_attention_for_lane_new(lane_id)
    
            obs.extend([feature1, feature2, feature3, feature4])
            
        return np.array(obs, dtype=np.float32)
    
    ###########################################################################

    def _get_lane_vehicles_detailed(self, lane_id: str) -> list:
        """
        Get detailed vehicle information on a lane.

        Returns:
            list of dicts:
            [{'id', 'waiting_time', 'is_emergency', 'position'}, ...]
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

# Fixed version is below called "_compute_outgoing_attention_for_lane_new" - retained old version for baselines?
# - or incase new version doesn't work properly

    def _compute_outgoing_attention_for_lane(self, lane_idx: int) -> float:
        """
        Compute outgoing road congestion state.

        Each 3 lanes correspond to one outgoing road:
            lanes 0–2 → road 0, lanes 3–5 → road 1, etc.

        Returns:
            float in [0, 1]: 1.0 = free flow, 0.0 = fully congested
        """
        outgoing_road_idx = lane_idx // 3 # TODO: fix this / 3 thing prolly won't work for all networks

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
                        # Only count vehicles within 100m of exit
                        if distance_to_end <= 100:
                            total_vehicles += 1
                except Exception:
                    continue

            # Attention value: 1 - (number of vehicles / capacity)
            capacity = 17 * len(out_lanes)
            congestion = 1.0 - min(total_vehicles / max(capacity, 1), 1.0)
            return float(congestion)

        except Exception:
            return 1.0
        
    # TODO: maybe make it so that this is 1 - feature so it is 1 = blocked and 0 = free (instead of the opposite, which it is rn)
    def _compute_outgoing_attention_for_lane_new(self, lane_id: str) -> float:
        """
        Compute outgoing congestion for a specific incoming lane using actual
        lane connections (left turn → left outgoing, through → through, etc.)
        Returns:
            float in [0, 1]: 1.0 = free flow, 0.0 = fully congested
        """
        try:
            eng = self.world.eng
            # getLinks returns list of (outgoing_lane_id, ...) tuples
            links = eng.lane.getLinks(lane_id)
            if not links:
                print("error? in _compute_outgoing_attention_for_lane_new")
                return 1.0
            occupancies = []
            for link in links:
                out_lane_id = link[0]  # first element is the connected lane ID
                try:
                    occ = eng.lane.getLastStepOccupancy(out_lane_id) / 100.0
                    occupancies.append(occ)
                except Exception:
                    continue
            if not occupancies:
                print("error? in _compute_outgoing_attention_for_lane_new")
                return 1.0
            avg_occupancy = sum(occupancies) / len(occupancies)
            return float(1.0 - avg_occupancy)
        except Exception:
            print("error? in _compute_outgoing_attention_for_lane_new")
            return 1.0
