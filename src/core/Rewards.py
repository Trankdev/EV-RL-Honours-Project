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
    # ================= Reward configuration definitions =================
    REWARD_CONFIGS = {
        'lane_waiting_count': {
            'description': 'Number of waiting vehicles per lane',
            'level': 'lane',  # ✅ added level
            'use_difference': False,
            'negative': True,
            'scale': 1.0,
            'clip_range': None
        },
        'lane_waiting_time_count': {
            'description': 'Total waiting time per lane',
            'level': 'lane',  # ✅ added
            'use_difference': False,
            'negative': True,
            'scale': 1.0,
            'clip_range': None
        },
        'lane_vehicle_count': {
            'description': 'Total number of vehicles per lane',
            'use_difference': False,
            'negative': False,
            'scale': 1.0,
            'clip_range': None
        },
        'avg_waiting_time_per_vehicle': {
            'description': 'Average waiting time per vehicle',
            'use_difference': False,
            'negative': True,
            'scale': 1.0,
            'clip_range': (-10.0, 0.0)
        },
        # ================= Pressure (PressLight) =================
        'pressure': {
            'description': 'Priority rewards for ambulances (civilian vehicles wait + ambulance penalty)',
            'level': 'intersection',  # ✅ marked as intersection-level
            'negative': True,  # minimize pressure
            'scale': 1.0,
        },
        # ================= Emergency vehicle priority =================
        'emergency_vehicle_priority': {
            'description': 'Priority rewards for ambulances (civilian vehicles wait + ambulance penalty)',
            'level': 'intersection',
            'negative': True,
            'civilian_weight': 0.7,      # civilian vehicle weight
            'ambulance_penalty': 5000,   # ambulance stopping penalty
            'ambulance_speed_threshold': 1.0,  # speed threshold (m/s)
            'ambulance_type_id': 'ambulance_type',  # ambulance type ID
            'scale': 1.0,
            'clip_range': None
        },
        'project1_std_reward': { # OLD CODE FUNCTION
            'description': 'project1_std_reward: 50 - (reg_mean + K*reg_std + Z*(emg_mean + K*emg_std))',
            'level': 'intersection',
            'negative': False,  # reward formula already includes negative logic
            'K': 0.5,  # standard deviation weight (tunable parameter)
            'Z': 1.0,  # emergency vehicle penalty multiplier (tunable parameter)
            'base_reward': 50.0,  # base reward value
            'scale': 1.0,
            'clip_range': None,
            'ambulance_type_ids': ['ambulance_type', 'emergency']  # emergency vehicle type list
        },
        'final_year_project_reward': { # new final year project reward
            'description': 'final_year_project_reward: 50 - (((X * (reg_group1_mean + K * reg_group1_std) + Y * (reg_group1_mean + K * reg_group1_std))  + Z*(emg_mean + K*emg_std))',
            'level': 'intersection',
            'negative': False,  # reward formula already includes negative logic
            'K': 0.5,  # standard deviation weight (tunable parameter)
            'Z': 1.0,  # emergency vehicle penalty multiplier (tunable parameter)
            'Y': 1.0, # group 2 vehicle weighting (tunable parameter)
            'X': 1.0, # group 1 vehicle weighting (tunable parameter)
            'base_reward': 50.0,  # base reward value
            'scale': 1.0,
            'clip_range': None,
            'ambulance_type_ids': ['ambulance_type', 'emergency']  # emergency vehicle type list
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
        # Special handling: emergency_vehicle_priority requires low-level data subscription
        if 'emergency_vehicle_priority' in reward_to_subscribe:
            # ambulance reward requires lane_waiting_time_count
            self.world.subscribe(['lane_waiting_time_count'])
        else:
            # other reward types subscribe directly
            self.world.subscribe(reward_to_subscribe)
        # self.world.subscribe(reward_to_subscribe) # obs_to_subscribe have been obs_subscribed here
        self.fns_subscribed = reward_to_subscribe # a list
        self.negative = negative
        # Algorithm name (for special reward logic)
        self.algorithm_name = getattr(world, '_algorithm_name', 'default')
        # Store previous rewards (for difference-based reward if needed)
        self.last_rewards = {fn: None for fn in self.fns_subscribed}
        
    def _compute_emergency_priority_reward(self):
        """
        Compute emergency vehicle priority reward.

        Aligned with original formulation:
        reward = -1 × [(civilian_penalty × 0.7) + ambulance_penalty]
        """
        config = self.REWARD_CONFIGS['emergency_vehicle_priority']
        
        # 1. compute civilian waiting-time penalty
        waiting_time_result = self.world.info_dynamics_real_time.get('lane_waiting_time_count', {})
        civilian_penalty = 0
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                civilian_penalty += waiting_time_result.get(lane_id, 0)
        
        # 2. compute ambulance penalty
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
        
        # 3. combine reward (aligned with original project formula)
        civilian_weight = config['civilian_weight']
        raw_value = (civilian_penalty * civilian_weight) + ambulance_penalty
        
        # 4. take negative value and apply scaling
        reward = -raw_value / config.get('scale', 1.0)
        
        return float(reward)
    
    def compute_reward(self) -> np.ndarray:
        """Compute reward with multiple algorithm-specific modes."""
        
        # Project1 / std-DQN special case
        if 'project1' in self.algorithm_name.lower() or 'std_dqn' in self.algorithm_name.lower():
            return self._compute_project1_std_reward()
        
        # final year project case - mirrors above logic
        if 'final_year_project' in self.algorithm_name.lower() or 'fyp' in self.algorithm_name.lower():
            return self._compute_project1_std_reward()
        
        print("The reward function is likely not being computed correctly, please use 'project1_std_dqn' for baseline or 'final_year_project' for an altered fyp version of reward")
        print("If confused, ctrl + f and find this print statement in Rewards.py")
        
        # Emergency priority special case
        if 'emergency_vehicle_priority' in self.fns_subscribed:
            return self._compute_emergency_priority_reward()
        
        subscribed_results = [self.world.info_dynamics_real_time[fn] for fn in self.fns_subscribed]
        
        # ========== MA2C special handling ==========
        if 'ma2c' in self.algorithm_name.lower():
            # ✅ Get MA2C-specific parameters from world
            reward_weights = getattr(self.world, '_reward_weights', [1.0, 0.2])
            reward_scale = getattr(self.world, '_reward_scale', 2000.0)
            reward_clip_range = getattr(self.world, '_reward_clip_range', [-2.0, 2.0])
            
            # compute each reward component
            reward_components = []
            for i, fn_name in enumerate(self.fns_subscribed):
                result = subscribed_results[i]
                
                # aggregate lane-level data (MA2C uses sum)
                fn_result = []
                for road_lanes in self.lanes_road_observed:
                    road_result = [result[lane_id] for lane_id in road_lanes]
                    fn_result.extend(road_result)
                raw_value = np.sum(fn_result)
                
                # take negative value (minimization objective)
                reward_components.append(-raw_value)
            
            # ✅ weighted combination: reward = -queue - 0.2 * wait
            weighted_reward = sum(w * r for w, r in zip(reward_weights, reward_components))
            
            # ✅ normalization: reward /= 2000.0
            normalized_reward = weighted_reward / reward_scale
            
            # ✅ clipping: clip(reward, -2, 2)
            if reward_clip_range:
                clipped_reward = np.clip(normalized_reward, 
                                        reward_clip_range[0], 
                                        reward_clip_range[1])
            else:
                clipped_reward = normalized_reward
            
            return float(clipped_reward)
        
        # ================= Default mode =================
        else:
            reward_components = []
            
            for i, fn_name in enumerate(self.fns_subscribed):
                result = subscribed_results[i]
                config = self.REWARD_CONFIGS[fn_name]
                
                if fn_name == 'pressure':
                    raw_value = result[self.ts.id]
                elif fn_name == 'lane_waiting_count':
                    # ===================aggregate lane-level data====================
                    # lane_waiting_count and lane_waiting_time_count are both lane-level data
                    fn_result = []
                    for road_lanes in self.lanes_road_observed:
                        road_result = [result[lane_id] for lane_id in road_lanes]
                        fn_result.append(np.sum(road_result))
                    
                    # average of all road segments
                    # raw_value = np.mean(fn_result)
                    # ✅ changed to sum
                    raw_value = np.sum(fn_result)
                # ✅ new: emergency vehicle priority reward
                elif fn_name == 'emergency_vehicle_priority':
                    # 1. compute civilian vehicle waiting time (from lane_waiting_time_count)
                    waiting_time_result = self.world.info_dynamics_real_time.get('lane_waiting_time_count', {})
                    civilian_penalty = 0
                    for road_lanes in self.lanes_road_observed:
                        for lane_id in road_lanes:
                            civilian_penalty += waiting_time_result.get(lane_id, 0)
                    
                    # 2. compute ambulance penalty (check all vehicles)
                    ambulance_penalty = 0
                    # get all vehicles in current simulation
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
                                    # print(f"⚠️ ambulance {veh_id} speed {speed:.2f} m/s < {speed_threshold} m/s")
                        except:
                            pass
                    # 3. combine reward
                    civilian_weight = config['civilian_weight']
                    raw_value = (civilian_penalty * civilian_weight) + ambulance_penalty
                else:
                    raise ValueError(f"Unknown reward level: {config['level']}")
                
                # process reward: negate -> scale -> clip
                value = -raw_value if config['negative'] else raw_value
                value = value / config['scale']
                # if config['clip_range']:
                #     value = np.clip(value, config['clip_range'][0], config['clip_range'][1])
                reward_components.append(value)
            
            return float(reward_components[0])

    # TODO: this one HAS to be right for which reward you want to use
    def _compute_project1_std_reward(self) -> float:
        """
        Project 1 standard deviation-aware reward.
    
        Reward formula (aligned with Project1 Agent.ipynb):
            reward = 50 - ((reg_mean + K * reg_std) + Z * (emg_mean + K * emg_std))
    
        Where:
            - reg_mean: mean waiting time of regular vehicles
            - reg_std: standard deviation of regular vehicle waiting time ⭐ core innovation
            - emg_mean: mean waiting time of emergency vehicles
            - emg_std: standard deviation of emergency vehicle waiting time
            - K: standard deviation weighting factor (default 0.5)
            - Z: emergency vehicle penalty multiplier (default 1.0)
    
        Returns:
            reward: float
                - approximately in range [-150, 50]
                - positive values indicate good traffic conditions
                - negative values indicate congestion
        """
        config = self.REWARD_CONFIGS['project1_std_reward']
        K = config['K']
        Z = config['Z']
        base_reward = config['base_reward']
        ambulance_type_ids = config['ambulance_type_ids']
        
        # ========== 1. collect waiting time distribution from all observed lanes ==========
        regular_waiting_times = []  # regular vehicles
        emergency_waiting_times = []  # emergency vehicles
        
        eng = self.world.eng
        
        # iterate over all observed lanes
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                # lanes of concern -E15_1 and E81_1
                try:
                    # get all vehicles on lane
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        try:
                            # Get accumulated waiting time
                            waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id) # this caps at 100 s
                            
                            # Determine vehicle type
                            veh_type = eng.vehicle.getTypeID(veh_id)
                            
                            if veh_type in ambulance_type_ids:
                                # Emergency vehicle
                                emergency_waiting_times.append(waiting_time)
                            else:
                                # Regular vehicle
                                regular_waiting_times.append(waiting_time)
                        
                        except Exception as e:
                            # Skip vehicles with failed data retrieval
                            continue
                
                except Exception as e:
                    # Skip lanes with errors
                    continue
        
        # ========== 2. Compute statistics for regular vehicles ==========
        if len(regular_waiting_times) > 0:
            reg_mean = float(np.mean(regular_waiting_times))
            reg_std = float(np.std(regular_waiting_times))
        else:
            # no regular vehicles, set to 0 (ideal state)
            reg_mean = 0.0
            reg_std = 0.0
        
        # ========== 3. Compute statistics for emergency vehicles ==========
        if len(emergency_waiting_times) > 0:
            emg_mean = float(np.mean(emergency_waiting_times))
            emg_std = float(np.std(emergency_waiting_times))
        else:
            # No emergency vehicles
            emg_mean = 0.0
            emg_std = 0.0
        
        # ========== 4. Compute reward (fully aligned with formula) ==========
        reward = base_reward - (
            (reg_mean + K * reg_std) + 
            Z * (emg_mean + K * emg_std)
        )
        
        # ========== 5. Optional debugging info ==========
        # Uncomment if debugging is needed
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

    # I don't think this is actually used for anything
    def _compute_project1_std_reward_with_configurable_params_old(self, K=None, Z=None) -> float:
        """
        Project 1 standard deviation reward with configurable K and Z parameters
        (used for parameter sweep experiments)
    
        Args:
            K: standard deviation weight factor (if None, use config default)
            Z: emergency vehicle penalty multiplier (if None, use config default)
    
        Returns:
            reward: float
        """
        config = self.REWARD_CONFIGS['project1_std_reward']
        
        # Use provided or default parameters
        K = K if K is not None else config['K']
        Z = Z if Z is not None else config['Z']
        
        base_reward = config['base_reward']
        ambulance_type_ids = config['ambulance_type_ids']
        
        # Collect waiting times
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
        
        # Compute statistics
        reg_mean = float(np.mean(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0
        reg_std = float(np.std(regular_waiting_times)) if len(regular_waiting_times) > 0 else 0.0
        emg_mean = float(np.mean(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0
        emg_std = float(np.std(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0
        
        # Compute reward
        reward = base_reward - (
            (reg_mean + K * reg_std) + 
            Z * (emg_mean + K * emg_std)
        )
        
        return float(reward)
    
    # this probably never needs to be used again as it is just for the testing/evaluation metrics and we want groups in ours
    def get_reward_statistics_old(self) -> dict:
        """
        Get reward statistics at the current timestep (for analysis).
    
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
        
        # Collect waiting times
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
        
        # Compute statistics
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
        
        # Compute reward components
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

##############################################################################
# New code for our final year project (FYP)
# note the way I been switching what is used is put _new at end of new and remove _old when using Baseline
# and when using new reward, I put _old at end of old/baseline code and remove _new from new code so new FYP Reward is used
##############################################################################


    # TODO: this one HAS to be right for which reward you want to use
    def _compute_project1_std_reward_new(self) -> float: # TODO: rename this and fix all calls to this to reflect FYP better
        """
        Final year project reward
        
        reward = - np.sum(np.array(lane_weights) * np.array(lane_queues))
        
        Where:
            - Z: emergency vehicle penalty multiplier (default 1.0)
    
        Returns:
            reward: float
        """
        config = self.REWARD_CONFIGS['final_year_project_reward'] # TODO: look into how this works
        Z = config['Z']
        #base_reward = config['base_reward'] 
        ambulance_type_ids = config['ambulance_type_ids']
        
        eng = self.world.eng
        
        # --- STEP 1: Detect EV ---
        EV_present = False
        EV_lane_id = None
        EV_position = None
        EV_id = None
    
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                            EV_present = True
                            EV_lane_id = eng.vehicle.getLaneID(veh_id)
                            EV_position = eng.vehicle.getLanePosition(veh_id)
                            EV_id = veh_id
                            break
                    if EV_present:
                        break
                except:
                    continue
            if EV_present:
                break
    
        # --- STEP 2: Compute lane queues + weights ---
        lane_queues = []
        lane_weights = []
    
        eps = 1e-4
        max_weight = 75.0  # TODO: refine this tuning - prevents explosion
        max_tta = 60.0    # TODO: tune this - FOR NORMALISATION
    
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    q_i = eng.lane.getLastStepHaltingNumber(lane_id)
    
                    w_i = 1.0
    
                    if EV_present and lane_id == EV_lane_id:
                        lane_length = eng.lane.getLength(lane_id)
                        dist_to_stop = lane_length - EV_position
    
                        EV_speed = eng.vehicle.getSpeed(EV_id)
                        EV_speed = max(EV_speed, 1e-3)
    
                        EV_tta = dist_to_stop / EV_speed
                        EV_tta_norm = min(EV_tta / max_tta, 1.0)
    
                        w_i = 1 + Z / max(EV_tta_norm, eps)
                        w_i = min(w_i, max_weight)
    
                    lane_queues.append(q_i)
                    lane_weights.append(w_i)
    
                except:
                    continue
    
        # --- STEP 3: Compute reward ---
        lane_queues = np.array(lane_queues)
        lane_weights = np.array(lane_weights)
    
        total_weighted_queue = np.sum(lane_weights * lane_queues)
        reward = -total_weighted_queue
        
        # ========== 5. Optional debugging info ==========
        # Uncomment if debugging is needed
        # if hasattr(self.world, '_debug_reward_stats'):
        #     self.world._debug_reward_stats = {
        #         'lane_queues': lane_queues.tolist(),
        #         'lane_weights': lane_weights.tolist(),
        #         'total_weighted_queue': float(total_weighted_queue),
        #         'reward': float(reward),
        #         'EV_present': EV_present,
        #         'EV_lane_id': EV_lane_id,
        #     }
        
        return float(reward)

    # I don't think this is used for anything
    def _compute_project1_std_reward_with_configurable_params(self, Z=None) -> float:
        """
        Project 1 standard deviation reward with configurable Z parameters
        (used for parameter sweep experiments)
    
        Args:
            Z: emergency vehicle penalty multiplier (if None, use config default)
    
        Returns:
            reward: float
        """
        config = self.REWARD_CONFIGS['final_year_project_reward'] # look into how this works
        
        # Use provided or default parameters
        Z = Z if Z is not None else config['Z']
        
        #base_reward = config['base_reward'] # I dont think this does anything lol, but could break potentially?
        ambulance_type_ids = config['ambulance_type_ids']
        
        eng = self.world.eng
        
        # --- STEP 1: Detect EV (separate pass) ---
        EV_present = False
        EV_lane_id = None
        EV_position = None
        EV_id = None
        
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                            EV_present = True
                            EV_lane_id = eng.vehicle.getLaneID(veh_id)
                            EV_position = eng.vehicle.getLanePosition(veh_id)
                            EV_id = veh_id
                            break
                    if EV_present:
                        break
                except:
                    continue
            if EV_present:
                break
        
            # --- STEP 2: Compute lane queues + weights ---
        lane_queues = []
        lane_weights = []
    
        eps = 1e-4 # to avoid divide by zero error
        max_weight = 75.0  # refine this tuning - prevents explosion
        max_tta = 60.0    # tune this - FOR NORMALISATION
    
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    # Queue length = halting vehicles
                    q_i = eng.lane.getLastStepHaltingNumber(lane_id)
    
                    # Default weight
                    w_i = 1.0
    
                    # EV weighting (only for EV lane)
                    if EV_present and lane_id == EV_lane_id:
                        lane_length = eng.lane.getLength(lane_id)
                        dist_to_stop = lane_length - EV_position
    
                        EV_speed = eng.vehicle.getSpeed(EV_id)
                        EV_speed = max(EV_speed, 1e-3)
    
                        EV_tta = dist_to_stop / EV_speed
    
                        # Normalise TTA
                        EV_tta_norm = min(EV_tta / max_tta, 1.0)
    
                        # Compute weight (stable)
                        w_i = 1 + Z / max(EV_tta_norm, eps)
                        w_i = min(w_i, max_weight)
    
                    lane_queues.append(q_i)
                    lane_weights.append(w_i)
    
                except:
                    continue
                
        # --- STEP 3: Compute reward ---
        lane_queues = np.array(lane_queues)
        lane_weights = np.array(lane_weights)
    
        total_weighted_queue = np.sum(lane_weights * lane_queues)
        reward = -total_weighted_queue

        
        return float(reward)

    # This is only used for getting the stats for evaluation/testing - is NOT used for training
    def get_reward_statistics(self) -> dict:
        """
        Gets new reward statistics at the current timestep (for analysis).
        Group 1 = regular traffic that has potential to disrupt EVs – ie. They are infront and travelling along the EVs path
        Group 2 = all other regular traffic (ie. Has no reasonable potential to get in the way of the EV)
        
        Returns:
            stats: dict
                {
                    'regular_group1_vehicles': {
                        'count': int,
                        'mean_waiting': float,
                        'std_waiting': float,
                        'max_waiting': float,
                        'min_waiting': float
                    },
                    'regular_group2_vehicles': {
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
                    'reward_components': { # to be changed
                        'base': float,
                        'regular_penalty': float,
                        'emergency_penalty': float,
                        'total_reward': float
                    }
                }
        """
        config = self.REWARD_CONFIGS.get('final_year_project_reward', {}) # TODO: look into how this works
        Z = config.get('Z', 1.0)
        
        
        #base_reward = config.get('base_reward', 50.0) # for OLD REWARD
        ambulance_type_ids = config.get('ambulance_type_ids', ['ambulance_type', 'emergency'])
        
        # Collect waiting times
        regular_group1_waiting_times = []
        regular_group2_waiting_times = []
        emergency_waiting_times = []
        
        # for new reward
        lane_queues = []
        lane_weights = []
        
        eng = self.world.eng
        
        
        # assumes only 1 EV present at a time for now
        # TODO: make this not break for multiple EV scenario  
        EV_present = False
        EV_lane_id = None
        EV_position = None
        EV_id = None # this only works for one/the first detected EV at the moment...
        
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                            EV_present = True
                            EV_lane_id = eng.vehicle.getLaneID(veh_id)
                            EV_position = eng.vehicle.getLanePosition(veh_id)
                            EV_id = veh_id
                            break
                    if EV_present:
                        break
                except:
                    continue
            if EV_present:
                break
            
        div_error_avoider = 1e-4
        max_weight = 75.0  # TODO: refine this tuning - prevents explosion
        max_tta = 60.0    # TODO: tune this - FOR NORMALISATION
        
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    
                    # Queue length = number of halting vehicles (speed less than 0.1 m/s)
                    q_i = eng.lane.getLastStepHaltingNumber(lane_id)
                    
                    # Default lane weighting factor
                    w_i = 1.0
                        
                    # If EV is in this lane -> Compute EV_tta
                    if EV_present == True and lane_id == EV_lane_id:
                        # distance remaining along lane
                        lane_length = eng.lane.getLength(lane_id)
                        dist_to_stop = lane_length - EV_position
                        
                        # simple TTA approximation
                        EV_speed = eng.vehicle.getSpeed(EV_id)
                        EV_speed = max(EV_speed, 1e-3)
                        
                        EV_tta = dist_to_stop / EV_speed
                        
                        # normalise TTA (important)
                        EV_tta_norm = min(EV_tta / max_tta, 1.0)
                        
                        w_i = 1 + Z / max(EV_tta_norm, div_error_avoider)
                        
                        # to avoid explosion of wi
                        w_i = min(w_i, max_weight)
                    
                    lane_queues.append(q_i)
                    lane_weights.append(w_i)
                    
                    for veh_id in vehicle_ids:
                        try:
                            waiting_time = self.world.vehicle_timeloss_delta.get(veh_id, 0.0)
                            veh_type = eng.vehicle.getTypeID(veh_id)

                            if veh_type in ambulance_type_ids:
                                emergency_waiting_times.append(waiting_time)
                                # NEW: running per-EV total over its whole trip
                                self.world.vehicle_ev_delay_totals[veh_id] = (
                                    self.world.vehicle_ev_delay_totals.get(veh_id, 0.0) + waiting_time
                                )
                            else:
                                # NEW: running per-vehicle group totals, split by whichever
                                # group it's classified as THIS step - naturally handles a
                                # vehicle flipping group1 -> group2 -> group1 over its trip.
                                totals = self.world.vehicle_group_delay_totals.setdefault(
                                    veh_id, {'group1': 0.0, 'group2': 0.0}
                                )
                                if EV_present == True and lane_id == EV_lane_id:
                                    veh_position = eng.vehicle.getLanePosition(veh_id)
                                    if veh_position > EV_position:
                                        regular_group1_waiting_times.append(waiting_time)
                                        totals['group1'] += waiting_time
                                        self.world.vehicle_ever_group1.add(veh_id)
                                    else:
                                        regular_group2_waiting_times.append(waiting_time)
                                        totals['group2'] += waiting_time
                                else:
                                    regular_group2_waiting_times.append(waiting_time)
                                    totals['group2'] += waiting_time

                        except Exception as e:
                            continue
                except:
                    continue
        
        # Compute statistics
        stats = {
            'regular_group1_vehicles': {
                'count': len(regular_group1_waiting_times),
                'mean_waiting': float(np.mean(regular_group1_waiting_times)) if len(regular_group1_waiting_times) > 0 else 0.0,
                'std_waiting': float(np.std(regular_group1_waiting_times)) if len(regular_group1_waiting_times) > 0 else 0.0,
                'max_waiting': float(np.max(regular_group1_waiting_times)) if len(regular_group1_waiting_times) > 0 else 0.0,
                'min_waiting': float(np.min(regular_group1_waiting_times)) if len(regular_group1_waiting_times) > 0 else 0.0,
            },
            'regular_group2_vehicles': {
                'count': len(regular_group2_waiting_times),
                'mean_waiting': float(np.mean(regular_group2_waiting_times)) if len(regular_group2_waiting_times) > 0 else 0.0,
                'std_waiting': float(np.std(regular_group2_waiting_times)) if len(regular_group2_waiting_times) > 0 else 0.0,
                'max_waiting': float(np.max(regular_group2_waiting_times)) if len(regular_group2_waiting_times) > 0 else 0.0,
                'min_waiting': float(np.min(regular_group2_waiting_times)) if len(regular_group2_waiting_times) > 0 else 0.0,
            },
            'emergency_vehicles': {
                'count': len(emergency_waiting_times),
                'mean_waiting': float(np.mean(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
                'std_waiting': float(np.std(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
                'max_waiting': float(np.max(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
                'min_waiting': float(np.min(emergency_waiting_times)) if len(emergency_waiting_times) > 0 else 0.0,
            }
        }
        
        # new reward function (sum of wi * qi)
        lane_queues = np.array(lane_queues)
        lane_weights = np.array(lane_weights)
        
        total_weighted_queue = np.sum(lane_weights * lane_queues)
        reward = -total_weighted_queue
        
        stats['reward_components'] = {
            'lane_queues': lane_queues,
            'lane_weights': lane_weights,
            'total_weighted_queue': float(total_weighted_queue),
            'total_reward': float(reward),
            'Z': Z,
        }
        
        # Combined regular vehicle stats for backward compatibility with training script
        all_regular = regular_group1_waiting_times + regular_group2_waiting_times
        stats['regular_vehicles'] = {
            'count': len(all_regular),
            'mean_waiting': float(np.mean(all_regular)) if all_regular else 0.0,
            'std_waiting': float(np.std(all_regular)) if all_regular else 0.0,
            'max_waiting': float(np.max(all_regular)) if all_regular else 0.0,
            'min_waiting': float(np.min(all_regular)) if all_regular else 0.0,
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
        
    #     # ✅ compute difference-based reward
    #     if self.last_reward is not None:
    #         # difference: previous - current (improvement is positive)
    #         reward = self.last_reward - reward_array
    #     else:
    #         # first call
    #         reward = np.zeros_like(reward_array)
    #      # update historical value
    #     self.last_reward = reward_array.copy()
    #     # ✅ add reward normalization/scaling
    #     # option 1: divide by a constant scale (recommended)
    #     # reward_array = reward_array / 1000.0  # adjust based on actual reward range
        
    #     # clip to reasonable range
    #     # reward_array = np.clip(reward_array, -10.0, 10.0)
        
    #     # option 3: use tanh compression (if reward range is large)
    #     # reward_array = np.tanh(reward_array / 1000.0) * 10.0
    #     return reward