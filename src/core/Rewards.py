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
            'ambulance_type_ids': ['ambulance_type', 'emergency'],  # emergency vehicle type list
            # NEW: which per-vehicle EV metric feeds ev_mean/ev_std in
            # _compute_project1_std_reward(). Options:
            #   'waiting_time'     -> eng.vehicle.getAccumulatedWaitingTime(veh_id)
            #                         (same metric as OLD/baseline used for EVs;
            #                         capped at 100s, only counts near-full stops)
            #   'delay_window'     -> self.world.vehicle_timeloss_delta.get(veh_id) <------------ THIS IS THE ONE FOR THE BEST RESULTS (have to set Z value high though (like 50s))
            #                         (CURRENT default: per-decision-window delta,
            #                         small magnitude, resets every step, doesn't
            #                         persist across the EV's time at this intersection)
            #   'delay_cumulative' -> eng.vehicle.getTimeLoss(veh_id)
            #                         (SUMO's own running cumulative delay for this
            #                         vehicle since it entered the sim - same
            #                         "any slowdown counts" concept as delay_window,
            #                         but persistent like waiting_time is, and needs
            #                         no manual per-episode reset since it's scoped
            #                         to the vehicle's own lifetime in the simulation)
            'ev_metric_mode': 'delay_window', # TODO: change this depending what reward thingy to use
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
        # old TODO??: register it in Registry
        for r in roads:
            if not self.world.RIGHT:
                tmp = sorted(self.ts.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
            else:
                tmp = sorted(self.ts.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
            self.lanes_road_observed.append(tmp)
            # old TODO??: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
            
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

    # =====================================================================
    # ================= Lexicographic multi-objective reward ==============
    # =====================================================================
    # Design: r1 (EV) is priority level 0 (highest), r2 (regular vehicles)
    # is priority level 1. Both are computed entirely in this file now -
    # r1 used to read a value cached by env.py, but that's gone: everything
    # a GetRewards instance needs (self.world.eng, self.world.
    # vehicle_timeloss_delta) is already available here, so there was no
    # real reason to split it across two files.
    #
    # EVERYTHING YOU'D WANT TO TWEAK LIVES IN THIS FILE:
    #   - LEXICOGRAPHIC_CONFIG below: scaling constants, weights, etc.
    #   - _lex_objective_ev_priority() / _lex_objective_regular_vehicles():
    #     each collects a list of per-vehicle numbers, then has ONE clearly
    #     marked line that combines the list into the final reward. That's
    #     the line to change for "try a different formula".
    #   - LEXICOGRAPHIC_OBJECTIVES: the priority ORDER (index 0 = highest).
    #
    # TO ADD A THIRD OBJECTIVE: write a new `_lex_objective_<name>` method
    # below following the same pattern, add its name to
    # LEXICOGRAPHIC_OBJECTIVES at whatever priority position you want, and
    # bump n_objectives in your YAML to match. Nothing else needs to
    # change - MAPPOagent.py reads n_objectives off the length of the
    # vector _compute_lexicographic_reward_vector() returns.
    LEXICOGRAPHIC_OBJECTIVES = [
        'ev_priority',        # r1 - highest priority - system-wide
        'regular_vehicles',   # r2 - lowest priority  - local to this intersection
    ]

    # TODO: config things for lexiocographic reward - IF USING THIS

    LEXICOGRAPHIC_CONFIG = {
        'ambulance_type_ids': ['ambulance_type', 'emergency'],
        # Scaling parameters - the first things you'll likely want to play
        # with. Both multiply the final reward AFTER it's combined below.
        'ev_scale': 5.0,
        'reg_scale': 1.0,

        # ---- r1 EV metric switch ----
        # Same three options and same tradeoffs as project1_std_reward's
        # 'ev_metric_mode' (see REWARD_CONFIGS above) - r1 currently sums
        # 'delay_window' (self.world.vehicle_timeloss_delta) across every EV
        # in the sim each step: small-magnitude, resets every step, no
        # persistence. 'delay_cumulative' (eng.vehicle.getTimeLoss) is the
        # persistent version of the same delay concept; 'waiting_time'
        # matches what the OLD/baseline aggregated reward used for EVs.
        'ev_metric_mode': 'delay_window',

        # ---- r2 mode switch ----
        # 'pooled'      : all non-EV vehicles observed by this intersection
        #                 pooled into one delay list (original behaviour).
        # 'group_split' : split into group1 (vehicles ahead of the EV, in
        #                 its lane) and group2 (everyone else), each with
        #                 its own scale + std weight. See
        #                 _lex_r2_group_split() for the exact formula.
        'reg_mode': 'pooled', # TODO: pick if you use All reg vehs (pooled) or group 1 and group 2 (group_split) for R2

        # --- pooled-mode params ---
        # Example of the "mean + weight * std" formula you mentioned -
        # 0.0 reproduces the old plain-sum behaviour exactly. Set it
        # nonzero to penalise UNEVEN delay (a few very-delayed vehicles)
        # on top of total delay.
        'reg_std_weight': 0.5,

        # --- group_split-mode params ---
        # combined = sum(g1)*group1_scale + sum(g2)*group2_scale
        #          + std(g1)*group1_std_weight + std(g2)*group2_std_weight
        # then the whole thing is scaled by reg_scale, same as pooled mode -
        # so reg_scale always controls r1-vs-r2 balance, and
        # group1_scale/group2_scale only control the balance WITHIN r2.
        'group1_scale': 1.5,
        'group2_scale': 1.0,
        'group1_std_weight': 0,
        'group2_std_weight': 0,
    }

    def _lex_objective_ev_priority(self) -> float: # modify this if want to try different r1 computations (EV)
        """
        r1: system-wide EV objective. Looks at EVERY vehicle currently in
        the simulation (not just this intersection's lanes) because a
        single EV's delay is a corridor-wide property - this makes r1
        identical across J1/J2/J3 automatically, without needing any
        broadcast/caching machinery in env.py.
        """
        cfg = self.LEXICOGRAPHIC_CONFIG
        ambulance_type_ids = cfg['ambulance_type_ids']
        eng = self.world.eng
        current_ev_ids = []
        try:
            for veh_id in eng.vehicle.getIDList():
                try:
                    if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                        current_ev_ids.append(veh_id)
                except Exception:
                    continue
        except Exception:
            pass

        # NEW: pick the per-vehicle EV metric according to ev_metric_mode,
        # instead of always reading vehicle_timeloss_delta directly - see
        # LEXICOGRAPHIC_CONFIG['ev_metric_mode'] for what each option means.
        def _ev_metric(vid):
            mode = cfg.get('ev_metric_mode', 'delay_window')
            try:
                if mode == 'waiting_time':
                    return eng.vehicle.getAccumulatedWaitingTime(vid)
                elif mode == 'delay_cumulative':
                    return eng.vehicle.getTimeLoss(vid)
                else:  # 'delay_window' - original behaviour
                    return self.world.vehicle_timeloss_delta.get(vid, 0.0)
            except Exception:
                return 0.0

        # ---- COMBINE STEP: this is the line to edit for a different r1 formula ----
        # 0.0 (neutral, not a penalty) whenever there's no EV at all - this
        # is what keeps r1 well-behaved during EV-free stretches, see
        # _update_lexicographic_duals in MAPPOagent.py for how that's used.
        combined = sum(_ev_metric(vid) for vid in current_ev_ids) if current_ev_ids else 0.0  # Original version
        #combined = max((self.world.vehicle_timeloss_delta.get(vid, 0.0) for vid in current_ev_ids), default=0.0)  # alternate idea i tried - doesn't seem to be as good
        
        return -float(combined) * cfg['ev_scale']

    def _lex_objective_regular_vehicles(self) -> float:
        """
        r2: local regular-vehicle objective for THIS intersection only.
        Dispatches on LEXICOGRAPHIC_CONFIG['reg_mode']:
          'pooled'      -> _lex_r2_pooled()
          'group_split' -> _lex_r2_group_split()
        Both return the same thing: a single float, already scaled by
        reg_scale, ready to drop straight into the reward vector.
        """
        cfg = self.LEXICOGRAPHIC_CONFIG
        mode = cfg.get('reg_mode', 'pooled')
        if mode == 'group_split':
            return self._lex_r2_group_split()
        return self._lex_r2_pooled()

    def _lex_r2_pooled(self) -> float: # this is the function for the mode where you treat ALL regular vehicles equally (for R2)
        """
        r2 (pooled mode): every non-EV vehicle observed by this
        intersection pooled into one delay list - group1/group2 are NOT
        distinguished here (that split is available for diagnostics via
        get_reward_statistics(), it's just not used to gate training in
        this mode).
        """
        cfg = self.LEXICOGRAPHIC_CONFIG
        ambulance_type_ids = cfg['ambulance_type_ids']
        eng = self.world.eng

        delays = []
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    continue
                for veh_id in vehicle_ids:
                    try:
                        if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                            continue
                        delays.append(self.world.vehicle_timeloss_delta.get(veh_id, 0.0))
                    except Exception:
                        continue

        # ---- COMBINE STEP: this is the line to edit for a different pooled r2 formula ----
        # Default (reg_std_weight=0.0): plain total delay, same as before.
        # reg_std_weight>0 adds an extra penalty for how UNEVENLY that
        # delay is spread across vehicles.
        if delays:
            combined = np.sum(delays) + cfg['reg_std_weight'] * np.std(delays)
        else:
            combined = 0.0

        return -float(combined) * cfg['reg_scale']

    def _lex_r2_group_split(self) -> float: # this is the function for the mode where you split regular vehicles into Groups 1 (EV blocking) and Groups 2 (all other reg vehs) for Reward 2 (R2) computation
        """
        r2 (group_split mode): separates this intersection's non-EV
        vehicles into group1 (ahead of the EV, in its lane - the ones that
        can actually block it) and group2 (everyone else), each with its
        own scale and std weight:

            combined = sum(g1)*group1_scale + sum(g2)*group2_scale
                     + std(g1)*group1_std_weight + std(g2)*group2_std_weight

        then scaled by reg_scale (same overall r1-vs-r2 balance knob as
        pooled mode - group1_scale/group2_scale only control the balance
        WITHIN r2). If no EV is present this step, every regular vehicle
        falls into group2 by definition (there's nothing to be "ahead of"),
        matching the convention already used in get_reward_statistics().

        EV detection here only tracks the FIRST EV found per intersection,
        same limitation as elsewhere in this file (get_reward_statistics(),
        the old project1_std_reward_new) - fine for the current
        single-EV-at-a-time scenario, would need extending for multi-EV.
        """
        cfg = self.LEXICOGRAPHIC_CONFIG
        ambulance_type_ids = cfg['ambulance_type_ids']
        eng = self.world.eng

        # --- STEP 1: find the (first) EV in this intersection's lanes ---
        EV_present = False
        EV_lane_id = None
        EV_position = None

        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                            EV_present = True
                            EV_lane_id = eng.vehicle.getLaneID(veh_id)
                            EV_position = eng.vehicle.getLanePosition(veh_id)
                            break
                    if EV_present:
                        break
                except Exception:
                    continue
            if EV_present:
                break

        # --- STEP 2: split non-EV vehicles into group1 / group2 ---
        delays_g1 = []
        delays_g2 = []
        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    continue
                for veh_id in vehicle_ids:
                    try:
                        if eng.vehicle.getTypeID(veh_id) in ambulance_type_ids:
                            continue
                        delay = self.world.vehicle_timeloss_delta.get(veh_id, 0.0)
                        if EV_present and lane_id == EV_lane_id:
                            veh_position = eng.vehicle.getLanePosition(veh_id)
                            if veh_position > EV_position:
                                delays_g1.append(delay)
                            else:
                                delays_g2.append(delay)
                        else:
                            delays_g2.append(delay)
                    except Exception:
                        continue

        # ---- COMBINE STEP: this is the line to edit for a different group_split r2 formula ----
        sum_g1 = np.sum(delays_g1) if delays_g1 else 0.0
        sum_g2 = np.sum(delays_g2) if delays_g2 else 0.0
        std_g1 = np.std(delays_g1) if delays_g1 else 0.0
        std_g2 = np.std(delays_g2) if delays_g2 else 0.0

        combined = (
            sum_g1 * cfg['group1_scale'] + sum_g2 * cfg['group2_scale']
            + std_g1 * cfg['group1_std_weight'] + std_g2 * cfg['group2_std_weight']
        )

        return -float(combined) * cfg['reg_scale']

    def _compute_lexicographic_reward_vector(self) -> np.ndarray:
        """
        Returns np.array([r1, r2, ...]) in priority order (index 0 = highest
        priority), built from LEXICOGRAPHIC_OBJECTIVES. This is what feeds
        the lexicographic MAPPO critic/actor - see MAPPOagent.py.
        """
        values = []
        for name in self.LEXICOGRAPHIC_OBJECTIVES:
            fn = getattr(self, f'_lex_objective_{name}')
            values.append(fn())
        return np.array(values, dtype=np.float32)
    
    # ========================================================================

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
        
        # NEW FOR FYP: exploratory variant - delay (timeLoss-based) for BOTH
        # regular and emergency vehicles, instead of only EVs. Checked before
        # the 'project1'/'fyp' substring checks below since algorithm_name
        # strings like 'project1_delay_all' or 'fyp_delay_all' would also
        # match those broader checks.
        if 'delay_all' in self.algorithm_name.lower():
            return self._compute_project1_std_reward_delay_all()
        
        # Project1 / std-DQN special case
        if 'project1' in self.algorithm_name.lower() or 'std_dqn' in self.algorithm_name.lower():
            return self._compute_project1_std_reward()
        
        # Lexicographic multi-objective case - returns a VECTOR (np.ndarray),
        # not a float. Everything downstream (env.py, parlenv.py,
        # MAPPOagent.py) is written to accept either shape.
        if 'lexicographic' in self.algorithm_name.lower() or 'lmorl' in self.algorithm_name.lower():
            return self._compute_lexicographic_reward_vector()

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
    def _compute_project1_std_reward_old(self) -> float:
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

    def _compute_project1_std_reward(self) -> float: # TODO: rename this to '_compute_project1_std_reward' when want to use it 
            """
            Project 1 standard deviation-aware reward.
        
            Reward formula (aligned with Project1 Agent.ipynb):
                reward = 50 - ((reg_mean + K * reg_std) + Z * (ev_delay_mean + K * ev_delay_std))
        
            Where:
                - reg_mean: mean waiting time of regular vehicles (getAccumulatedWaitingTime)
                - reg_std: standard deviation of regular vehicle waiting time ⭐ core innovation
                - ev_delay_mean: mean timeLoss-based delay of emergency vehicles (NEW FOR FYP -
                  see vehicle loop below; this is NOT the same metric as reg_mean)
                - ev_delay_std: standard deviation of emergency vehicle delay
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
            # NEW: which per-vehicle EV metric to use - see the comment on
            # 'ev_metric_mode' in REWARD_CONFIGS['project1_std_reward'] above
            # for what each option means and why it matters.
            ev_metric_mode = config.get('ev_metric_mode', 'delay_window')
            
            # ========== 1. collect waiting-time / delay distribution from all observed lanes ==========
            regular_waiting_times = []  # regular vehicles - getAccumulatedWaitingTime
            ev_delays = []  # emergency vehicles - metric depends on ev_metric_mode
            
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
                                # Determine vehicle type
                                veh_type = eng.vehicle.getTypeID(veh_id)
    
                                if veh_type in ambulance_type_ids:
                                    # Emergency vehicle - metric selectable via
                                    # ev_metric_mode:
                                    #   'waiting_time'     -> getAccumulatedWaitingTime
                                    #                         (capped at 100s, only counts
                                    #                         near-full stops - same metric
                                    #                         the OLD/baseline reward used)
                                    #   'delay_window'      -> World.vehicle_timeloss_delta,
                                    #                         delay accrued THIS decision
                                    #                         window only (small magnitude,
                                    #                         resets every step)
                                    #   'delay_cumulative'  -> eng.vehicle.getTimeLoss,
                                    #                         SUMO's own running delay total
                                    #                         for this vehicle since it
                                    #                         entered the sim - persistent
                                    #                         like waiting_time, but still
                                    #                         counts any sub-max-speed driving
                                    #                         rather than just full stops
                                    if ev_metric_mode == 'waiting_time':
                                        delay = eng.vehicle.getAccumulatedWaitingTime(veh_id)
                                    elif ev_metric_mode == 'delay_cumulative':
                                        delay = eng.vehicle.getTimeLoss(veh_id)
                                    else:  # 'delay_window' - original behaviour
                                        delay = self.world.vehicle_timeloss_delta.get(veh_id, 0.0)
                                    ev_delays.append(delay)
                                else:
                                    # Regular vehicle - unchanged: accumulated waiting time
                                    waiting_time = eng.vehicle.getAccumulatedWaitingTime(veh_id) # this caps at 100 s
                                    regular_waiting_times.append(waiting_time)
    
                            except Exception as e:
                                # Skip vehicles with failed data retrieval
                                continue
    
                    except Exception as e:
                        # Skip lanes with errors
                        continue
    
            
            # ========== 2. Compute statistics for regular vehicles (waiting time) ==========
            if len(regular_waiting_times) > 0:
                reg_mean = float(np.mean(regular_waiting_times))
                reg_std = float(np.std(regular_waiting_times))
            else:
                # no regular vehicles, set to 0 (ideal state)
                reg_mean = 0.0
                reg_std = 0.0
            
            # ========== 3. Compute statistics for emergency vehicles (delay) ==========
            if len(ev_delays) > 0:
                ev_delay_mean = float(np.mean(ev_delays))
                ev_delay_std = float(np.std(ev_delays))
            else:
                # No emergency vehicles
                ev_delay_mean = 0.0
                ev_delay_std = 0.0
            
            # ========== 4. Compute reward (fully aligned with formula) ==========
            reward = base_reward - (
                (reg_mean + K * reg_std) + 
                Z * (ev_delay_mean + K * ev_delay_std)
            )
            
            # ========== 5. Optional debugging info ==========
            # Uncomment if debugging is needed
            # if hasattr(self.world, '_debug_reward_stats'):
            #     self.world._debug_reward_stats = {
            #         'reg_mean': reg_mean,
            #         'reg_std': reg_std,
            #         'ev_delay_mean': ev_delay_mean,
            #         'ev_delay_std': ev_delay_std,
            #         'reward': reward,
            #         'num_regular': len(regular_waiting_times),
            #         'num_emergency': len(ev_delays)
            #     }
            
            return float(reward)

    def _compute_project1_std_reward_delay_all(self) -> float: # rename this one to '_compute_project1_std_reward' if which for it to be the reward function used
        """
        NEW FOR FYP - exploratory variant of _compute_project1_std_reward.

        Identical std-aware formula, but BOTH regular and emergency vehicles
        use timeLoss-based delay (World.vehicle_timeloss_delta, accrued THIS
        decision window) instead of regular vehicles using
        getAccumulatedWaitingTime(). Use this to compare against the
        EV-only-delay version above. Select it by putting 'delay_all'
        anywhere in algorithm_name (see compute_reward()), e.g.
        'fyp_delay_all' or 'project1_delay_all'.

        Reward formula (same as _compute_project1_std_reward):
            reward = base_reward - ((reg_mean + K * reg_std) + Z * (emg_mean + K * emg_std))

        Note on scale: getAccumulatedWaitingTime is a running "current wait
        streak" capped at 100s, whereas vehicle_timeloss_delta is delay
        accrued only within the current decision window. Switching regular
        vehicles onto delay will likely shrink reg_mean/reg_std substantially
        relative to the baseline version - expect to retune K/base_reward if
        you compare runs.
        """
        config = self.REWARD_CONFIGS['project1_std_reward']
        K = config['K']
        Z = config['Z']
        base_reward = config['base_reward']
        ambulance_type_ids = config['ambulance_type_ids']

        regular_delays = []    # regular vehicles
        emergency_delays = []  # emergency vehicles

        eng = self.world.eng

        for road_lanes in self.lanes_road_observed:
            for lane_id in road_lanes:
                try:
                    vehicle_ids = eng.lane.getLastStepVehicleIDs(lane_id)
                    for veh_id in vehicle_ids:
                        try:
                            veh_type = eng.vehicle.getTypeID(veh_id)
                            delay = self.world.vehicle_timeloss_delta.get(veh_id, 0.0)

                            if veh_type in ambulance_type_ids:
                                emergency_delays.append(delay)
                            else:
                                regular_delays.append(delay)

                        except Exception as e:
                            continue

                except Exception as e:
                    continue

        if len(regular_delays) > 0:
            reg_mean = float(np.mean(regular_delays))
            reg_std = float(np.std(regular_delays))
        else:
            reg_mean = 0.0
            reg_std = 0.0

        if len(emergency_delays) > 0:
            emg_mean = float(np.mean(emergency_delays))
            emg_std = float(np.std(emergency_delays))
        else:
            emg_mean = 0.0
            emg_std = 0.0

        reward = base_reward - (
            (reg_mean + K * reg_std) +
            Z * (emg_mean + K * emg_std)
        )

        return float(reward)

    def _compute_project1_std_reward_progress_report_ver(self) -> float: 
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