import os
import gymnasium as gym
from math import atan2, pi
import numpy as np
import xml.etree.ElementTree as ET
import traci
import sumolib
import libsumo
import json
from .Rewards import GetRewards
from .Observations import Observation, get_fyp_observation_dims

class parse_sumo_config(): # Does static information extraction affect parallel simulation?
    def __init__(self, sumo_config, **kwargs):
        self.sumo_config = sumo_config
        with open(sumo_config) as f:
            self.sumo_dict = json.load(f)
        # ✅ Received from training script (must be provided, not read from CFG)）
        self._obs_to_subscribe = kwargs.get('obs_to_subscribe')
        self._reward_to_subscribe = kwargs.get('reward_to_subscribe')
        self._algorithm_name = kwargs.get('algorithm_name')
        self._normalize_observation = kwargs.get('normalize_observation', False)
        self._norm_params = kwargs.get('norm_params', {})  # ✅ added line
        self.RIGHT = True
        self.traffic_light_ids = []

        if kwargs['interface'] == 'libsumo':
            self.interface_flag = True
        elif kwargs['interface'] == 'traci':
            self.interface_flag = False
        else:
            raise Exception('NOT IMPORTED YET')
        # ✅ Add traffic scaling factor support
        self.traffic_scale = kwargs.get('traffic_scale', 1.0)
        self.seed = kwargs.get('seed', None)

        self.connection_name = self.get_connection_name()
        if self.interface_flag:
                self.eng = libsumo
        else:            
            self.eng = traci

        # 🔑 Call preprocessing here (order is important!)
        self.traffic_light_ids = self._get_traffic_light_ids()  # parse traffic light IDs first
        self.traffic_light_info = self._parse_all_traffic_light_info()

    def generate_sumo_cmd(self):
        if self.sumo_dict['gui'] == "True" or self.sumo_dict['gui'] == True:
            sumo_cmd = [sumolib.checkBinary('sumo-gui')]
        else:
            sumo_cmd = [sumolib.checkBinary('sumo')]
        if not self.sumo_dict.get('combined_file'):
            sumo_cmd += ['-n', os.path.join(self.sumo_dict['dir'], self.sumo_dict['roadnetFile']),
                        '-r', os.path.join(self.sumo_dict['dir'], self.sumo_dict['flowFile']),
                        '--no-warnings', str(self.sumo_dict['no_warning'])]
        else:
            sumo_cmd += ['-c', os.path.join(self.sumo_dict['dir'], self.sumo_dict['combined_file']),
                        '--no-warnings', str(self.sumo_dict['no_warning'])]
        # 1. Enable console statistics output
        sumo_cmd += ['--duration-log.statistics', 'true']
        
        # 2. Add random seed (if provided)
        if self.seed is not None:
            sumo_cmd += ['--seed', str(self.seed)]
        
        # 3. Add traffic scaling factor
        if self.traffic_scale != 1.0:
            sumo_cmd += ['--scale', str(self.traffic_scale)]
            print(f"⚙️ SUMO 流量缩放因子: {self.traffic_scale}")
            
        # 4. Add support for additional files (e.g. rerouter)
        if self.sumo_dict.get('additional_files'):
            additional_files = self.sumo_dict['additional_files']
            if isinstance(additional_files, str):
                additional_files = [additional_files]
            
            for add_file in additional_files:
                # If relative path, prepend directory
                if not os.path.isabs(add_file):
                    add_file = os.path.join(self.sumo_dict['dir'], add_file)
                sumo_cmd += ['--additional-files', add_file]
                print(f"⚙️ SUMO 加载 additional file: {add_file}")
        # # 2. 输出详细统计到文件
        # stats_dir = os.path.join(self.sumo_dict['dir'], 'statistics')
        # os.makedirs(stats_dir, exist_ok=True)
        
        # # 获取唯一的文件名（避免覆盖）
        # import time
        # script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # 例如 'train_parl_dqn'
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # # 创建输出目录：output_results/脚本名_时间戳/
        # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # A/ 的父目录
        # output_dir = os.path.join(project_root, 'output_results', f'{script_name}_{timestamp}')
        # os.makedirs(output_dir, exist_ok=True)
        
        # # 添加统计输出文件
        # sumo_cmd += [
        #     '--tripinfo-output', os.path.join(output_dir, 'tripinfo.xml'),
        #     '--statistic-output', os.path.join(output_dir, 'statistics.xml'),
        #     '--summary', os.path.join(output_dir, 'summary.xml'),
        # ]
        return sumo_cmd
    
    def get_net_file_address(self):
        net = os.path.join(self.sumo_dict['dir'], self.sumo_dict['roadnetFile'])
        return net

    def no_warning(self):
        return self.sumo_dict['no_warning']
    
    def get_connection_name(self):
        return self.sumo_dict['name']
    
    def get_map_name(self):
        return self.sumo_dict['roadnetFile'].split('/')[-1].split('.')[0]
    
    def get_decision_interval(self):
        return self.sumo_dict['decision_interval']

    def get_min_green(self):
        return self.sumo_dict['min_green']

    def get_yellow_length(self):
        """从配置文件获取黄灯时长"""
        return self.sumo_dict.get('yellow_length', 3)  # default 3 seconds
    # road/lane related functions: all roads, lanes
    def _get_roads(self):
        # # 方法1: 使用sumolib
        # import sumolib
        # net = sumolib.net.readNet(self.net)
        # road_ids = [edge.getID() for edge in net.getEdges()]
    
        # 方法2: 直接XML解析
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.get_net_file_address())
        root = tree.getroot()
        road_ids = [edge.get('id') for edge in root.findall('edge') 
                    if edge.get('id') and not edge.get('function')]  # exclude internal edges
        return road_ids
    
    def _get_lanes(self):
        # # 方法1: 使用sumolib
        # import sumolib
        # net = sumolib.net.readNet(self.net)
        # lane_ids = []
        # for edge in net.getEdges():
        #     for lane in edge.getLanes():
        #         lane_ids.append(lane.getID())
    
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.get_net_file_address())
        root = tree.getroot()
        
        lane_ids = []
        for edge in root.findall('edge'):
            # Skip internal edges (function="internal")
            if edge.get('function') == 'internal':
                continue
                
            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                if lane_id:
                    lane_ids.append(lane_id)
        
        return lane_ids

    # intersection related functions: valid phases
    def _get_traffic_light_ids(self):
        '''
        Optimized method to get traffic light IDs before starting SUMO simulation.
        This method parses the network file directly using sumolib, avoiding the need
        to start SUMO just to get the traffic light IDs.
        
        :return: list of traffic light IDs
        '''         
        # get from traffic light programs (more reliable)
        if not self.traffic_light_ids:
            try:
                # Parse XML directly to get tlLogic IDs
                import xml.etree.ElementTree as ET
                tree = ET.parse(self.get_net_file_address())
                root = tree.getroot()
                self.traffic_light_ids = [tl.get('id') for tl in root.findall('tlLogic')]
            except Exception as xml_e:
                print(f"XML parsing also failed: {xml_e}")
        
        if not self.traffic_light_ids:
            print("Warning: No traffic lights found in network file")
        
        print(f"Pre-parsed {len(self.traffic_light_ids)} traffic lights from network file: {self.get_map_name()}")
        return self.traffic_light_ids
    
    def _generate_valid_phase(self):
        '''
        Generate valid green phases using SUMO's traffic light logic directly.
        
        :return: dictionary of valid phases for each intersection
        '''
        valid_phases = dict()
        
        # Parse network file directly to get traffic light phases
        
        tree = ET.parse(self.get_net_file_address())
        root = tree.getroot()
        
        # Get phases from tlLogic elements
        for tl_logic in root.findall('tlLogic'):
            tl_id = tl_logic.get('id')
            
            if tl_id in self.traffic_light_ids:
                valid_phases[tl_id] = []
                
                # Extract all phases for this traffic light
                for phase in tl_logic.findall('phase'):
                    phase_state = phase.get('state')
                    phase_duration = float(phase.get('duration', 30))
                    
                    # Only include phases with green lights (G or g)
                    if phase_state and ('G' in phase_state or 'g' in phase_state) and ('y' not in phase_state) and ('Y' not in phase_state):
                        existing_states = [p.state for p in valid_phases[tl_id]]
                        if phase_state not in existing_states:
                        # 创建 Phase 对象（此时 SUMO 已经启动）
                            valid_phases[tl_id].append(self.eng.trafficlight.Phase(phase_duration, phase_state))
                        
                
                if not valid_phases[tl_id]:
                    print(f"Warning: No valid green phases found for {tl_id} in network file")
        
        # Check if we got phases for all intersections
        missing_intersections = set(self.traffic_light_ids) - set(valid_phases.keys())
        if missing_intersections:
            print(f"Warning: No phase data found for intersections: {missing_intersections}")

        # Validate that all intersections have at least one phase
        empty_intersections = [tl_id for tl_id, phases in valid_phases.items() if not phases]
        if empty_intersections:
            print(f"Warning: No green phases found for intersections: {empty_intersections}")

        # for tl_id, phases in valid_phases.items():
        #     print(f"  {tl_id}: {len(phases)} green phases") 
        return valid_phases


    def _parse_all_lanelinks(self):
        """
        Parse controlled links for all traffic lights (single XML pass).
        Results are cached for later use.
        
        Returns:
            Dictionary mapping traffic light ID to controlled links
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.get_net_file_address())
        root = tree.getroot()
        
        # Initialize for all traffic lights
        tl_lanelinks = {tl_id: {} for tl_id in self.traffic_light_ids}
        
        # Single pass through all connections
        for connection in root.findall('connection'):
            tl_id = connection.get('tl')
            
            if tl_id and tl_id in tl_lanelinks:
                link_index = int(connection.get('linkIndex', 0))
                from_edge = connection.get('from')
                to_edge = connection.get('to')
                from_lane_idx = connection.get('fromLane')
                to_lane_idx = connection.get('toLane')
                via = connection.get('via', '')
                
                from_lane = f"{from_edge}_{from_lane_idx}"
                to_lane = f"{to_edge}_{to_lane_idx}"
                
                if link_index not in tl_lanelinks[tl_id]:
                    tl_lanelinks[tl_id][link_index] = []
                
                tl_lanelinks[tl_id][link_index].append((from_lane, to_lane, via))
        
        # Convert to list format for each TL
        result = {}
        for tl_id, links_dict in tl_lanelinks.items():
            if links_dict:
                max_index = max(links_dict.keys())
                result[tl_id] = [links_dict.get(i, []) for i in range(max_index + 1)]
            else:
                result[tl_id] = []
        
        return result

    def get_controlled_links(self, tl_id):
        """
        Get controlled links for a specific traffic light (TraCI-compatible API).
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            List of controlled links (same format as TraCI)
        """
        if not hasattr(self, 'lanelinks_static'):
            self.lanelinks_static = self._parse_all_lanelinks()
        
        return self.lanelinks_static.get(tl_id, [])
    
    def _sort_roads(self):
        '''
        _sort_roads
        Sort roads information by arranging an order.
        
        :param: None
        :return: None
        '''
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i],
                                      self.outs[i] if self.RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]  # old todo was here - check if its 4

    
    # old todo was here - revert x and y
    def _get_direction(self, road, out=True):
        if out:
            x = road[1][0] - road[0][0]
            y = road[1][1] - road[0][1]
        else:
            x = road[-2][0] - road[-1][0]
            y = road[-2][1] - road[-1][1]
        tmp = atan2(x, y)
        return tmp if tmp >= 0 else (tmp + 2 * pi)

    def get_lane_shape_from_net(self, lane_id):
        tree = ET.parse(self.get_net_file_address())
        root = tree.getroot()
        
        for lane in root.iter('lane'):
            if lane.get('id') == lane_id:
                shape_str = lane.get('shape')
                if shape_str is None:
                    return []
                
                # 解析为 [(x, y), (x, y), ...]
                points = []
                for pair in shape_str.strip().split():
                    x, y = map(float, pair.split(','))
                    points.append((x, y))
                return points
        
        # 如果找不到该 lane
        return []

    def _get_road_lane_mapping(self):
        self.road_lane_mapping = {}
        self.roads = []
        self.outs = []
        self.directions = []
        for link in self.lanelinks:
            # skip if empty link
            if not link:
                continue
            link = link[0]
            if link[0][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[0][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[0][:-2]].append(link[0])
                self.roads.append(link[0][:-2])
                self.outs.append(False)
                road = self.get_lane_shape_from_net(link[0])
                self.directions.append(self._get_direction(road, False))
            elif link[0][:-2] in self.road_lane_mapping.keys() and link[0] not in self.road_lane_mapping[link[0][:-2]]:
                self.road_lane_mapping[link[0][:-2]].append(link[0])
            if link[1][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[1][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[1][:-2]].append(link[1])
                self.roads.append(link[1][:-2])
                self.outs.append(True)
                road = self.get_lane_shape_from_net(link[1])
                self.directions.append(self._get_direction(road, True))
            elif link[1][:-2] in self.road_lane_mapping.keys() and link[1] not in self.road_lane_mapping[link[1][:-2]]:
                self.road_lane_mapping[link[1][:-2]].append(link[1])
                self.lanes = []
        
        self.out_roads = []
        self.in_roads = []
        self._sort_roads()
        for key in self.road_lane_mapping.keys():
            for lane in self.road_lane_mapping[key]:
                self.lanes.append(lane)

    def _parse_all_traffic_light_info(self):
        """
        Pre-parse complete information for all traffic lights
        (lanelinks + road_lane_mapping + lanes).
    
        Called during __init__ to avoid parsing at runtime.
    
        Returns:
            {
                tl_id: {
                    'lanelinks': [...],
                    'road_lane_mapping': {...},
                    'roads': [...],
                    'outs': [...],
                    'directions': [...],
                    'lanes': [...],
                    'in_roads': [...],
                    'out_roads': [...]
                }
            }
        """
        # First get all lanelinks
        all_lanelinks = self._parse_all_lanelinks()
        
        result = {}
        
        for tl_id in self.traffic_light_ids:
            lanelinks = all_lanelinks.get(tl_id, [])
            
            # Initialize data structures
            road_lane_mapping = {}
            roads = []
            outs = []
            directions = []
            
            # Process each link (replicates original logic)
            for link_list in lanelinks:
                if not link_list:
                    continue
                
                link = link_list[0]  # take the first connection
                from_lane = link[0]
                to_lane = link[1]
                
                # Extract road IDs
                from_road = from_lane[:-2]
                to_road = to_lane[:-2]
                
                # Handle from_lane (incoming road)
                if from_road not in road_lane_mapping:
                    road_lane_mapping[from_road] = []
                    roads.append(from_road)
                    outs.append(False)
                    
                    # Get lane shape and compute direction
                    road_shape = self.get_lane_shape_from_net(from_lane)
                    directions.append(self._get_direction(road_shape, False))
                
                if from_lane not in road_lane_mapping[from_road]:
                    road_lane_mapping[from_road].append(from_lane)
                
                # Handle to_lane (outgoing road)
                if to_road not in road_lane_mapping:
                    road_lane_mapping[to_road] = []
                    roads.append(to_road)
                    outs.append(True)
                    
                    # Get lane shape and compute direction
                    road_shape = self.get_lane_shape_from_net(to_lane)
                    directions.append(self._get_direction(road_shape, True))
                
                if to_lane not in road_lane_mapping[to_road]:
                    road_lane_mapping[to_road].append(to_lane)
            
            # Sort roads
            if roads:
                order = sorted(range(len(roads)),
                            key=lambda i: (directions[i],
                                            outs[i] if self.RIGHT else not outs[i]))
                
                sorted_roads = [roads[i] for i in order]
                sorted_outs = [outs[i] for i in order]
                sorted_directions = [directions[i] for i in order]
            else:
                sorted_roads = []
                sorted_outs = []
                sorted_directions = []
            
            # Build ordered lane list and in/out roads
            lanes = []
            in_roads = []
            out_roads = []
            
            for idx, road in enumerate(sorted_roads):
                if road in road_lane_mapping:
                    for lane in road_lane_mapping[road]:
                        lanes.append(lane)
                
                if sorted_outs[idx]:
                    out_roads.append(road)
                else:
                    in_roads.append(road)
            
            # Store full information
            result[tl_id] = {
                'lanelinks': lanelinks,
                'road_lane_mapping': road_lane_mapping,
                'roads': sorted_roads, # in_only==False
                'outs': sorted_outs,
                'directions': sorted_directions,
                'lanes': lanes,
                'in_roads': in_roads, # in_only==True
                'out_roads': out_roads,
                # 🔑 NEW: pre-store both observation modes
                'lanes_road_observed': self.build_lanes_road_observed(sorted_roads, road_lane_mapping, self.RIGHT),
                'lanes_road_observed_in_only': self.build_lanes_road_observed(in_roads, road_lane_mapping, self.RIGHT),
            }
        
        return result
    
    def get_observation_space_static(self, tl_id: str, obs_to_subscribe: list, in_only: bool = True):
        """
        Compute the observation space statically (without instantiating an Intersection).
    
        Args:
            tl_id: Traffic light ID
            obs_to_subscribe: List of observation features (e.g. ['lane_waiting_count'])
            in_only: Whether to observe only incoming lanes
    
        Returns:
            gym.spaces.Box: Observation space
        """
        if not hasattr(self, 'traffic_light_info'):
            raise ValueError("traffic_light_info not initialized.")
        
        if tl_id not in self.traffic_light_info:
            raise ValueError(f"Traffic light {tl_id} not found.")
        
        tl_info = self.traffic_light_info[tl_id]
        
        use_presslight = 'presslight' in obs_to_subscribe
        in_only = False if use_presslight else in_only
        # 🔑 Directly read from pre-stored data
        if in_only:
            lanes_road_observed = tl_info['lanes_road_observed_in_only']
        else:
            lanes_road_observed = tl_info['lanes_road_observed']
        
        # ✅ Compute observation dimensions
        num_phases = len(self.green_phases[tl_id])  # number of phases
        phase_onehot_dim = num_phases                # phase one-hot encoding
        min_green_dim = 1                            # minimum green time flag
        # ========== NEW: special handling for project1 mode ==========
        algorithm_name = getattr(self, '_algorithm_name', '')
        if 'project1' in algorithm_name.lower() or 'std_dqn' in algorithm_name.lower():
            # Dynamic format: phase(N_phases) + N_in_lanes × 5 features
            # Number of lanes is determined by the network, not hardcoded as 12
            num_phases   = len(self.green_phases[tl_id])
            num_in_lanes = sum(len(lanes) for lanes in tl_info['lanes_road_observed_in_only'])
            ob_length    = num_phases + num_in_lanes * 5

            return gym.spaces.Box(
                low=np.zeros(ob_length, dtype=np.float32),
                high=np.ones(ob_length, dtype=np.float32),
                dtype=np.float32
            )
        
        if 'final_year_project_lane_mode' in algorithm_name.lower() or 'fyp_lane' in algorithm_name.lower():
            # Dynamic format: state space changes depending what is used please EDIT ob_length with the TODO:
            # Number of lanes is determined by the network, not hardcoded
            num_phases   = len(self.green_phases[tl_id])
            num_in_lanes = sum(len(lanes) for lanes in tl_info['lanes_road_observed_in_only'])
            ob_length    = num_phases + num_in_lanes * 7 # will need to adjust to match observation space length

            return gym.spaces.Box(
                low=np.zeros(ob_length, dtype=np.float32),
                high=np.ones(ob_length, dtype=np.float32),
                dtype=np.float32
            )
        
        if 'final_year_project' in algorithm_name.lower() or 'fyp' in algorithm_name.lower():
            # Dynamic format: intersection-level and per-lane feature counts are
            # read from FYP_OBS_CONFIG in Observations.py, so toggling features
            # there automatically resizes this observation space - no manual
            # editing needed here any more.
            # Number of lanes is determined by the network, not hardcoded
            num_phases   = len(self.green_phases[tl_id])
            num_in_lanes = sum(len(lanes) for lanes in tl_info['lanes_road_observed_in_only'])
            inter_dim, lane_dim = get_fyp_observation_dims()
            ob_length    = inter_dim + num_phases + num_in_lanes * lane_dim # TODO: this should be automatic now - just change Observations.py
        
            return gym.spaces.Box(
                low=np.zeros(ob_length, dtype=np.float32),
                high=np.ones(ob_length, dtype=np.float32),
                dtype=np.float32
            )

        if use_presslight:
            # PressLight mode: incoming lanes × 3 + outgoing lanes × 1
            num_in_lanes = sum(len(x) for x in tl_info['lanes_road_observed_in_only'])
            
            # Compute number of outgoing lanes
            out_roads = tl_info['out_roads']
            road_lane_mapping = tl_info['road_lane_mapping']
            num_out_lanes = sum(len(road_lane_mapping.get(road, [])) for road in out_roads)
            
            lane_features_dim = num_in_lanes * 3 + num_out_lanes
        else:
            # Compute observation dimension (mimics Observation.observation_space() logic)
            num_lanes = sum(len(x) for x in lanes_road_observed)  # total number of lanes
            lane_features_dim = len(obs_to_subscribe)*num_lanes
        # Total dimension
        ob_length = phase_onehot_dim+min_green_dim+lane_features_dim 

        return gym.spaces.Box(
            low=np.zeros(ob_length, dtype=np.float32),
            high=np.ones(ob_length, dtype=np.float32),
            dtype=np.float32
        )
    
    # Precompute lanes_road_observed (two modes: in_only=True/False)
    def build_lanes_road_observed(self, roads_list, road_lane_mapping, right_traffic):
        lanes_road_observed = []
        for r in roads_list:
            if r in road_lane_mapping:
                if not right_traffic:
                    tmp = sorted(road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
                else:
                    tmp = sorted(road_lane_mapping[r], key=lambda ob: int(ob[-1]))
                lanes_road_observed.append(tmp)
        return lanes_road_observed
    def get_obs_to_subscribe(self):
        """Get observation subscription configuration; return default if not set"""
        return self._obs_to_subscribe
    
    def get_reward_to_subscribe(self):
        """Get reward subscription configuration; return default if not set"""
        return self._reward_to_subscribe
    
    # =============================================Road Disruption Functions==========================================
    def get_route_file_path(self):
        """Get the path to the route file"""
        if not self.sumo_dict.get('combined_file'):
            # Directly use flowFile
            return os.path.join(self.sumo_dict['dir'], self.sumo_dict['flowFile'])
        else:
            # Parse from combined_file (.cfg)
            cfg_path = os.path.join(self.sumo_dict['dir'], self.sumo_dict['combined_file'])
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(cfg_path)
                root = tree.getroot()
                
                # Find route-files tag
                for input_tag in root.findall('.//route-files'):
                    route_file = input_tag.get('value')
                    if route_file:
                        # Path may be relative to cfg file
                        if not os.path.isabs(route_file):
                            route_file = os.path.join(self.sumo_dict['dir'], route_file)
                        return route_file
            except Exception as e:
                print(f"⚠ Failed to parse route file from cfg: {e}")
        
        return None

    def parse_od_routes_from_file(self):
        """
        Parse all OD pairs and their routes from rou.xml file
        
        Returns:
            dict: {(origin, destination): [route1, route2, ...]}
                  where each route is a list of edge IDs
        """
        route_file = self.get_route_file_path()
        
        if not route_file or not os.path.exists(route_file):
            print(f"⚠ Route file does not exist: {route_file}")
            return {}
        
        print(f"📄 Parsing route file: {route_file}")
        
        od_routes = {}  # {(origin, dest): [route1, route2, ...]}
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            # Parse <route> tags (can be standalone or inside vehicle/flow)
            route_definitions = {}  # {route_id: edge_list}
            
            # 1. Parse standalone <route> definitions
            for route_tag in root.findall('route'):
                route_id = route_tag.get('id')
                edges_str = route_tag.get('edges', '')
                if route_id and edges_str:
                    edges = edges_str.strip().split()
                    route_definitions[route_id] = edges
            
            # 2. Parse routes in <vehicle> and <flow>
            for element in root.findall('vehicle') + root.findall('flow'):
                # Case 1: referenced via route attribute
                route_ref = element.get('route')
                if route_ref and route_ref in route_definitions:
                    edges = route_definitions[route_ref]
                else:
                    # Case 2: nested <route> tag
                    route_tag = element.find('route')
                    if route_tag is not None:
                        edges_str = route_tag.get('edges', '')
                        edges = edges_str.strip().split() if edges_str else []
                    else:
                        # Case 3: from/to attributes (requires SUMO path computation, skip for now)
                        continue
                
                if len(edges) >= 2:
                    origin = edges[0]
                    destination = edges[-1]
                    od_pair = (origin, destination)
                    
                    if od_pair not in od_routes:
                        od_routes[od_pair] = []
                    
                    # Avoid duplicate routes
                    if edges not in od_routes[od_pair]:
                        od_routes[od_pair].append(edges)
            
            # Statistics
            total_od_pairs = len(od_routes)
            total_routes = sum(len(routes) for routes in od_routes.values())
            od_with_multiple_routes = sum(1 for routes in od_routes.values() if len(routes) >= 2)
            
            print(f"✓ Parsed {total_od_pairs} OD pairs from rou.xml (for connectivity testing)")
            
            return od_routes
            
        except Exception as e:
            print(f"⚠ Error parsing route file: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_all_depart_edges_from_rou(self):
        """
        Parse all departure edges from rou.xml file
        
        Returns:
            set: set of all departure edges
        """
        route_file = self.get_route_file_path()
        
        if not route_file or not os.path.exists(route_file):
            print(f"⚠ Route file does not exist: {route_file}")
            return set()
        
        depart_edges = set()
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            # 1. Parse standalone <route> definitions
            route_definitions = {}
            for route_tag in root.findall('route'):
                route_id = route_tag.get('id')
                edges_str = route_tag.get('edges', '')
                if route_id and edges_str:
                    edges = edges_str.strip().split()
                    if edges:
                        route_definitions[route_id] = edges[0]  # store only the first edge
            
            # 2. Parse <vehicle> and <flow>
            for element in root.findall('vehicle') + root.findall('flow'):
                # Case 1: referenced via route attribute
                route_ref = element.get('route')
                if route_ref and route_ref in route_definitions:
                    depart_edges.add(route_definitions[route_ref])
                else:
                    # Case 2: nested <route> tag
                    route_tag = element.find('route')
                    if route_tag is not None:
                        edges_str = route_tag.get('edges', '')
                        edges = edges_str.strip().split() if edges_str else []
                        if edges:
                            depart_edges.add(edges[0])
                    else:
                        # Case 3: from attribute (departure edge)
                        from_edge = element.get('from')
                        if from_edge:
                            depart_edges.add(from_edge)
        
            print(f"📄 Parsed {len(depart_edges)} unique departure edges from rou.xml")
            return depart_edges
            
        except Exception as e:
            print(f"⚠ Error parsing departure edges: {e}")
            return set()
    def get_closable_edges(self):
        """
        Identify edges that can be closed based on connectivity testing
        
        Idea:
        1. For each OD pair, test whether a path still exists after removing each edge
        2. If still connected → edge is closable
        3. If disconnected → edge is critical (not closable)
        """
        import sumolib
        import networkx as nx
        
        # 1. Build road network graph
        print("📊 Building road network topology...")
        net = sumolib.net.readNet(self.get_net_file_address())
        
        G = nx.DiGraph()
        for edge in net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            G.add_edge(from_node, to_node, edge_id=edge.getID())
        
        print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        # 2. Get OD pairs from rou.xml
        od_routes = self.parse_od_routes_from_file()
        
        if not od_routes:
            print("⚠ Failed to parse OD routes")
            return []
        
        # Convert to node pairs (edge → node)
        od_node_pairs = set()
        for (origin_edge, dest_edge), routes in od_routes.items():
            try:
                origin_node = net.getEdge(origin_edge).getToNode().getID()
                dest_node = net.getEdge(dest_edge).getFromNode().getID()
                od_node_pairs.add((origin_node, dest_node))
            except:
                pass
        
        print(f"📊 Analyzing {len(od_node_pairs)} OD pairs for destroyable edges...")
        
        # 3. Test each edge
        destroyable_edges = set()
        critical_edges = set()
        
        edge_id_to_nodes = {}
        for edge in net.getEdges():
            edge_id = edge.getID()
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            edge_id_to_nodes[edge_id] = (from_node, to_node)
        
        total_edges = len(edge_id_to_nodes)
        
        for idx, (edge_id, (from_node, to_node)) in enumerate(edge_id_to_nodes.items()):
            if (idx + 1) % 20 == 0:
                print(f"  进度: {idx+1}/{total_edges}")
            
            is_critical = False
            
            # Test impact of removing this edge for each OD pair
            for source, target in od_node_pairs:
                # ✅ Core logic: simulate removal
                G_temp = G.copy()
                if G_temp.has_edge(from_node, to_node):
                    G_temp.remove_edge(from_node, to_node)
                
                # ✅ Check connectivity
                if not nx.has_path(G_temp, source, target):
                    # Disconnected → critical edge
                    is_critical = True
                    critical_edges.add(edge_id)
                    break
            
            if not is_critical:
                destroyable_edges.add(edge_id)
        
        # 4. Summary
        print(f"\n✅ Analysis complete:")
        print(f"  - Total edges: {total_edges}")
        print(f"  - Critical edges (not closable): {len(critical_edges)}")
        print(f"  - Closable edges: {len(destroyable_edges)}")
        
        return list(destroyable_edges)
    
# ===============================================================================================


class World(parse_sumo_config, gym.Env):
    '''
    World Class is mainly used for creating a SUMO engine and maintain information about SUMO world.
    '''

    def __init__(self, sumo_config, **kwargs):
        super().__init__(sumo_config, **kwargs)
        # ✨ New: control for synchronous/asynchronous decision mode
        self.sync_mode = kwargs.get('sync_mode', False)  # default is asynchronous mode
        # ✅ Add these 3 lines: get reward configuration (used for special algorithms like MA2C)
        self._reward_weights = kwargs.get('reward_weights', [1.0])
        self._reward_scale = kwargs.get('reward_scale', 1.0)
        self._reward_clip_range = kwargs.get('reward_clip_range', None)
        self.sumo_cmd = self.generate_sumo_cmd()
        self.warning = self.no_warning()
        self.connection_name = self.get_connection_name() # default: debug
        self.map_name = self.get_map_name()
        self.net = self.get_net_file_address()
        self.RIGHT = True
        
        # self.step_ratio = 1  # old todo was here -  register in Registry later
        # self.step_length = 1  # should be 1 in our setting, how much duration time in sim per step
        # self.sim_max_steps = 1000
        self.sim_max_steps = kwargs.get('sim_max_steps', 1000)
        self.step_counter = 0
        self.max_distance = 200 #应该是intersection level
        
        # if kwargs['interface'] == 'libsumo':
        #     self.interface_flag = True
        # elif kwargs['interface'] == 'traci':
        #     self.interface_flag = False
        # else:
        #     raise Exception('NOT IMPORTED YET')
        # ============Pre-fetch traffic lights and phase information (no need to start SUMO)============
        self.traffic_light_ids = self._get_traffic_light_ids()
        self.all_roads = self._get_roads()
        self.all_lanes = self._get_lanes()
        
        # Key change: generate green_phases in __init__ (parse XML only, no SUMO start)
        self.green_phases = self._generate_valid_phase()        
    
        # ==================world level dynamic statistics/ ==================
        self.num_arrived_vehicles = 0 # total number of vehicles that have arrived in the world
        self.num_departed_vehicles = 0 # total number of vehicles that have departed in the world   
        self.num_teleported_vehicles = 0 # total number of vehicles that have teleported in the world
       # Distinguish between world-level and intersection-level metrics       
        self.total_info_labels = [
            "time",
            "system_total_running_num",
            "system_total_backlogged_num",
            "system_total_stopped",
            "system_total_arrived",
            "system_total_departed",
            "system_total_teleported",
            "system_total_waiting_time",
            "system_mean_waiting_time",
            "system_mean_speed",
            "pressure", # get pressure at each intersection (incoming - outgoing vehicle count) (get_pressure method)
            "phase", # get current signal phase at each intersection (get_cur_phase method)
            "lane_count", # get vehicle count per lane (get_lane_vehicle_count method)
            "lane_vehicles", # get list of vehicles per lane (get_lane_vehicles method)
            "lane_waiting_count", # get number of waiting vehicles per lane (get_lane_waiting_vehicle_count method)
            "lane_pressure", # get pressure per incoming lane (incoming - corresponding outgoing vehicles) (get_lane_pressure method)
            "lane_waiting_time_count", # get total waiting time of all waiting vehicles per lane (get_lane_waiting_time_count method)
            "lane_delay", # compute average delay per lane (1 - avg_speed / speed_limit) (get_lane_delay method)
            "real_delay", # get real average delay for all vehicles (get_real_delay method), based on trajectory vs expected time
            "vehicle_trajectory", # get trajectory of all vehicles (lane changes and timestamps) (get_vehicle_trajectory method)
            "vehicles_average_trip_time",
            "outgoing_lane_vehicles",
        ] # get average trip time of all completed vehicles (get_vehicles method)
            
        self.fns_subscribed = []
        self.info_dynamics_real_time = {}

        # ==================every vehicle's dynamic statistics/individual level=================
        self.vehicles_entering_time = dict()
        self.vehicles_trip_time = dict() # vehicle_id: time_in_simulation
        self.ev_lane_entry_time = {}  # {veh_id: (lane_id, entry_time)} - NEW FROM FYP
        self.vehicle_last_timeloss = {}   # {veh_id: last known getTimeLoss() reading} - NEW: delay metric
        self.vehicle_timeloss_delta = {}  # {veh_id: timeLoss accrued THIS step} - NEW: delay metric
        
        self.vehicle_group_delay_totals = {}   # regular vehicles: {veh_id: {'group1': float, 'group2': float}}
        self.vehicle_ever_group1 = set()       # veh_ids ever classified group1 at any point in their trip
        self.vehicle_ev_delay_totals = {}      # emergency vehicles: {veh_id: float}  (no group split needed)
        
        self.finished_reg_group1_delays = []   # one number per vehicle: its total group1-attributed delay
        self.finished_reg_group2_delays = []   # one number per vehicle: its total group2-attributed delay
        self.finished_reg_all_delays = []      # one number per vehicle: total delay (group1+group2)
        self.finished_ev_delays = []           # one number per EV: total delay over its whole trip
        
        # # test generate observation information
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay = {}
        # =============================================
        # if self.interface_flag:
        #     libsumo.start(self.sumo_cmd) 
        #     self.eng = libsumo
        # else:
            
        #     traci.start(self.sumo_cmd, label=self.connection_name)
        #     self.eng = traci.getConnection(self.connection_name)

        
        # ==================static information in the world=================
        # self.traffic_light_ids = self._get_traffic_light_ids()
        # self.all_roads = self._get_roads()
        # self.all_lanes = self._get_lanes()
        # These will be initialized during the first reset
        # self.green_phases = None
        self.id2intersection = None
        self.intersections = None
        self.in_lanes = None
        self.out_lanes = None
        self.action_space = None
        
        # self.eng.close()
        # ==basic constant information have been prepared, then launch the simulation==
        # ======================================
        print('Connection ID', self.connection_name)
        # create logger folder
        # if not os.path.exists(os.path.join(Registry.mapping['logger_mapping']['path'].path,
        #                                    self.connection_name)):
        #     os.mkdir(os.path.join(Registry.mapping['logger_mapping']['path'].path, self.connection_name))
        self.observations = {tl: None for tl in self.traffic_light_ids}
        self.rewards = {tl: None for tl in self.traffic_light_ids}
        
        # Create combined action space (if needed later)
        # action_dims = [intersection._action_space.n for intersection in self.intersections]
        # self.action_space = gym.spaces.MultiDiscrete(action_dims)
        
        
    
    # ======================================Non-dynamic information functions==========================================
    def get_in_out_lanes(self):
        in_lanes = []
        out_lanes = []
        for i in self.intersections:
            for road in i.in_roads:
                for lane in i.road_lane_mapping[road]:
                    in_lanes.append(lane)
            for road in i.out_roads:
                for lane in i.road_lane_mapping[road]:
                    out_lanes.append(lane)
        # add in_lanes of virtual intersections which can be regarded as out_lanes of non-virtual intersections.
        for lane in self.all_lanes:
            if lane not in out_lanes:
                out_lanes.append(lane)
        return in_lanes, out_lanes
    # ==========================================statistics functions (world level)==========================================
    
    def get_current_time(self):
        '''
        get_current_time
        Get simulation time (in seconds).
        
        :param: None
        :return result: current time
        '''
        
        return self.eng.simulation.getTime()
    
    def _get_system_info(self):
        world_vehicles_num = self.eng.vehicle.getIDList() # all vehicles in the world
        all_vehicle_speed = [self.eng.vehicle.getSpeed(vehicle) for vehicle in world_vehicles_num]
        all_vehicle_waiting_times = [self.eng.vehicle.getWaitingTime(vehicle) for vehicle in world_vehicles_num]
        num_backlogged_vehicles = len(self.eng.simulation.getPendingVehicles()) # all vehicles that are waiting to depart
        return {
            "system_total_running_num": world_vehicles_num,
            "system_total_backlogged_num": num_backlogged_vehicles,
            "system_total_stopped": sum(
                int(speed < 0.1) for speed in all_vehicle_speed
            ),  # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_arrived": self.num_arrived_vehicles,
            "system_total_departed": self.num_departed_vehicles,
            "system_total_teleported": self.num_teleported_vehicles,
            "system_total_waiting_time": sum(all_vehicle_waiting_times),
            "system_mean_waiting_time": 0.0 if len(world_vehicles_num) == 0 else np.mean(all_vehicle_waiting_times),
            "system_mean_speed": 0.0 if len(world_vehicles_num) == 0 else np.mean(all_vehicle_speed),
        } 
    
    # ==========================================statistics functions (intersection level)==========================================
    def get_pressure(self):
        '''
        get_pressure
        Get pressure of each intersection. 
        Pressure of an intersection equals to number of vehicles that in in_lanes minus number of vehicles that in out_lanes.
        
        Get pressure of each intersection following PressLight paper Definition 3.4.
        P_i = |Σ w(l,m)| where w(l,m) = x(l)/x_max(l) - x(m)/x_max(m)
    
        :return: pressures dict {intersection_id: pressure_value}
        '''
        pressures = dict()
        lane_vehicles = self.get_lane_vehicle_count()
        VEHICLE_LENGTH = 5.0
        for i in self.intersections:
            total_pressure = 0
            for road in i.in_roads:
                for k in i.road_lane_mapping[road]:
                    lane_length = self.eng.lane.getLength(k)
                    capacity = lane_length/VEHICLE_LENGTH
                    vehicle_density = lane_vehicles[k]/ capacity
                    total_pressure += vehicle_density
            for road in i.out_roads:
                for k in i.road_lane_mapping[road]:
                    lane_length = self.eng.lane.getLength(k)
                    capacity = lane_length/VEHICLE_LENGTH
                    vehicle_density = lane_vehicles[k]/ capacity
                    total_pressure -= vehicle_density
            pressures[i.id] = abs(total_pressure)
        return pressures
    
    def get_cur_phase(self):
        '''
        get_cur_phase
        Get current phase of each intersection.

        :param: None
        :return result: current phase of each intersection
        '''
        result = []
        for intsec in self.intersections:
            result.append(intsec.get_current_phase())
        return result
    # ==========================================statistics functions (lane level)==========================================
    def get_lane_vehicle_count(self):
        '''
        get_lane_vehicle_count
        Get number of vehicles in each lane.
        
        :param: None
        :return result: number of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_count']})
        return result
    
    def get_lane_vehicles(self):
        '''
        get_lane_vehicles
        Get vehicles' id of each lane.

        :param: None
        :return vehicle_lane: vehicles' id of each lane
        '''
        result = dict()
        for inter in self.intersections:
            for key in inter.full_observation.keys():
                result.update({key: inter.full_observation[key]})
        return result
    
    def get_lane_waiting_vehicle_count(self):
        '''
        get_lane_waiting_vehicle_count
        Get number of waiting vehicles in each lane.
        
        :param: None
        :return result: number of waiting vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: np.float32(intsec.full_observation[lane]['lane_waiting_count'])})
        return result
    
    def get_lane_pressure(self):
        '''
        get_lane_pressure
        Get pressure of each lane in an intersection. 
        Pressure of each lane equals to number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        
        :param: None
        :return pressures: pressure of each lane
        '''
        lvc = self.get_lane_vehicle_count()
        pressures = {}
        pressures = {x:0 for x in self.in_lanes}
        for inter_obj in self.intersections:
            for lanelink in inter_obj.lanelinks:
                start, end = lanelink[0][0], lanelink[0][1]
                pressures[start] += lvc[start]
                pressures[start] -= lvc[end]
        return pressures
    
    def get_lane_waiting_time_count(self):
        '''
        get_lane_waiting_time_count
        Get waiting time of vehicles in each lane.
        
        :param: None
        :return result: waiting time of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            for lane in intsec.lanes:
                result.update({lane: intsec.full_observation[lane]['lane_waiting_time_count']})
        return result
    
    def get_lane_delay(self):
        '''
        get_lane_delay
        Get approximate delay of each lane. 
        Approximate delay of each lane equals to (1 - lane_avg_speed)/lane_speed_limit.
        
        :param: None
        :return lane_delay: approximate delay of each lane
        '''
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        lane_vehicles = self.get_lane_vehicles()
        lane_delay = dict()
        for key in lane_vehicles.keys():
            vehicles = lane_vehicles[key]['vehicles']
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            speed_limit = self.eng.lane.getMaxSpeed(key)
            for vehicle in vehicles:
                speed = vehicle['speed']
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[key] = 1 - lane_avg_speed / speed_limit
        return lane_delay
    # ==========================================statistics functions (individual level)==========================================
    def get_vehicle_lane(self):
        '''
        get_vehicle_lane
        Get current lane id and max speed of each vehicle that is running.

        :param: None
        :return vehicle_lane: current lane id of each vehicle
        :return vehicle_maxspeed: max speed of each vehicle
        '''
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        for lane in self.all_lanes:
            vehicles = 	self.eng.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                vehicle_lane[vehicle] = lane
                self.vehicle_maxspeed[(vehicle,lane)] = self.eng.vehicle.getAllowedSpeed(vehicle)
        return vehicle_lane, self.vehicle_maxspeed
    
    def get_vehicles_average_trip_time(self):
        '''
        Get all vehicles' average trip time.
        :param: None
        :return: average trip time
        '''
        trip_time_result = 0
        count = 0
        for v in self.vehicles_trip_time.keys():
            count += 1
            trip_time_result += self.vehicles_trip_time[v]
        if count == 0:
            return 0
        else:
            return trip_time_result/count    
    
    def get_vehicle_trajectory(self):
        '''
        get_vehicle_trajectory
        Get trajectory of vehicles that have entered in roadnet, including vehicle_id, enter time, leave time or current time.
        the detailed route information includes lane_id, enter time, duration.
        :param: None
        :return vehicle_trajectory: trajectory of vehicles that have entered in roadnet
        :return vehicle_maxspeed: max speed of each vehicle that have entered in roadnet
        '''
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane, self.vehicle_maxspeed = self.get_vehicle_lane() # get vehicles on the roads except turning
        vehicles = list(self.eng.vehicle.getIDList())
        # vehicles = [x for x in vehicle_lane]
        for vehicle in vehicles:
            # 检查车辆是否在 vehicle_lane 中（避免转弯或在交叉口内的车辆）
            if vehicle not in vehicle_lane:
                continue
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[vehicle_lane[vehicle], int(self.eng.simulation.getTime()), 0]]
            else:
                if vehicle not in vehicle_lane.keys(): # vehicle is turning
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][0]: # vehicle is running on the same lane 
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else: # vehicle has changed the lane
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.simulation.getTime()), 0])
        return self.vehicle_trajectory, self.vehicle_maxspeed

    def get_real_delay(self):
        '''
        get_real_delay
        Calculate average real delay. 
        Real delay of a vehicle is defined as the time a vehicle has traveled within the environment minus the expected travel time.
        
        :param: None
        :return avg_delay: average real delay of all vehicles
        '''
        # Only update trajectory when needed
        if 'vehicle_trajectory' not in self.fns_subscribed:
            self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()
        for v in self.vehicle_trajectory:
            # get road level routes of vehicle
            routes = self.vehicle_trajectory[v] # lane_level
            for idx, lane in enumerate(routes):
                speed = min(self.eng.lane.getMaxSpeed(lane[0]), self.vehicle_maxspeed[(v,lane[0])])
                lane_length = self.eng.lane.getLength(lane[0])
                if idx == len(routes)-1: # the last lane
                    # judge whether the vehicle run over the whole lane.
                    lane_length = self.eng.vehicle.getLanePosition(v) if v in self.eng.vehicle.getIDList() else lane_length
                planned_tt = float(lane_length)/speed
                real_delay = lane[-1] - planned_tt if lane[-1]>planned_tt else 0.
                if v not in self.real_delay.keys():
                    self.real_delay[v] = real_delay
                else:
                    self.real_delay[v] += real_delay

        avg_delay = 0.
        count = 0
        for dic in self.real_delay.items():
            avg_delay += dic[1]
            count += 1
        avg_delay = avg_delay / count
        return avg_delay
    
    def get_outgoing_lane_vehicles(self):
        """
        Get vehicle information on outgoing lanes (used for Project 1 attention mechanism)
        
        Returns:
            outgoing_vehicles: dict
                {
                    'lane_id': {
                        'total_count': int,
                        'near_junction_count': int,  # vehicles more than 100m from junction end (upstream congestion)
                        'vehicles': [veh_id, ...]
                    },
                    ...
                }
        """
        outgoing_vehicles = {}
        
        # iterate through all traffic lights' outgoing roads
        for ts in self.intersections:
            for out_road in ts.out_roads:
                lanes = ts.road_lane_mapping.get(out_road, [])
                
                for lane_id in lanes:
                    try:
                        vehicle_ids = self.eng.lane.getLastStepVehicleIDs(lane_id)
                        lane_length = self.eng.lane.getLength(lane_id)
                        
                        near_junction_count = 0
                        
                        # count vehicles that are ≥100m from junction end (congestion region upstream)
                        for veh_id in vehicle_ids:
                            try:
                                position = self.eng.vehicle.getLanePosition(veh_id)
                                distance_to_end = lane_length - position
                                
                                # Project 1 logic: vehicles ≥100m from end are considered congested
                                if distance_to_end >= 100:
                                    near_junction_count += 1
                            except:
                                continue
                        
                        outgoing_vehicles[lane_id] = {
                            'total_count': len(vehicle_ids),
                            'near_junction_count': near_junction_count,
                            'vehicles': list(vehicle_ids)
                        }
                    
                    except Exception as e:
                        # return empty data if error occurs
                        outgoing_vehicles[lane_id] = {
                            'total_count': 0,
                            'near_junction_count': 0,
                            'vehicles': []
                        }
        
        return outgoing_vehicles
    # ====================================================================================

    def _update_infos(self):
        # old todo was here -  add normalization value in the info_functions
        self.info_dynamics_real_time = {}
        self.info_functions = {
            "time": self.get_current_time,
            "system_total_running_num": self._get_system_info,
            "system_total_backlogged_num": self._get_system_info,
            "system_total_stopped": self._get_system_info,
            "system_total_arrived": self._get_system_info,
            "system_total_departed": self._get_system_info,
            "system_total_teleported": self._get_system_info,
            "system_total_waiting_time": self._get_system_info,
            "system_mean_waiting_time": self._get_system_info,
            "system_mean_speed": self._get_system_info,
            "pressure": self.get_pressure, # intersection level
            "phase": self.get_cur_phase,
            "lane_count": self.get_lane_vehicle_count,
            "lane_vehicles": self.get_lane_vehicles,
            "lane_waiting_count": self.get_lane_waiting_vehicle_count,
            "lane_pressure": self.get_lane_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "real_delay": self.get_real_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "vehicles_average_trip_time": self.get_vehicles_average_trip_time,
            "outgoing_lane_vehicles": self.get_outgoing_lane_vehicles,
        }
        for fn in self.fns_subscribed:
            if "system" in fn:
                self.info_dynamics_real_time[fn] = self.info_functions[fn][fn]
            else:
                self.info_dynamics_real_time[fn] = self.info_functions[fn]()            

    def subscribe(self, fns):
        '''
        subscribe
        Subscribe information you want to get when training the model.
        
        :param fns: information name you want to get
        :return: None
        '''
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.total_info_labels:
                if fn not in self.fns_subscribed:
                    self.fns_subscribed.append(fn)
            else:
                raise Exception(f'Info function {fn} not implemented')

    # =====================================
    def step_sim_and_statistics(self): # actually is statistic first and then step sim
        """
        Collect step-level statistics on world level, intersection level, and vehicle level.
        """
        # follow the warmup
        # CONCLUDE each step statistics on world level
        for intsec in self.intersections:
            intsec.collect_objective_traffic_state(self.max_distance) # repeated call compare to reset
            # ✅ 累积每一步的奖励
            intsec.accumulate_reward()
            
        # register vehicles here
        entering_v = self.eng.simulation.getDepartedIDList()
        exiting_v = self.eng.simulation.getArrivedIDList()
        for v in entering_v:
            self.vehicles_entering_time.update({v: self.get_current_time()})
                
        for v in exiting_v:
            if v in self.vehicles_entering_time:
                self.vehicles_trip_time.update({v: self.get_current_time() - self.vehicles_entering_time[v]})

            # NEW: finalize this vehicle's accumulated group delay into the finished lists
            if v in self.vehicle_group_delay_totals:
                totals = self.vehicle_group_delay_totals.pop(v)
                if totals['group1'] > 0:
                    self.finished_reg_group1_delays.append(totals['group1'])
                if totals['group2'] > 0:
                    self.finished_reg_group2_delays.append(totals['group2'])
                self.finished_reg_all_delays.append(totals['group1'] + totals['group2'])
                self.vehicle_ever_group1.discard(v)
            if v in self.vehicle_ev_delay_totals:
                self.finished_ev_delays.append(self.vehicle_ev_delay_totals.pop(v))

        # NEW: per-step timeLoss delta, computed ONCE here for every vehicle.
        # getTimeLoss() is cumulative since departure, not a per-step rate, so we
        # track each vehicle's last-seen reading and take the difference. This is
        # centralized in World (rather than inside Rewards.get_reward_statistics)
        # so that multiple intersections reading the same vehicle in the same step
        # don't each try to "consume" the delta and double-count / zero it out.
        for veh_id in self.eng.vehicle.getIDList():
            try:
                current_timeloss = self.eng.vehicle.getTimeLoss(veh_id)
                if veh_id in self.vehicle_last_timeloss:
                    delta = current_timeloss - self.vehicle_last_timeloss[veh_id]
                else:
                    delta = 0.0
                delta = max(delta, 0.0)
                # ACCUMULATE across sub-steps within a decision interval — do NOT overwrite.
                # Reset happens once per decision, in step_sim_until_time_to_act(), not here.
                self.vehicle_timeloss_delta[veh_id] = self.vehicle_timeloss_delta.get(veh_id, 0.0) + delta
                self.vehicle_last_timeloss[veh_id] = current_timeloss
            except Exception:
                continue

        # clean up last-seen baseline for vehicles that just left
        for veh_id in exiting_v:
            self.vehicle_last_timeloss.pop(veh_id, None)
            # NOTE: do NOT pop from vehicle_timeloss_delta here — a vehicle that
            # arrives mid-decision-window should still have its accrued delay
            # counted when get_reward_statistics() reads it. It clears naturally
            # at the next reset in step_sim_until_time_to_act().  
        
        # NEW FOR FYP
        # Track EV lane entry times for delay ratio computation
        current_time = self.get_current_time()
        for veh_id in self.eng.vehicle.getIDList():
            try:
                veh_type = self.eng.vehicle.getTypeID(veh_id)
                if veh_type not in ['ambulance_type', 'emergency']:
                    continue
                current_lane = self.eng.vehicle.getLaneID(veh_id)
                if not current_lane or current_lane.startswith(':'): # not sure why it does the 'or' bit here??
                    # Skip internal junction lanes
                    continue
                if veh_id not in self.ev_lane_entry_time:
                    self.ev_lane_entry_time[veh_id] = (current_lane, current_time)
                else:
                    stored_lane, entry_time = self.ev_lane_entry_time[veh_id]
                    if current_lane != stored_lane:
                        self.ev_lane_entry_time[veh_id] = (current_lane, current_time)
            except Exception:
                print("something went wrong with EV delay ratio computation in env.py - world - step_sim_and_statistics")
                continue
            
        # world system level statistics
        self.num_arrived_vehicles += self.eng.simulation.getArrivedNumber() # total number of vehicles that have arrived in the world
        self.num_departed_vehicles += self.eng.simulation.getDepartedNumber() # total number of vehicles that have departed in the world
        self.num_teleported_vehicles += self.eng.simulation.getEndingTeleportNumber() # total number of vehicles that have teleported in the world
        
        self._update_infos() # step level dynamic information (world level, intersection level, vehicle level)
        # self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()

    def step_sim_until_time_to_act(self):
        #print("DEBUG: step_sim_until_time_to_act v2")  # ← add this

        time_to_act = False

        # Reset the per-decision timeLoss accumulator ONCE per decision window
        # (not per raw sim-second) so it captures the FULL decision_interval's
        # worth of accrued delay by the time it's read via get_reward_statistics().
        self.vehicle_timeloss_delta = {}

        while not time_to_act:
            if self.step_counter >= self.sim_max_steps:
                break
            self.eng.simulationStep()
            self.step_counter += 1
            self.step_sim_and_statistics()
    
            # First pass: update ALL intersections' timers and yellow transitions
            for intersection in self.intersections:
                intersection.time_since_last_phase_change += 1
                if intersection.is_yellow and intersection.time_since_last_phase_change == intersection.yellow_phase_time:
                    #print(f"DEBUG YELLOW END: {intersection.id} at sim_time={self.get_current_time():.1f}, time_since_change={intersection.time_since_last_phase_change}")

                    self.eng.trafficlight.setRedYellowGreenState(
                        intersection.id, 
                        intersection.all_phases[intersection.green_phase].state)
                    intersection.is_yellow = False
                    intersection.time_since_last_phase_change = 0
    
            # Second pass: check if it's time to act (separate so break doesn't skip updates)
            if self.sync_mode:
                if all(intsec.time_to_act for intsec in self.intersections):
                    time_to_act = True
            else:
                if any(intsec.time_to_act for intsec in self.intersections):
                    time_to_act = True
        
    def reset(self):
        '''
        reset
        reset information, including vehicles, vehicle_trajectory, etc.
    
        :param: None
        :return: None
        '''
        
        if self.step_counter != 0:
            # old todo was here -  set trip info output
            self.close()
        # =============================================

        self.vehicles_trip_time = dict()
        self.ev_lane_entry_time = {} # {veh_id: (lane_id, entry_time)} - NEW FOR FYP
        self.vehicles_entering_time = dict()
        self.vehicle_last_timeloss = {}   # NEW: reset each episode
        self.vehicle_timeloss_delta = {}  # NEW: reset each episode
        # NEW: clear per-vehicle delay tracking each episode
        self.vehicle_group_delay_totals = {}
        self.vehicle_ever_group1 = set()
        self.vehicle_ev_delay_totals = {}
        self.finished_reg_group1_delays = []
        self.finished_reg_group2_delays = []
        self.finished_reg_all_delays = []
        self.finished_ev_delays = []
        # old todo was here - check when to close traci
        if self.interface_flag:
            libsumo.start(self.sumo_cmd)
            self.eng = libsumo
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.eng = traci.getConnection(self.connection_name)
        # ==================Initialize static information on first run=================
        # if self.green_phases is None:
        # self.green_phases = self._generate_valid_phase()
        
        # ==================warmup simulation=================
        # Run several steps to stabilize the environment
        for _ in range(300):
            self.eng.simulationStep()
        current_time = self.get_current_time()
        self.step_counter = int(current_time)
        print(f"DEBUG: Reset complete, current step_counter = {self.step_counter}")
        
        # =====================Create intersection objects==================================
        self.id2intersection = dict()
        self.intersections = []
        # Get observation and reward subscription configs from configuration
        obs_to_subscribe = self.get_obs_to_subscribe()
        reward_to_subscribe = self.get_reward_to_subscribe()
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(
                ts, 
                self, 
                obs_to_subscribe, 
                reward_to_subscribe,
                self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])
        self.id2idx = {i: idx for idx,i in enumerate(self.id2intersection)}
        
        # ============Initialize in/out lanes on first run=================
        if self.in_lanes is None:
            self.in_lanes, self.out_lanes = self.get_in_out_lanes()
        # =============Create action space=========================
        if self.action_space is None:
            action_dims = [intersection._action_space.n for intersection in self.intersections]
            self.action_space = gym.spaces.MultiDiscrete(action_dims)
        # Reset each intersection to initialize all necessary attributes        
        for intsec in self.intersections:
            intsec.reset()
            intsec.next_action_time = current_time
            intsec.collect_objective_traffic_state(self.max_distance)
            # Check if the current phase state is yellow
            phase_state = intsec.eng.trafficlight.getRedYellowGreenState(intsec.id)
            intsec.is_yellow = 'y' in phase_state
            if not intsec.is_yellow:
                try:
                    intsec.green_phase = next(
                        idx for idx, green_phase_obj in enumerate(intsec.green_phases) 
                        if green_phase_obj.state == phase_state)
                    intsec.time_since_last_phase_change = intsec.min_green
                except: # If no matching green phase is found, raise error (do not fallback)
                    raise ValueError(f"Surprise: phase state of {intsec.id} : '{phase_state}' is not found in green phases")
            else:
                intsec.time_since_last_phase_change = intsec.yellow_phase_time
        # Initialize variables without performing step-level statistics
        self._update_infos()
        # old todo was here - check if its the problem
        
        
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay= {}
        # old todo was here - compute initial observations
        return self._get_observations()
    
    def _get_observations(self):
        """Get observations for agents that need to act"""
        if self.sync_mode:
            # ✨ Synchronous mode: return observations for all agents
            observations = {
                tl: self.id2intersection[tl].get_observation()
                for tl in self.traffic_light_ids
            }
        else: # Asynchronous mode: only return agents that need to act
            # Get agents that need to act
            acting_agents = [
                tl for tl in self.traffic_light_ids 
                if self.id2intersection[tl].time_to_act
            ]
            # Only compute and return observations for these agents
            observations = {
                tl: self.id2intersection[tl].get_observation()
                for tl in acting_agents
            }
        
        # Update cache (optional)
        self.observations.update(observations)
        
        return observations
    
    def _get_rewards(self):
        """Get rewards for agents that need to act"""
        if self.sync_mode:
            # ✨ Synchronous mode: return rewards for all agents
            rewards = {
                tl: self.id2intersection[tl].get_reward()
                for tl in self.traffic_light_ids
            }
        else: # Asynchronous mode: only return agents that need to act
            # Get agents that need to act (consistent with observations)
            acting_agents = [
                tl for tl in self.traffic_light_ids 
                if self.id2intersection[tl].time_to_act
            ]
            # Only compute and return rewards for these agents
            rewards = {
                tl: self.id2intersection[tl].get_reward()
                for tl in acting_agents
            }
        
        # Update cache (optional)
        self.rewards.update(rewards)
        
        return rewards
    
    def step(self, actions: dict):
        '''
        step
        Take relative actions and update information.
        
        :param actions: actions list to be executed at all intersections at the next step
        :return: None
        '''
        # ========Record agents that executed actions (important!)=========
        # These agents acted, RLlib expects their reward and next_obs
        agents_that_acted = list(actions.keys())
        
        # ========Execute actions for these agents===================
        for tl, action in actions.items():
            if self.id2intersection[tl].time_to_act:
                actual_action=self.id2intersection[tl].pseudo_step(action)
                actions[tl] = actual_action
        # ========Simulate until new agents need to act================
        self.step_sim_until_time_to_act()
        # Initialize observation and reward dictionaries
        observations = {}
        rewards = {}
        # Check if new agents need to act (for next step)
        newly_acting_agents = [
            tl for tl in self.traffic_light_ids 
            if self.id2intersection[tl].time_to_act
        ]
        
        # Provide observations (and rewards) for new agents
        for tl in newly_acting_agents:
            observations[tl] = self.id2intersection[tl].get_observation()
            rewards[tl] =  self.id2intersection[tl].get_reward()
        # dones: no agent is done yet
        all_agents = set(agents_that_acted) | set(newly_acting_agents)
        dones = {tl: False for tl in all_agents}
        dones["__all__"] = self.step_counter >= self.sim_max_steps
        
        # 🔑 Debug info
        if dones["__all__"]:
            decision_counts = {tl: self.id2intersection[tl].action_count for tl in self.traffic_light_ids}
            total_decisions = sum(decision_counts.values())
            avg_decisions = total_decisions / len(self.traffic_light_ids)
        
            if self.sync_mode:
                theoretical_decisions = (self.step_counter - 300) // 5
                print(f"   Mode: synchronous")
                print(f"   Decisions per agent: {avg_decisions:.0f}")
                print(f"   Theoretical decisions: {theoretical_decisions}")
            else:
                theoretical_decisions_total = len(self.traffic_light_ids) * ((self.step_counter - 300) // 5)
                print(f"   Mode: asynchronous")
                print(f"   Total actual decisions: {total_decisions}")
                print(f"   Total theoretical decisions: {theoretical_decisions_total}")
                print(f"   Average per agent: {avg_decisions:.1f}")
        
        # Force all agents to be included in final step so obs/rewards are always complete
        if dones["__all__"]:
            newly_acting_agents = list(self.traffic_light_ids)
        else:
            newly_acting_agents = [
                tl for tl in self.traffic_light_ids
                if self.id2intersection[tl].time_to_act
            ]
            # ========== ✅ Add SUMO statistics ==========

        # info

        info = {tl: {} for tl in all_agents}
        # # ======================================================================
        # # # statistic follow the warmup
        # # if self.step_counter == 300:
        # #     pass
        # # else: # 否则执行每次状态更新统计工作
        # #     self.step_sim_and_statistics() # if repeated with reset?
        # for tl, action in actions.items():
        #         if self.id2intersection[tl].time_to_act:
        #             self.id2intersection[tl].pseudo_step(action)
        # # when warmup passed, we can take action
        # # if action is not None:
        # #     for i, intersection in enumerate(self.intersections):
        # #         if intersection.time_to_act:
        # #                 intersection.pseudo_step(action[i])
        # self.step_sim_until_time_to_act(actions)
        # # 在获取rewards和observations之前，先确定哪些agents需要决策
        # acting_agents = [
        #     tl for tl in self.traffic_light_ids 
        #     if self.id2intersection[tl].time_to_act
        # ]   
        # # 为这些固定的agents获取奖励
        # rewards = {
        #     tl: self.id2intersection[tl].get_reward()
        #     for tl in acting_agents
        # }
        
        # # =============compute observation for the coming decision================================
        # # 为相同的agents获取观测
        # observations = {
        #     tl: self.id2intersection[tl].get_observation()
        #     for tl in acting_agents
        # } 
        # # =============CONCLUDE the recent steps and compute rewards result from the last action decision========================
        # # CONCLUDE the steps (rewards) and update observations. 
        # # This section is the end of recent steps from last action and the start of new action.
        # # rewards = self._get_rewards()     
        # # return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act)
        
        # # =============compute observation for the coming decision================================
        # #找准 更新状态 观测 奖励 的位置， 修改观测 奖励计算的方法， 形成标准接口的方法. 订阅被用于计算观测对象。
        # # observations = self._get_observations()
        # # 重要：确保observations和rewards的keys完全一致
        # # 如果某个agent在其中一个字典中缺失，添加默认值
        
        # dones = {tl: False for tl in acting_agents} #这个应该放在reset中
        # dones["__all__"] = self.step_counter >= self.sim_max_steps #这个应该放在step_sim_until_time_to_act中

        # # terminated = False  # there are no 'terminal' states in this environment
        # # truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        # # info = self._compute_info()
        # info = {tl: {} for tl in acting_agents}
        return observations, rewards, dones, info

    def close(self):
        # Print final statistics when the simulation ends
        
        if self.interface_flag:
            try:
                libsumo.close()
            except:
                pass
        else:
            try:
                traci.switch(self.connection_name)
                traci.close()
            except:
                pass # already closed or connection doesnot exist
    
    def set_traffic_scale(self, scale: float):
        """
        Set the traffic flow scaling factor
        
        ⚠️ Note: must be called before reset() to take effect
        because SUMO needs to apply this parameter at startup
        
        Args:
            scale: traffic scaling factor
                   - 1.0: normal flow (default)
                   - 0.5: half flow
                   - 2.0: double flow
        
        Example:
            >>> env.set_traffic_scale(1.5)  # increase traffic by 50%
            >>> obs = env.reset()  # new traffic setting takes effect on this reset
        """
        if scale <= 0:
            raise ValueError(f"Traffic scaling factor must be greater than 0, current value: {scale}")
        
        self.traffic_scale = scale
        # Regenerate SUMO command (including new scale parameter)
        self.sumo_cmd = self.generate_sumo_cmd()
        
        print(f"✅ Traffic scaling factor set to: {scale} (will take effect on next reset)")
    
    def get_traffic_scale(self):
        """Get the current traffic scaling factor"""
        return self.traffic_scale
        
    def observation_spaces(self, ts_id: str):
        """
        Return the observation space of a traffic signal (static method).
        Uses pre-parsed static information, no need to instantiate Intersection or call reset()
        """
        if not hasattr(self, 'traffic_light_info'):
            raise RuntimeError(
                "traffic_light_info not initialized. "
                "This should be set in parse_sumo_config.__init__()"
            )
        
        # ✅ Get obs_to_subscribe from config
        obs_to_subscribe = self.get_obs_to_subscribe()
        
        # ✅ Check whether to use presslight (forces in_only=False)
        use_presslight = 'presslight' in obs_to_subscribe
        in_only = False if use_presslight else True
        return self.get_observation_space_static(ts_id, obs_to_subscribe, in_only=in_only)

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal (static method)."""
        if not hasattr(self, 'green_phases'):
            raise RuntimeError("green_phases not initialized.")
        
        if ts_id not in self.green_phases:
            raise ValueError(f"Traffic light '{ts_id}' not found.")
        
        # Action space = number of green phases
        num_actions = len(self.green_phases[ts_id])
        return gym.spaces.Discrete(num_actions)
    # ==============================================Road Closure==============================
    def close_edges(self, edge_ids, skip_validation=False):
        """
        Close specified edges
        
        Args:
            edge_ids: list of edge IDs to close
            skip_validation: skip validation and close directly
        """
        if not edge_ids:
            return

        if skip_validation:
            print("⚠ Skipping validation, closing roads directly")
            valid_edges = edge_ids
        else:
            # Validate using connectivity test
            print("🔍 Validating closable edges using connectivity test...")
            closable_edges_set = set(self.get_closable_edges())
            valid_edges = [e for e in edge_ids if e in closable_edges_set]
            invalid_edges = [e for e in edge_ids if e not in closable_edges_set]
            
            if invalid_edges:
                print(f"\n⚠ Warning: {len(invalid_edges)} edges do not satisfy closure conditions (they are required paths for some OD pairs):")
                for edge_id in invalid_edges:
                    print(f"   {edge_id}")
        
        if not valid_edges:
            print("\n❌ No valid edges can be closed")
            return
        
        # ✅ New: parse and exclude all vehicle departure edges from rou.xml
        try:
            # Get all departure edges from rou.xml
            depart_edges = self.get_all_depart_edges_from_rou()
            
            # Filter out departure edges
            edges_before_filter = len(valid_edges)
            valid_edges = [e for e in valid_edges if e not in depart_edges]
            
            if edges_before_filter > len(valid_edges):
                excluded_count = edges_before_filter - len(valid_edges)
                print(f"\n⚠ Excluded {excluded_count} edges (vehicle departure edges)")
                excluded_edges = [e for e in edge_ids if e in depart_edges]
                for edge in excluded_edges[:5]:  # show only first 5
                    print(f"   - {edge}")
                if len(excluded_edges) > 5:
                    print(f"   ... and {len(excluded_edges)-5} more edges")
            
            if not valid_edges:
                print("\n❌ All edges are vehicle departure edges, cannot close any")
                return
                
        except Exception as e:
            print(f"⚠ Error while excluding departure edges: {e}")


        # 2. Identify affected vehicles (both running and pending)
        affected_vehicles = set()
        try:
            # 2.1 Vehicles already on the network
            all_vehicles = self.eng.vehicle.getIDList()
            for veh_id in all_vehicles:
                try:
                    remaining_route = self.eng.vehicle.getRoute(veh_id)
                    if any(edge in remaining_route for edge in valid_edges):
                        affected_vehicles.add(veh_id)
                except Exception as e:
                    pass
            
            print(f"\n📊 On-road vehicles: found {len(affected_vehicles)}/{len(all_vehicles)} vehicles whose routes include edges to be closed")
            
        except Exception as e:
            print(f"⚠ Error finding affected vehicles: {e}")
        
        # 3. Close edges using setEffort (logical closure)
        closed_count = 0
        for edge_id in valid_edges:
            try:
                # self.eng.edge.setDisallowed(edge_id, ["all"])
                # print(f"✓ 道路 {edge_id} 已封闭")
                # Instead of physically blocking, assign extremely high cost
                self.eng.edge.setEffort(edge_id, float('inf'))
                
                # Optional: also set travel time very high (used by some routing algorithms)
                self.eng.edge.adaptTraveltime(edge_id, float('inf'))
                
                print(f"✓ Edge {edge_id} set to extremely high cost (logically closed)")
                closed_count += 1
            except Exception as e:
                print(f"✗ Failed to close edge {edge_id}: {e}")
        
        print(f"\n🚧 Successfully closed {closed_count}/{len(valid_edges)} edges")
        
        # 4. Reroute affected vehicles
        if affected_vehicles:
            rerouted_count = 0
            failed_count = 0
            
            for veh_id in affected_vehicles:
                try:
                    # self.eng.vehicle.reroute(veh_id)
                    # Use rerouteTraveltime to consider effort values
                    self.eng.vehicle.rerouteTraveltime(veh_id)
                    rerouted_count += 1
                except Exception as e:
                    failed_count += 1
            print("✓ Successfully rerouted {rerouted_count} affected vehicles")
            if failed_count > 0:
                print(f"⚠ {failed_count} vehicles encountered issues during rerouting (will continue using original route)")
        else:
            print("✓ No vehicles currently affected")
            
    def finalize_stranded_vehicles(self):
        """Flush vehicles still in the sim at episode end into the finished lists,
        using their partial accumulated totals (trip didn't complete before cutoff)."""
        for v, totals in list(self.vehicle_group_delay_totals.items()):
            if totals['group1'] > 0:
                self.finished_reg_group1_delays.append(totals['group1'])
            if totals['group2'] > 0:
                self.finished_reg_group2_delays.append(totals['group2'])
            self.finished_reg_all_delays.append(totals['group1'] + totals['group2'])
        self.vehicle_group_delay_totals = {}
        self.vehicle_ever_group1 = set()
    
        for v, total in list(self.vehicle_ev_delay_totals.items()):
            self.finished_ev_delays.append(total)
        self.vehicle_ev_delay_totals = {}

class Intersection():
    '''
    Intersection Class is mainly used for describing crossing information and defining acting methods.
    '''
    def __init__(self, id, world, obs_to_subscribe, reward_to_subscribe, green_phases):
        
        self.world = world
        self.world_current_time = world.get_current_time
        self.interface_flag = world.interface_flag
        self.eng = world.eng
           
        self.id = id

        
        self.green_phase=0
        self.decision_interval=world.get_decision_interval() # decision interval read from config
        # links and phase information of certain intersection
        # self.current_phase = 0
        self.steps_since_action = 0  # track number of steps since last decision
        self.time_since_last_phase_change = 0
        self.is_yellow = False
        self.next_action_time = 300 # warmup time
        # self.yellow_phase_time = min([i.duration for i in self.eng.trafficlight.getAllProgramLogics(self.id)[0].phases])
        self.yellow_phase_time = world.get_yellow_length()
        self.min_green = world.get_min_green()
        self.action_count = 0  # track actual number of decisions

        # Get static information from pre-parsed data (completed in parse_sumo_config.__init__)
        # if hasattr(world, 'traffic_light_info') and id in world.traffic_light_info:
        # 使用静态预解析的数据
        # 🔑 从预解析的静态信息中获取（必须存在）
        if not hasattr(world, 'traffic_light_info'):
            raise RuntimeError(
                "traffic_light_info not found in World object. "
                "Static preprocessing must be completed in parse_sumo_config.__init__()"
            )

        if id not in world.traffic_light_info:
            raise ValueError(
                f"Traffic light '{id}' not found in traffic_light_info. "
                f"Available IDs: {list(world.traffic_light_info.keys())}"
            )
        
        tl_info = world.traffic_light_info[id]
        self.lanelinks = tl_info['lanelinks']
        self.road_lane_mapping = tl_info['road_lane_mapping']
        self.roads = tl_info['roads']
        self.outs = tl_info['outs']
        self.directions = tl_info['directions']
        self.lanes = tl_info['lanes']
        self.in_roads = tl_info['in_roads']
        self.out_roads = tl_info['out_roads']

        self.green_phases = green_phases
        
        self.phases_idx = [i for i in range(len(self.green_phases))]
        self.phase_available_startlanes: dict[int, list[str]]  = {}
        self.startlanes: list[str] = [] # represents all entry lane ids of the whole intersection
        self.phase_available_lanelinks: dict[int, list[tuple[str, str]]] = {}
        
        # old todo was here: - 以下注释代码可考虑静态获取
        # # 在循环外获取一次连接数据
        # all_links = self.eng.trafficlight.getControlledLinks(self.id) # repeated call for line 282
        # for p_idx, phase in enumerate(self.green_phases): # 遍历你的绿灯相位
        # # phase.state 是相位状态字符串，例如 'GGGrrr'
        #     tmp_startanes = []
        #     tmp_lanelinks = []
            
        #     for signal_index, signal_state in enumerate(phase.state): # 遍历该相位的每个信号状态
        #         # 检查信号状态是否为绿灯或绿闪
        #         if signal_state in ['G', 'g', 's']: 
        #             # 确保信号索引在有效范围内
        #             if signal_index < len(all_links):
        #                 # 获取该信号状态控制的所有连接
        #                 links_under_this_signal = all_links[signal_index]
        #                 if not links_under_this_signal:
        #                     continue
        #                 # 现在处理这些连接
        #                 for link_info in links_under_this_signal:
        #                     from_lane, to_lane, via_lane = link_info
        #                     # ... 处理你的逻辑，例如记录起始车道等 ...
        #                     if from_lane not in tmp_startanes:
        #                         tmp_startanes.append(from_lane)
        #                     tmp_lanelinks.append((from_lane, to_lane))

        #                     if from_lane not in self.startlanes:
        #                         self.startlanes.append(from_lane)
        
        #     self.phase_available_startlanes[p_idx] = tmp_startanes # get entry lane ids of the phase
        #     self.phase_available_lanelinks[p_idx] = tmp_lanelinks # get entry and exit lane ids of the phase

        self.all_phases, self.yellow_dict = self.create_yellows(self.green_phases, self.yellow_phase_time, self.interface_flag)
        
        # get the full phases including yellow phases and set the program logic
        # programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        tl_id = self.id + "_rl"
        logic = self.eng.trafficlight.Logic(tl_id, 0, 0, self.all_phases)
        self.eng.trafficlight.setProgramLogic(self.id, logic)
        self.eng.trafficlight.setProgram(self.id, tl_id)
        # ===================================================
        
        # dictionary of remembered features
        # self.waiting_times = dict()
        self.full_observation = None
        self.last_step_vehicles = None
        # 🔧 新增：跟踪每辆车在当前路口各车道的等待时间
        self.lane_vehicle_waiting_times = {}  # {lane: {vehicle_id: waiting_time}}
        # - old todo was here check .signals .full_observation .last_stet_vehicles need to be set or not
        
        # ===========================observation action and reward===================================
        # self.obs_to_subscribe = obs_to_subscribe
        self.obs_to_subscribe = [fn for fn in obs_to_subscribe 
                              if fn not in ['presslight']]
        self.reward_to_subscribe = reward_to_subscribe
        self.world.subscribe(self.obs_to_subscribe)
        if 'real_delay' in self.reward_to_subscribe or 'real_delay' in self.obs_to_subscribe:
            self.world.subscribe(['vehicle_trajectory'])
        self.Observations = Observation(self, world, obs_to_subscribe, in_only=True)
        # self.observation_space = self.Observations.observation_space()
        self.observation_space = world.get_observation_space_static(self.id, obs_to_subscribe, in_only=True)
        self._action_space = gym.spaces.Discrete(len(self.green_phases))
        self.Rewards = GetRewards(self, world, self.reward_to_subscribe, in_only=True, negative=True)
        self.last_reward = None
        self.accumulated_reward_since_last_action = 0  # 初始化累积奖励
        

    def create_yellows(self, green_phases, yellow_duration, interface_flag):
        """
        find all possible yellow phases between each pair of phases
        """
        # =========================================================================================
        # interface_flag: 1:libsumo, 0: traci
        all_phases = green_phases.copy() # all phases including yellow phases
        yellow_dict: dict[tuple, int] = {}    # currentphase_nextphase: yellow phase index
        # Automatically create yellow phases, traci will report missing phases as it assumes execution by index order
        
        for i, p1 in enumerate(green_phases):
            for j, p2 in enumerate(green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                yellow_dict[(i, j)] = len(all_phases)
                all_phases.append(self.eng.trafficlight.Phase(yellow_duration, yellow_state))
                # all_phases.append(traci.trafficlight.Phase(yellow_duration, yellow_state))
        return all_phases, yellow_dict
    

    def reset(self):
        '''
        reset
        Reset information, including current_phase, full_observation and last_step_vehicles, etc.
        
        :param: None
        :return: None
        '''

        # self.current_phase_time = 0
        self.time_since_last_phase_change = 0
        # self.virtual_phase = 0
        # self.next_phase = 0
        self.is_yellow = False
        self.next_action_time = 300
        # self.waiting_times = dict()
        self.full_observation = None
        self.last_step_vehicles = None
        self.action_count = 0  # reset decision counter
        # 🔧 新增：重置车道等待时间跟踪
        self.lane_vehicle_waiting_times = {}
        # self.current_phase = self.get_current_phase()
        # eng is set in world
        # programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        # logic = programs[0]
        # logic.type = 0
        # logic.phases = self.all_phases
        # self.eng.trafficlight.setProgramLogic(self.id, logic)
        # ✅ FIX: Don't modify the program logic here - it was already set in __init__
        # Just ensure the RL program is active
        tl_id = self.id + "_rl"
        try:
            self.eng.trafficlight.setProgram(self.id, tl_id)
        except Exception as e:
            print(f"Warning: Could not set program for {self.id}: {e}")
    def get_current_phase(self):
        '''
        get_current_phase
        Get current phase of current intersection.
        
        :param: None
        :return cur_phase: current phase of current intersection
        '''
        cur_phase = self.eng.trafficlight.getPhase(self.id)
        return cur_phase
    
    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.world_current_time() >= self.next_action_time

    def pseudo_step(self, action):
        self.next_action_time = float('inf')
        self.action_count += 1
        self.accumulated_reward_since_last_action = 0
        self.steps_since_action = 0
        
        # is_yellow is always False here (yellow already ended in step_sim_until_time_to_act)
        #if self.green_phase == action or self.time_since_last_phase_change < self.min_green:
        if self.green_phase == action or self.time_since_last_phase_change <= self.min_green:

            self.eng.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.world_current_time() + self.decision_interval
            return self.green_phase
    
        else:  # green_phase != action and min_green satisfied
            #print(f"DEBUG YELLOW START: {self.id} at sim_time={self.world_current_time():.1f}, switching {self.green_phase}->{action}")
            self.eng.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, action)]].state)
            self.green_phase = action
            self.next_action_time = (self.world_current_time() 
                                     + self.yellow_phase_time 
                                     + self.decision_interval)
            self.is_yellow = True
            self.time_since_last_phase_change = 0
            return action

    #==========================original objective traffic state statistics===========================
    def collect_objective_traffic_state(self, max_distance, step_length=1):
        '''
        # - old todo was here para step_length can be discarded
        collect_objective_traffic_state
        Get observation of the whole roadnet, including lane_waiting_time_count, lane_waiting_count, lane_count and queue_length.
        
        :param step_length: time duration of step
        :param distance: distance limitation that it can only get vehicles which are within the length of the road
        :return: None
        '''
        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'lane_waiting_time_count': 0, 'lane_waiting_count': 0, 'lane_count': 0}
            lane_measures['lane_waiting_count'] = self.eng.lane.getLastStepHaltingNumber(lane)
            lane_measures['lane_count'] = self.eng.lane.getLastStepVehicleNumber(lane)
        
            lane_vehicles = self._get_vehicles(lane, max_distance)
            # initialize waiting time dictionary for this lane
            if lane not in self.lane_vehicle_waiting_times:
                self.lane_vehicle_waiting_times[lane] = {}
            # 记录当前车道上的车辆ID，用于清理已离开的车辆
            current_lane_vehicles_set = set(lane_vehicles)
            for v in lane_vehicles:
                all_vehicles.add(v)
                v_measures = dict()
                v_measures['name'] = v
                v_measures['speed'] = self.eng.vehicle.getSpeed(v)
                v_measures['position'] = self.eng.vehicle.getLanePosition(v)

                # determine whether vehicle is waiting based on speed threshold
                is_waiting = v_measures['speed'] < 0.1  # 速度阈值 0.1 m/s
                
                # maintain per-vehicle waiting time per lane
                if v not in self.lane_vehicle_waiting_times[lane]:
                    self.lane_vehicle_waiting_times[lane][v] = 0.0
                
                if is_waiting:
                    # accumulate waiting time if vehicle is stopped
                    self.lane_vehicle_waiting_times[lane][v] += step_length
                    # lane_measures['lane_waiting_count'] += 1  # 🔧 当前排队车辆数
                else:
                    # 车辆正在移动，不累加等待时间（但保留历史值）
                    pass
                v_measures['wait'] = self.lane_vehicle_waiting_times[lane][v]
                # lane_measures['queue_length'] = lane_measures['queue_length'] + 1
                # 🔧 修正3: 只累加当前排队车辆的等待时间
                if is_waiting:
                    lane_measures['lane_waiting_time_count'] += v_measures['wait']
                # 统计车道总车辆数
                # lane_measures['lane_count'] += 1
                vehicles.append(v_measures)
            # remove vehicles that left the lane
            vehicles_to_remove = set(self.lane_vehicle_waiting_times[lane].keys()) - current_lane_vehicles_set
            for v in vehicles_to_remove:
                del self.lane_vehicle_waiting_times[lane][v]
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures
        """
        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = self.last_step_vehicles.difference(all_vehicles)
            departs = all_vehicles.difference(self.last_step_vehicles)
            full_observation['departures'] = departs
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)
        self.last_step_vehicles = all_vehicles
        """
        self.full_observation = full_observation

    def _get_vehicles(self, lane, max_distance):
        '''
        _get_vehicles
        Get number of vehicles running on the specific lane within max distance.
        
        :param lane: lane id
        :param max_distance: distance limitation that it can only get vehicles which are within the length of the lane
        :return detectable: number of vehicles
        '''
        # old todo was here - reduce complexity -> find all vehicles within max_distance and on this lane
        # detectable = []
        # for v in self.eng.lane.getLastStepVehicleIDs(lane):
        #     path = self.eng.vehicle.getNextTLS(v)
        #     if len(path) > 0:
        #         next_light = path[0]
        #         distance = next_light[2]
        #         if distance <= max_distance:
        #             detectable.append(v)
        # return detectable
        '''
        Get vehicles within detection range on a given lane.

        :param lane: lane id
        :param max_distance: detection range limit
        :return detectable: list of vehicles within range
        '''
        detectable = []
        
        # 获取车道长度（信号灯通常在车道末端）
        lane_length = self.eng.lane.getLength(lane)
        # 遍历车道上的所有车辆
        for v in self.eng.lane.getLastStepVehicleIDs(lane):
            # 获取车辆在车道上的位置（从车道起点算起）
            vehicle_position = self.eng.vehicle.getLanePosition(v)
            # 计算车辆到车道末端（信号灯）的距离
            distance_to_tls = lane_length - vehicle_position
            # 只保留在检测范围内的车辆
            if distance_to_tls <= max_distance and distance_to_tls >= 0:
                detectable.append(v)
        
        return detectable

# ========================================================================
# - old todo was here reward computation, action space, observation space design
# multi-agent environment design question: should reward belong to intersection level?
# =========obeservation, reward and done compute=============================================
    def get_observation(self):
        return self.Observations.compute_observation()

    # def get_reward(self):
    #     """Computes the reward of the traffic signal. If it is a list of rewards, it returns a numpy array."""
        
    #     self.last_reward = self.Rewards.compute_reward()
    #     return self.last_reward
        # else:
        #     self.last_reward = np.array([reward_fn(self) for reward_fn in self.reward_list], dtype=np.float32)
        #     if self.reward_weights is not None:
        #         self.last_reward = np.dot(self.last_reward, self.reward_weights)  # Linear combination of rewards
        # ==============================================================================
    def get_reward(self):
        """Return average reward since last action."""
        if self.steps_since_action > 0:
            avg_reward = self.accumulated_reward_since_last_action / self.steps_since_action
        else:
            avg_reward = 0.0  # 避免除以零
        return avg_reward  # ✅ 返回平均值，不重置（在pseudo_step中重置）

    def accumulate_reward(self):
        """Accumulate instantaneous reward at each simulation step."""
        instant_reward = self.Rewards.compute_reward()
        self.accumulated_reward_since_last_action += instant_reward
        self.steps_since_action += 1  # ✅ 添加：累加步数