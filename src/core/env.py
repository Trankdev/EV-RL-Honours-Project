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
from .Observations import Observation
class parse_sumo_config(): # 静态信息获取会影响并行仿真吗？
    def __init__(self, sumo_config, **kwargs):
        self.sumo_config = sumo_config
        with open(sumo_config) as f:
            self.sumo_dict = json.load(f)
        # ✅ 从训练脚本接收（必须提供，不从CFG读取）
        self._obs_to_subscribe = kwargs.get('obs_to_subscribe')
        self._reward_to_subscribe = kwargs.get('reward_to_subscribe')
        self._algorithm_name = kwargs.get('algorithm_name')
        self._normalize_observation = kwargs.get('normalize_observation', False)
        self._norm_params = kwargs.get('norm_params', {})  # ✅ 添加这一行
        self.RIGHT = True
        self.traffic_light_ids = []

        if kwargs['interface'] == 'libsumo':
            self.interface_flag = True
        elif kwargs['interface'] == 'traci':
            self.interface_flag = False
        else:
            raise Exception('NOT IMPORTED YET')
        # ✅ 添加流量缩放因子支持
        self.traffic_scale = kwargs.get('traffic_scale', 1.0)
        self.seed = kwargs.get('seed', None)

        self.connection_name = self.get_connection_name()
        if self.interface_flag:
                self.eng = libsumo
        else:            
            self.eng = traci

        # 🔑 在这里调用预解析方法（顺序很重要！）
        self.traffic_light_ids = self._get_traffic_light_ids()  # 先解析交通灯ID
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
        # 1. 控制台显示摘要
        sumo_cmd += ['--duration-log.statistics', 'true']
        
        # 2. 添加随机种子支持（如果提供）
        if self.seed is not None:
            sumo_cmd += ['--seed', str(self.seed)]
        
        # 3. 添加流量缩放因子（新增）
        if self.traffic_scale != 1.0:
            sumo_cmd += ['--scale', str(self.traffic_scale)]
            print(f"⚙️ SUMO 流量缩放因子: {self.traffic_scale}")
            
        # 4. 添加 additional files 支持（用于 rerouter 等）
        if self.sumo_dict.get('additional_files'):
            additional_files = self.sumo_dict['additional_files']
            if isinstance(additional_files, str):
                additional_files = [additional_files]
            
            for add_file in additional_files:
                # 如果是相对路径，加上 dir 前缀
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
        return self.sumo_dict.get('yellow_length', 3)  # 默认3秒
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
                    if edge.get('id') and not edge.get('function')]  # 排除内部边
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
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]  # TODO: check if its 4

    
    # TODO: revert x and y
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
        预解析所有交通灯的完整信息(lanelinks + road_lane_mapping + lanes)
        在 __init__ 时调用，避免运行时解析
        
        Returns:
            {tl_id: {
                'lanelinks': [...],
                'road_lane_mapping': {...},
                'roads': [...],
                'outs': [...],
                'directions': [...],
                'lanes': [...],
                'in_roads': [...],
                'out_roads': [...]
            }}
        """
        # 先获取所有 lanelinks
        all_lanelinks = self._parse_all_lanelinks()
        
        result = {}
        
        for tl_id in self.traffic_light_ids:
            lanelinks = all_lanelinks.get(tl_id, [])
            
            # 初始化数据结构
            road_lane_mapping = {}
            roads = []
            outs = []
            directions = []
            
            # 处理每个 link（复制原逻辑）
            for link_list in lanelinks:
                if not link_list:
                    continue
                
                link = link_list[0]  # 取第一个连接
                from_lane = link[0]
                to_lane = link[1]
                
                # 提取道路 ID
                from_road = from_lane[:-2]
                to_road = to_lane[:-2]
                
                # 处理 from_lane (入口道路)
                if from_road not in road_lane_mapping:
                    road_lane_mapping[from_road] = []
                    roads.append(from_road)
                    outs.append(False)
                    
                    # 获取车道形状并计算方向
                    road_shape = self.get_lane_shape_from_net(from_lane)
                    directions.append(self._get_direction(road_shape, False))
                
                if from_lane not in road_lane_mapping[from_road]:
                    road_lane_mapping[from_road].append(from_lane)
                
                # 处理 to_lane (出口道路)
                if to_road not in road_lane_mapping:
                    road_lane_mapping[to_road] = []
                    roads.append(to_road)
                    outs.append(True)
                    
                    # 获取车道形状并计算方向
                    road_shape = self.get_lane_shape_from_net(to_lane)
                    directions.append(self._get_direction(road_shape, True))
                
                if to_lane not in road_lane_mapping[to_road]:
                    road_lane_mapping[to_road].append(to_lane)
            
            # 排序道路
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
            
            # 构建有序车道列表和 in/out roads
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
            
            # 存储完整信息
            result[tl_id] = {
                'lanelinks': lanelinks,
                'road_lane_mapping': road_lane_mapping,
                'roads': sorted_roads, # in_only==False
                'outs': sorted_outs,
                'directions': sorted_directions,
                'lanes': lanes,
                'in_roads': in_roads, # in_only==True
                'out_roads': out_roads,
                # 🔑 新增：预存储两种模式的 lanes_road_observed
                'lanes_road_observed': self.build_lanes_road_observed(sorted_roads, road_lane_mapping, self.RIGHT),
                'lanes_road_observed_in_only': self.build_lanes_road_observed(in_roads, road_lane_mapping, self.RIGHT),
            }
        
        return result
    
    def get_observation_space_static(self, tl_id: str, obs_to_subscribe: list, in_only: bool = True):
        """
        静态计算观测空间，无需实例化 Intersection
        
        Args:
            tl_id: 交通灯ID
            obs_to_subscribe: 观测特征列表（如 ['lane_waiting_count'])
            in_only: 是否只观测入口车道
            
        Returns:
            gym.spaces.Box: 观测空间
        """
        if not hasattr(self, 'traffic_light_info'):
            raise ValueError("traffic_light_info not initialized.")
        
        if tl_id not in self.traffic_light_info:
            raise ValueError(f"Traffic light {tl_id} not found.")
        
        tl_info = self.traffic_light_info[tl_id]
        
        use_presslight = 'presslight' in obs_to_subscribe
        in_only = False if use_presslight else in_only
        # 🔑 直接从预存储的数据中读取
        if in_only:
            lanes_road_observed = tl_info['lanes_road_observed_in_only']
        else:
            lanes_road_observed = tl_info['lanes_road_observed']
        
        # ✅ 计算观测维度
        num_phases = len(self.green_phases[tl_id])  # 相位数量
        phase_onehot_dim = num_phases                # 相位 one-hot 编码
        min_green_dim = 1                            # 最小绿灯时间标志
        # ========== 新增：项目1模式特殊处理 ==========
        algorithm_name = getattr(self, '_algorithm_name', '')
        if 'project1' in algorithm_name.lower() or 'std_dqn' in algorithm_name.lower():
            # 动态格式: phase(N_phases) + N_in_lanes × 5 features
            # 车道数由路网自动决定，不硬编码为 12
            num_phases   = len(self.green_phases[tl_id])
            num_in_lanes = sum(len(lanes) for lanes in tl_info['lanes_road_observed_in_only'])
            ob_length    = num_phases + num_in_lanes * 5

            return gym.spaces.Box(
                low=np.zeros(ob_length, dtype=np.float32),
                high=np.ones(ob_length, dtype=np.float32),
                dtype=np.float32
            )
        if use_presslight:
            # PressLight 模式：入口车道×3 + 出口车道×1
            num_in_lanes = sum(len(x) for x in tl_info['lanes_road_observed_in_only'])
            
            # 计算出口车道数
            out_roads = tl_info['out_roads']
            road_lane_mapping = tl_info['road_lane_mapping']
            num_out_lanes = sum(len(road_lane_mapping.get(road, [])) for road in out_roads)
            
            lane_features_dim = num_in_lanes * 3 + num_out_lanes
        else:
            # 计算观测维度（模拟 Observation.observation_space() 的逻辑）
            num_lanes = sum(len(x) for x in lanes_road_observed)  # 总车道数
            lane_features_dim = len(obs_to_subscribe)*num_lanes
        # 总维度
        ob_length = phase_onehot_dim+min_green_dim+lane_features_dim 

        return gym.spaces.Box(
            low=np.zeros(ob_length, dtype=np.float32),
            high=np.ones(ob_length, dtype=np.float32),
            dtype=np.float32
        )
    
    # 预计算 lanes_road_observed（两种模式 in_only==True/False）
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
        """获取观测订阅配置，如果未配置则返回默认值"""
        return self._obs_to_subscribe
    
    def get_reward_to_subscribe(self):
        """获取奖励订阅配置，如果未配置则返回默认值"""
        return self._reward_to_subscribe
    
    # =============================================Road Disruption Functions==========================================
    def get_route_file_path(self):
        """获取路径文件的路径"""
        if not self.sumo_dict.get('combined_file'):
            # 直接使用flowFile
            return os.path.join(self.sumo_dict['dir'], self.sumo_dict['flowFile'])
        else:
            # 从combined_file(.cfg)中解析
            cfg_path = os.path.join(self.sumo_dict['dir'], self.sumo_dict['combined_file'])
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(cfg_path)
                root = tree.getroot()
                
                # 查找route-files标签
                for input_tag in root.findall('.//route-files'):
                    route_file = input_tag.get('value')
                    if route_file:
                        # 路径可能是相对于cfg文件的
                        if not os.path.isabs(route_file):
                            route_file = os.path.join(self.sumo_dict['dir'], route_file)
                        return route_file
            except Exception as e:
                print(f"⚠ 无法从cfg文件解析路径文件: {e}")
        
        return None

    def parse_od_routes_from_file(self):
        """
        从rou.xml文件解析所有OD对及其路径
        
        Returns:
            dict: {(origin, destination): [route1, route2, ...]}
                其中每个route是边ID的列表
        """
        route_file = self.get_route_file_path()
        
        if not route_file or not os.path.exists(route_file):
            print(f"⚠ 路径文件不存在: {route_file}")
            return {}
        
        print(f"📄 解析路径文件: {route_file}")
        
        od_routes = {}  # {(origin, dest): [route1, route2, ...]}
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            # 解析<route>标签（可能是独立的或在vehicle/flow中）
            route_definitions = {}  # {route_id: edge_list}
            
            # 1. 解析独立的<route>定义
            for route_tag in root.findall('route'):
                route_id = route_tag.get('id')
                edges_str = route_tag.get('edges', '')
                if route_id and edges_str:
                    edges = edges_str.strip().split()
                    route_definitions[route_id] = edges
            
            # 2. 解析<vehicle>和<flow>中的路径
            for element in root.findall('vehicle') + root.findall('flow'):
                # 情况1: 使用route属性引用
                route_ref = element.get('route')
                if route_ref and route_ref in route_definitions:
                    edges = route_definitions[route_ref]
                else:
                    # 情况2: 嵌套的<route>标签
                    route_tag = element.find('route')
                    if route_tag is not None:
                        edges_str = route_tag.get('edges', '')
                        edges = edges_str.strip().split() if edges_str else []
                    else:
                        # 情况3: from/to属性（需要SUMO计算路径，暂时跳过）
                        continue
                
                if len(edges) >= 2:
                    origin = edges[0]
                    destination = edges[-1]
                    od_pair = (origin, destination)
                    
                    if od_pair not in od_routes:
                        od_routes[od_pair] = []
                    
                    # 避免重复路径
                    if edges not in od_routes[od_pair]:
                        od_routes[od_pair].append(edges)
            
            # 统计信息
            total_od_pairs = len(od_routes)
            total_routes = sum(len(routes) for routes in od_routes.values())
            od_with_multiple_routes = sum(1 for routes in od_routes.values() if len(routes) >= 2)
            
            print(f"✓ 从rou.xml解析了 {total_od_pairs} 个OD对（用于连通性测试）")
            
            return od_routes
            
        except Exception as e:
            print(f"⚠ 解析路径文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_all_depart_edges_from_rou(self):
        """
        从rou.xml文件解析所有车辆的起始边
        
        Returns:
            set: 所有起始边的集合
        """
        route_file = self.get_route_file_path()
        
        if not route_file or not os.path.exists(route_file):
            print(f"⚠ 路径文件不存在: {route_file}")
            return set()
        
        depart_edges = set()
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            # 1. 解析独立的<route>定义
            route_definitions = {}
            for route_tag in root.findall('route'):
                route_id = route_tag.get('id')
                edges_str = route_tag.get('edges', '')
                if route_id and edges_str:
                    edges = edges_str.strip().split()
                    if edges:
                        route_definitions[route_id] = edges[0]  # 只保存第一条边
            
            # 2. 解析<vehicle>和<flow>
            for element in root.findall('vehicle') + root.findall('flow'):
                # 情况1: 使用route属性引用
                route_ref = element.get('route')
                if route_ref and route_ref in route_definitions:
                    depart_edges.add(route_definitions[route_ref])
                else:
                    # 情况2: 嵌套的<route>标签
                    route_tag = element.find('route')
                    if route_tag is not None:
                        edges_str = route_tag.get('edges', '')
                        edges = edges_str.strip().split() if edges_str else []
                        if edges:
                            depart_edges.add(edges[0])
                    else:
                        # 情况3: from属性（起始边）
                        from_edge = element.get('from')
                        if from_edge:
                            depart_edges.add(from_edge)
        
            print(f"📄 从rou.xml解析了 {len(depart_edges)} 个不同的起始边")
            return depart_edges
            
        except Exception as e:
            print(f"⚠ 解析起始边时出错: {e}")
            return set()
    def get_closable_edges(self):
        """
        基于连通性测试的可封闭边识别
        
        思想：
        1. 对每个OD对，测试移除每条边后是否仍有路径
        2. 如果移除后仍连通 → 可封闭
        3. 如果移除后不连通 → 关键边（不可封闭）
        """
        import sumolib
        import networkx as nx
        
        # 1. 构建路网图
        print("📊 构建路网拓扑图...")
        net = sumolib.net.readNet(self.get_net_file_address())
        
        G = nx.DiGraph()
        for edge in net.getEdges():
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            G.add_edge(from_node, to_node, edge_id=edge.getID())
        
        print(f"   节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        
        # 2. 从rou.xml获取OD对
        od_routes = self.parse_od_routes_from_file()
        
        if not od_routes:
            print("⚠ 无法解析OD路径")
            return []
        
        # 转换为节点对（边 → 节点）
        od_node_pairs = set()
        for (origin_edge, dest_edge), routes in od_routes.items():
            try:
                origin_node = net.getEdge(origin_edge).getToNode().getID()
                dest_node = net.getEdge(dest_edge).getFromNode().getID()
                od_node_pairs.add((origin_node, dest_node))
            except:
                pass
        
        print(f"📊 分析 {len(od_node_pairs)} 个OD对的可摧毁边...")
        
        # 3. 测试每条边
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
            
            # 对每个OD对测试移除该边的影响
            for source, target in od_node_pairs:
                # ✅ 核心逻辑：模拟摧毁
                G_temp = G.copy()
                if G_temp.has_edge(from_node, to_node):
                    G_temp.remove_edge(from_node, to_node)
                
                # ✅ 检查连通性
                if not nx.has_path(G_temp, source, target):
                    # 移除后不连通 → 这是关键边
                    is_critical = True
                    critical_edges.add(edge_id)
                    break
            
            if not is_critical:
                destroyable_edges.add(edge_id)
        
        # 4. 统计结果
        print(f"\n✅ 分析完成:")
        print(f"  - 总边数: {total_edges}")
        print(f"  - 关键边（不可封闭）: {len(critical_edges)}")
        print(f"  - 可摧毁边（可封闭）: {len(destroyable_edges)}")
        
        return list(destroyable_edges)
# ===============================================================================================


class World(parse_sumo_config, gym.Env):
    '''
    World Class is mainly used for creating a SUMO engine and maintain information about SUMO world.
    '''

    def __init__(self, sumo_config, **kwargs):
        super().__init__(sumo_config, **kwargs)
        # ✨ 新增：同步/异步决策模式控制
        self.sync_mode = kwargs.get('sync_mode', False)  # 默认异步模式
        # ✅ 添加这3行：获取奖励配置（用于MA2C等特殊算法）
        self._reward_weights = kwargs.get('reward_weights', [1.0])
        self._reward_scale = kwargs.get('reward_scale', 1.0)
        self._reward_clip_range = kwargs.get('reward_clip_range', None)
        self.sumo_cmd = self.generate_sumo_cmd()
        self.warning = self.no_warning()
        self.connection_name = self.get_connection_name() # default: debug
        self.map_name = self.get_map_name()
        self.net = self.get_net_file_address()
        self.RIGHT = True
        
        # self.step_ratio = 1  # TODO: register in Registry later
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
        # ============预先获取交通灯和相位信息（不需要启动SUMO）============
        self.traffic_light_ids = self._get_traffic_light_ids()
        self.all_roads = self._get_roads()
        self.all_lanes = self._get_lanes()
        
        # 关键改动：在 __init__ 中就生成 green_phases（只解析XML，不启动SUMO）
        self.green_phases = self._generate_valid_phase()        
    
        # ==================world level dynamic statistics/ ==================
        self.num_arrived_vehicles = 0 # total number of vehicles that have arrived in the world
        self.num_departed_vehicles = 0 # total number of vehicles that have departed in the world   
        self.num_teleported_vehicles = 0 # total number of vehicles that have teleported in the world
        # 分清楚哪些是世界层面的哪些是交叉口面的指标       
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
            "pressure", # 获取每个路口的压强（入口车道车辆数减出口车道车辆数）（get_pressure 方法）
            "phase",#获取每个路口当前的信号相位（get_cur_phase 方法）
            "lane_count", # 获取每个车道的车辆数量（get_lane_vehicle_count 方法）。
            "lane_vehicles", # 获取每个车道的车辆列表（get_lane_vehicles 方法）。
            "lane_waiting_count", # 获取每个车道的等待车辆数量（get_lane_waiting_vehicle_count 方法）。
            "lane_pressure", #获取每个入口车道的压强（入口车道车辆数减对应出口车道车辆数）（get_lane_pressure 方法）
            "lane_waiting_time_count",#获取每个车道上所有等待车辆的总等待时间（get_lane_waiting_time_count 方法）
            "lane_delay",#计算每个车道的平均延误（1-平均速度/限速）（get_lane_delay 方法）
            "real_delay",#获取所有车辆的真实平均延误（get_real_delay 方法），基于车辆轨迹与理论期望时间的差值
            "vehicle_trajectory",#获取所有车辆的轨迹（车道变换及对应时间）（get_vehicle_trajectory 方法)
            "vehicles_average_trip_time",
            "outgoing_lane_vehicles",
        ] # 获取所有车辆的平均旅行时间（get_vehicles 方法），即所有已离开车辆的平均通过时间。
            
        self.fns_subscribed = []
        self.info_dynamics_real_time = {}

        # ==================every vehicle's dynamic statistics/individual level=================
        self.vehicles_entering_time = dict()
        self.vehicles_trip_time = dict() # vehicle_id: time_in_simulation
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
        # 这些会在第一次reset时初始化
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
        
        # 创建组合的动作空间
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
        vehicle_lane, self.vehicle_maxspeed = self.get_vehicle_lane() # get vehicles on tne roads except turning
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
        # 只在需要时才更新轨迹
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
        获取出口车道的车辆信息 (用于项目1的attention机制)
        
        Returns:
            outgoing_vehicles: dict
                {
                    'lane_id': {
                        'total_count': int,
                        'near_junction_count': int,  # 距离路口终点100米以上的车辆
                        'vehicles': [veh_id, ...]
                    },
                    ...
                }
        """
        outgoing_vehicles = {}
        
        # 遍历所有交通灯的出口道路
        for ts in self.intersections:
            for out_road in ts.out_roads:
                lanes = ts.road_lane_mapping.get(out_road, [])
                
                for lane_id in lanes:
                    try:
                        vehicle_ids = self.eng.lane.getLastStepVehicleIDs(lane_id)
                        lane_length = self.eng.lane.getLength(lane_id)
                        
                        near_junction_count = 0
                        
                        # 统计距离路口终点100米以上的车辆（靠近上游，拥堵区域）
                        for veh_id in vehicle_ids:
                            try:
                                position = self.eng.vehicle.getLanePosition(veh_id)
                                distance_to_end = lane_length - position
                                
                                # 项目1逻辑: 距离终点>=100米的车辆算拥堵
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
                        # 出错时返回空数据
                        outgoing_vehicles[lane_id] = {
                            'total_count': 0,
                            'near_junction_count': 0,
                            'vehicles': []
                        }
        
        return outgoing_vehicles
    # ====================================================================================

    def _update_infos(self):
        # TODO: add normalization value in the info_functions
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
            # else:
            #     print(f"Warning: No entry time recorded for vehicle {v}")
            # self.vehicles_trip_time.update({v: self.get_current_time() - self.vehicles_entering_time[v]})
        # world system level statistics
        self.num_arrived_vehicles += self.eng.simulation.getArrivedNumber() # total number of vehicles that have arrived in the world
        self.num_departed_vehicles += self.eng.simulation.getDepartedNumber() # total number of vehicles that have departed in the world
        self.num_teleported_vehicles += self.eng.simulation.getEndingTeleportNumber() # total number of vehicles that have teleported in the world
        
        self._update_infos() # step level dynamic information (world level, intersection level, vehicle level)
        # self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()


    def step_sim_until_time_to_act(self):
        time_to_act = False
        
        while not time_to_act:
            if self.step_counter >= self.sim_max_steps:
                print(f"DEBUG: 达到 sim_max_steps={self.sim_max_steps}，强制退出循环")
                break
            self.eng.simulationStep()            
            self.step_counter += 1
            self.step_sim_and_statistics() # 累计奖励
            for i, intersection in enumerate(self.intersections): # check if there is a intersection that needs to act
                
                # self.intersection.update()
                intersection.time_since_last_phase_change += 1
                if intersection.is_yellow and intersection.time_since_last_phase_change == intersection.yellow_phase_time:
                    # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
                    self.eng.trafficlight.setRedYellowGreenState(intersection.id, intersection.all_phases[intersection.green_phase].state)
                    intersection.is_yellow = False                
                # ✨ 根据模式判断是否需要决策
                if self.sync_mode:
                    # 同步模式：检查所有智能体是否都到达决策时间
                    if all(intsec.time_to_act for intsec in self.intersections):
                        time_to_act = True
                        break
                else:
                    # 异步模式：任意一个智能体需要决策即可
                    if intersection.time_to_act:
                        time_to_act = True
        
    def reset(self):
        '''
        reset
        reset information, including vehicles, vehicle_trajectory, etc.
    
        :param: None
        :return: None
        '''
        
        if self.step_counter != 0:
            # TODO: set trip info output
            self.close()
        # =============================================

        self.vehicles_trip_time = dict()
        self.vehicles_entering_time = dict()
        # TODO: check when to close traci
        if self.interface_flag:
            libsumo.start(self.sumo_cmd)
            self.eng = libsumo
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.eng = traci.getConnection(self.connection_name)
        # ==================首次运行时获取静态信息=================
        # if self.green_phases is None:
        # self.green_phases = self._generate_valid_phase()
        
        # ==================warmup simulation=================
        # 执行若干步以确保环境稳定
        for _ in range(300):
            self.eng.simulationStep()
        current_time = self.get_current_time()
        self.step_counter = int(current_time)
        print(f"DEBUG: Reset完成,当前step_counter = {self.step_counter}")
        
        # =====================创建交叉口对象==================================
        self.id2intersection = dict()
        self.intersections = []
        # 从配置文件获取观测和奖励订阅配置
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
        
        # ============首次运行获取in/out lanes=================
        if self.in_lanes is None:
            self.in_lanes, self.out_lanes = self.get_in_out_lanes()
        # =============创建动作空间=========================
        if self.action_space is None:
            action_dims = [intersection._action_space.n for intersection in self.intersections]
            self.action_space = gym.spaces.MultiDiscrete(action_dims)
        # Reset each intersection to initialize all necessary attributes        
        for intsec in self.intersections:
            intsec.reset()
            intsec.next_action_time = current_time
            intsec.collect_objective_traffic_state(self.max_distance)
            # 检查当前相位状态是否是黄灯
            phase_state = intsec.eng.trafficlight.getRedYellowGreenState(intsec.id)
            intsec.is_yellow = 'y' in phase_state
            if not intsec.is_yellow:
                try:
                    intsec.green_phase = next(
                        idx for idx, green_phase_obj in enumerate(intsec.green_phases) 
                        if green_phase_obj.state == phase_state)
                    intsec.time_since_last_phase_change = intsec.min_green
                except: # 如果找不到匹配的绿灯相位，报错，不使用默认相位
                    raise ValueError(f"Surprise: phase state of {intsec.id} : '{phase_state}' is not found in green phases")
            else:
                intsec.time_since_last_phase_change = intsec.yellow_phase_time
        # 初始化变量不进行step level的统计
        self._update_infos()
        # TODO: check if its the problem
        
        
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay= {}
        # TODO:compute initial observations
        return self._get_observations()
    
    def _get_observations(self):
        """获取当前需要决策的agents的观测"""
        if self.sync_mode:
            # ✨ 同步模式：返回所有智能体的观测
            observations = {
                tl: self.id2intersection[tl].get_observation()
                for tl in self.traffic_light_ids
            }
        else: # 异步模式：只返回当前需要决策的智能体
            # 获取当前需要决策的agents
            acting_agents = [
                tl for tl in self.traffic_light_ids 
                if self.id2intersection[tl].time_to_act
            ]
            # 只为当前需要决策的agents计算和返回观测
            observations = {
                tl: self.id2intersection[tl].get_observation()
                for tl in acting_agents
            }
        
        # 更新缓存（可选）
        self.observations.update(observations)
        
        return observations
    
    def _get_rewards(self):
        """获取当前需要决策的agents的奖励"""
        if self.sync_mode:
            # ✨ 同步模式：返回所有智能体的奖励
            rewards = {
                tl: self.id2intersection[tl].get_reward()
                for tl in self.traffic_light_ids
            }
        else: # 异步模式：只返回当前需要决策的智能体
            # 获取当前需要决策的agents（与observations保持一致）
            acting_agents = [
                tl for tl in self.traffic_light_ids 
                if self.id2intersection[tl].time_to_act
            ]
            # 只为当前需要决策的agents计算和返回奖励
            rewards = {
                tl: self.id2intersection[tl].get_reward()
                for tl in acting_agents
            }
        
        # 更新缓存（可选）
        self.rewards.update(rewards)
        
        return rewards
    
    def step(self, actions: dict):
        '''
        step
        Take relative actions and update information.
        
        :param actions: actions list to be executed at all intersections at the next step
        :return: None
        '''
        # ========记录执行action的agents（关键！）=========
        # 这些agents执行了action，RLlib期望收到它们的reward和next_obs
        agents_that_acted = list(actions.keys())
        
        # ========为这些agents执行action===================
        for tl, action in actions.items():
            if self.id2intersection[tl].time_to_act:
                actual_action=self.id2intersection[tl].pseudo_step(action)
                actions[tl] = actual_action
        # ========仿真直到有agents需要决策================
        self.step_sim_until_time_to_act()
        # 初始化观测和奖励字典
        observations = {}
        rewards = {}
        # 检查是否有新的agents需要决策（用于下一次step）
        # 这些agents将在下一次step中执行action
        newly_acting_agents = [
            tl for tl in self.traffic_light_ids 
            if self.id2intersection[tl].time_to_act
        ]
        
        # 如果有新的agents需要决策，也为它们提供观测（但不需要reward，因为它们还没执行action）
        for tl in newly_acting_agents:
            observations[tl] = self.id2intersection[tl].get_observation()
            rewards[tl] =  self.id2intersection[tl].get_reward()
        # dones: 刚执行过action的agents还没done，新需要决策的agents也没done
        all_agents = set(agents_that_acted) | set(newly_acting_agents)
        dones = {tl: False for tl in all_agents}
        dones["__all__"] = self.step_counter >= self.sim_max_steps
        # 🔑 添加调试信息
        if dones["__all__"]:
            # 统计每个agent的实际决策次数
            decision_counts = {tl: self.id2intersection[tl].action_count for tl in self.traffic_light_ids}
            
            total_decisions = sum(decision_counts.values())
            avg_decisions = total_decisions / len(self.traffic_light_ids)
            theoretical_decisions = len(self.traffic_light_ids) * ((self.step_counter-300)//5) # decision_interval
            
            if self.sync_mode:
                # 同步模式统计
                theoretical_decisions = (self.step_counter - 300) // 5
                print(f"   模式: 同步")
                print(f"   每个agent决策次数: {avg_decisions:.0f}")
                print(f"   理论决策次数: {theoretical_decisions}")
            else:
                # 异步模式统计
                theoretical_decisions_total = len(self.traffic_light_ids) * ((self.step_counter - 300) // 5)
                print(f"   模式: 异步")
                print(f"   实际决策总数: {total_decisions}")
                print(f"   理论决策总数: {theoretical_decisions_total}")
                print(f"   每个agent平均: {avg_decisions:.1f} 次")
            # ========== ✅ 添加SUMO统计指标 ==========

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
        # 在仿真结束时打印最终统计信息
        
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
        设置流量缩放因子
        
        ⚠️ 注意：必须在 reset() 之前调用才能生效
        因为 SUMO 需要在启动时应用该参数
        
        Args:
            scale: 流量缩放因子
                   - 1.0: 正常流量（默认）
                   - 0.5: 减半流量
                   - 2.0: 加倍流量
        
        Example:
            >>> env.set_traffic_scale(1.5)  # 增加50%流量
            >>> obs = env.reset()  # 新的流量设置会在这次reset时生效
        """
        if scale <= 0:
            raise ValueError(f"流量缩放因子必须大于0，当前值: {scale}")
        
        self.traffic_scale = scale
        # 重新生成 SUMO 命令（包含新的 scale 参数）
        self.sumo_cmd = self.generate_sumo_cmd()
        
        print(f"✅ 已设置流量缩放因子: {scale} (将在下次 reset 时生效)")
    
    def get_traffic_scale(self):
        """获取当前的流量缩放因子"""
        return self.traffic_scale
        
    def observation_spaces(self, ts_id: str):
        """
        Return the observation space of a traffic signal (static method).
        使用预解析的静态信息，无需实例化 Intersection 或调用 reset()
        """
        if not hasattr(self, 'traffic_light_info'):
            raise RuntimeError(
                "traffic_light_info not initialized. "
                "This should be set in parse_sumo_config.__init__()"
            )
        
        # ✅ 从配置中获取 obs_to_subscribe
        obs_to_subscribe = self.get_obs_to_subscribe()
        
        # ✅ 判断是否使用 presslight（强制 in_only=False）
        use_presslight = 'presslight' in obs_to_subscribe
        in_only = False if use_presslight else True
        return self.get_observation_space_static(ts_id, obs_to_subscribe, in_only=in_only)

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal (static method)."""
        if not hasattr(self, 'green_phases'):
            raise RuntimeError("green_phases not initialized.")
        
        if ts_id not in self.green_phases:
            raise ValueError(f"Traffic light '{ts_id}' not found.")
        
        # 动作空间 = 绿灯相位的数量
        num_actions = len(self.green_phases[ts_id])
        return gym.spaces.Discrete(num_actions)
    # ==============================================封闭道路==============================
    def close_edges(self, edge_ids, skip_validation=False):
        """
        封闭指定的边
        
        Args:
            edge_ids: 要封闭的边ID列表
            skip_validation: 跳过验证直接封闭
        """
        if not edge_ids:
            return

        if skip_validation:
            print("⚠ 跳过验证，直接封闭道路")
            valid_edges = edge_ids
        else:
            # 使用连通性测试验证
            print("🔍 使用连通性测试验证可封闭边...")
            closable_edges_set = set(self.get_closable_edges())
            valid_edges = [e for e in edge_ids if e in closable_edges_set]
            invalid_edges = [e for e in edge_ids if e not in closable_edges_set]
            
            if invalid_edges:
                print(f"\n⚠ 警告：{len(invalid_edges)} 条道路不满足封闭条件（是某些OD对的必经之路）:")
                for edge_id in invalid_edges:
                    print(f"   {edge_id}")
        
        if not valid_edges:
            print("\n❌ 没有符合条件的道路可以封闭")
            return
        
        # ✅ 新增：从rou.xml解析并排除所有车辆的起始边
        try:
            # 从rou.xml文件获取所有起始边
            depart_edges = self.get_all_depart_edges_from_rou()
            
            # 过滤掉起始边
            edges_before_filter = len(valid_edges)
            valid_edges = [e for e in valid_edges if e not in depart_edges]
            
            if edges_before_filter > len(valid_edges):
                excluded_count = edges_before_filter - len(valid_edges)
                print(f"\n⚠ 排除了 {excluded_count} 条道路（是车辆起始边）")
                excluded_edges = [e for e in edge_ids if e in depart_edges]
                for edge in excluded_edges[:5]:  # 只显示前5个
                    print(f"   - {edge}")
                if len(excluded_edges) > 5:
                    print(f"   ... 以及其他 {len(excluded_edges)-5} 条边")
            
            if not valid_edges:
                print("\n❌ 所有道路都是车辆起始边，无法封闭")
                return
                
        except Exception as e:
            print(f"⚠ 排除起始边时出错: {e}")


        # 2. 找出受影响的车辆（包括已上路和待出发）
        affected_vehicles = set()
        try:
            # 2.1 已上路的车辆
            all_vehicles = self.eng.vehicle.getIDList()
            for veh_id in all_vehicles:
                try:
                    remaining_route = self.eng.vehicle.getRoute(veh_id)
                    if any(edge in remaining_route for edge in valid_edges):
                        affected_vehicles.add(veh_id)
                except Exception as e:
                    pass
            
            print(f"\n📊 已上路：找到 {len(affected_vehicles)}/{len(all_vehicles)} 辆车的路线包含将要封闭的道路")
            
        except Exception as e:
            print(f"⚠ 查找受影响车辆时出错: {e}")
        
        # 3. 使用 setDisallowed 封闭道路
        closed_count = 0
        for edge_id in valid_edges:
            try:
                # self.eng.edge.setDisallowed(edge_id, ["all"])
                # print(f"✓ 道路 {edge_id} 已封闭")
                # ✅ 设置极高的通行代价，而不是物理禁止
                self.eng.edge.setEffort(edge_id, float('inf'))
                
                # 可选：同时设置 traveltime 为极大值（某些路由算法会用到）
                self.eng.edge.adaptTraveltime(edge_id, float('inf'))
                
                print(f"✓ 道路 {edge_id} 已设置为极高代价（逻辑封闭）")
                closed_count += 1
            except Exception as e:
                print(f"✗ 无法封闭道路 {edge_id}: {e}")
        
        print(f"\n🚧 总共成功封闭了 {closed_count}/{len(valid_edges)} 条道路")
        
        # 4. 让受影响的车辆重新计算路线
        if affected_vehicles:
            rerouted_count = 0
            failed_count = 0
            
            for veh_id in affected_vehicles:
                try:
                    # self.eng.vehicle.reroute(veh_id)
                    # 使用 rerouteTraveltime 以考虑 effort 值
                    self.eng.vehicle.rerouteTraveltime(veh_id)
                    rerouted_count += 1
                except Exception as e:
                    failed_count += 1
            print(f"✓ 成功为 {rerouted_count} 辆受影响车辆重新规划路线")
            if failed_count > 0:
                print(f"⚠ {failed_count} 辆车重新规划时遇到问题（但不会卡住，会使用原路线）")
        else:
            print(f"✓ 当前无车辆受影响")

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
        self.decision_interval=world.get_decision_interval() # 决策间隔从配置文件中读取
        # links and phase information of certain intersection
        # self.current_phase = 0
        self.steps_since_action = 0  # ✅ 添加：追踪自上次决策以来的步数
        self.time_since_last_phase_change = 0
        self.is_yellow = False
        self.next_action_time = 300 # warmup time
        # self.yellow_phase_time = min([i.duration for i in self.eng.trafficlight.getAllProgramLogics(self.id)[0].phases])
        self.yellow_phase_time = world.get_yellow_length()
        self.min_green = world.get_min_green()
        self.action_count = 0  # 🔑 跟踪实际决策次数


        # 🔑 从预解析的静态信息中获取（已在 parse_sumo_config.__init__ 中完成）
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
        
        # TODO: 以下注释代码可考虑静态获取
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
        # TODO: check .signals .full_observation .last_stet_vehicles need to be set or not
        
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
        self.action_count = 0  # 🔑 重置决策计数
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
        return self.world_current_time() == self.next_action_time

    def pseudo_step(self, action):
        '''
        pseudo_step
        Take relative actions and calculate time duration of current phase.
        
        :param action: the changes to take
        :return: None
        '''
        self.action_count += 1  # 🔑 每次决策时计数
        # 执行action后重置累积器
        self.accumulated_reward_since_last_action = 0
        self.steps_since_action = 0  # ✅ 添加：重置步数
        
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_phase_time: #接管预热阶段黄灯相位
            self.eng.trafficlight.setRedYellowGreenState(self.id, self.all_phases[action].state)
            self.next_action_time = self.world_current_time() + self.decision_interval
            self.is_yellow = False
            self.time_since_last_phase_change = 0
            
            self.green_phase = action
            return action

        elif self.green_phase == action or self.time_since_last_phase_change < self.min_green:
            self.eng.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.world_current_time() + self.decision_interval
            return self.green_phase

        elif self.green_phase != action:
            self.eng.trafficlight.setRedYellowGreenState(self.id,self.all_phases[self.yellow_dict[(self.green_phase, action)]].state)
            self.green_phase = action
            # ✅ 根据模式设置下次决策时间
            if self.world.sync_mode:
                # 同步模式：只加决策间隔（忽略黄灯时间）
                self.next_action_time = self.world_current_time() + self.decision_interval
            else:
                self.next_action_time = self.world_current_time() + self.yellow_phase_time + self.decision_interval
            self.is_yellow = True
            self.time_since_last_phase_change = 0
            return action

    #==========================original objective traffic state statistics===========================
    def collect_objective_traffic_state(self, max_distance, step_length=1):
        '''
        # TODO: para step_length can be discarded
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
            # 🔧 初始化当前车道的等待时间字典
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

                # 🔧 修正1: 基于当前速度判断是否正在等待（而非累积的waiting time）
                is_waiting = v_measures['speed'] < 0.1  # 速度阈值 0.1 m/s
                
                # 🔧 修正2: 维护当前车道上每辆车的等待时间
                if v not in self.lane_vehicle_waiting_times[lane]:
                    self.lane_vehicle_waiting_times[lane][v] = 0.0
                
                if is_waiting:
                    # 车辆正在等待，累加等待时间
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
            # 🔧 清理已离开当前车道的车辆
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
        # # TODO: reduce complexity -> find all vehicles within max_distance and on this lane
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
        优化版：直接使用车道位置信息筛选车辆
        
        :param lane: lane id
        :param max_distance: 检测距离范围
        :return detectable: 在检测范围内的车辆列表
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
# TODO: reward compute, 动作空间，观测空间放哪（也在intersection层）？如何做通用多智能体环境？奖励计算还是归结到intersection层
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
        """返回自上次action以来的平均奖励"""
        if self.steps_since_action > 0:
            avg_reward = self.accumulated_reward_since_last_action / self.steps_since_action
        else:
            avg_reward = 0.0  # 避免除以零
        return avg_reward  # ✅ 返回平均值，不重置（在pseudo_step中重置）

    def accumulate_reward(self):
        """在每个仿真步调用，累积瞬时奖励"""
        instant_reward = self.Rewards.compute_reward()
        self.accumulated_reward_since_last_action += instant_reward
        self.steps_since_action += 1  # ✅ 添加：累加步数
