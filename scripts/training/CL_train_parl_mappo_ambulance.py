"""
MAPPO training script for emergency vehicle priority traffic signal control.

Observation / reward aligned with train_project1_dqn.py: - HAVEN'T' UPDATED THIS STRING FOR FYP
  - Observation : 68-dim project1_std vector
      [phase_onehot(N)  +  12 lanes × 5 features]
      per-lane features: vehicle_count/17, avg_wait/100, wait_std/100 ⭐,
                         emg_max_wait/100, downstream_congestion
  - Reward      : 50 - [(reg_mean + K*reg_std) + Z*(emg_mean + K*emg_std)]

Both are activated by passing algorithm_name="project1_std_dqn" to the
environment, which routes Observations.py and Rewards.py to the project1
implementations.
"""

import os
import sys
import random
import numpy as np
import argparse
import yaml
import json
import time
import torch

# for plotting reward changes each episode
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Force working directory to project root (two levels up)
# ---------------------------------------------------------
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
os.chdir(project_root)        # change Python working directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # add root to Python path

print(f"Working directory set to: {os.getcwd()}")

from src.agents.MAPPOagent import MAPPOAgent
from src.core.parlenv import PARLSumoEnv
from src.core.Rewards import GetRewards
from src.core.Observations import FYP_OBS_CONFIG, get_fyp_observation_dims, print_fyp_obs_config


# ============================================================================
# Tee: mirror stdout/stderr to both terminal and log file
# ============================================================================

class Tee:
    """Write to multiple streams simultaneously (terminal + file)."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ============================================================================
# Episode runner
# ============================================================================


#def run_episode(env, agent, train=True):
def run_episode(env, agent, obs_dim, train=True):
    """Run one episode and return metrics including EMV-aware reward stats."""
    obs = env.reset()
    agent.reset()
    agent_ids = env.get_agent_ids()
    n_agents = len(agent_ids)
    n_objectives = agent.n_objectives  # NEW FOR LEXICOGRAPHIC MORL
    total_reward = {agent_id: np.zeros(n_objectives) for agent_id in agent_ids}
    steps = 0

    # Accumulators for reward-component stats across the episode
    ep_reg_mean_sum = 0.0
    ep_reg_std_sum  = 0.0
    ep_emg_mean_sum = 0.0
    ep_emg_std_sum  = 0.0
    stat_steps = 0

    while True:
        steps += 1

        obs_list = [obs[aid] for aid in agent_ids]
        actions  = agent.select_action(obs_list, deterministic=not train)
        action_dict = {agent_id: actions[i] for i, agent_id in enumerate(agent_ids)}
        next_obs, reward_dict, done, info = env.step(action_dict)

        # NEW FOR LEXICOGRAPHIC MORL: reward_dict values are now always
        # arrays of shape (n_objectives,) (parlenv.py wraps legacy scalar
        # rewards as shape-(1,) arrays), so the fallback for a missing
        # agent must match that shape rather than a bare 0.0.
        if agent.args['common_reward']:
            common_reward = sum(reward_dict.values())
            rewards = np.array([common_reward] * n_agents)
        else:
            rewards = np.array([reward_dict.get(aid, np.zeros(n_objectives)) for aid in agent_ids])

        if train:
            agent.store_transition(obs_list, actions, rewards)

        for agent_id, r in reward_dict.items():
            total_reward[agent_id] += r

        # Collect per-step reward statistics from the first intersection
        try:
            world = env.env
            first_ts = world.id2intersection[agent_ids[0]]
            stats = first_ts.Rewards.get_reward_statistics()
            ep_reg_mean_sum += stats['regular_vehicles']['mean_waiting']
            ep_reg_std_sum  += stats['regular_vehicles']['std_waiting']
            ep_emg_mean_sum += stats['emergency_vehicles']['mean_waiting']
            ep_emg_std_sum  += stats['emergency_vehicles']['std_waiting']
            stat_steps += 1
        except Exception:
            pass

        obs = next_obs
        if done:
            break

    if train:
        #final_obs_list = [obs[aid] for aid in agent_ids]
        final_obs_list = [obs.get(aid, np.zeros(obs_dim, dtype=np.float32)) for aid in agent_ids]

        agent.finish_episode(final_obs_list)
        train_stats = agent.update()
    else:
        train_stats = {}

    # Episode-level averages of reward statistics - NEW: this is actually just the evaluation metrics now
    if stat_steps > 0:
        reward_stats = {
            'reg_waiting_mean': ep_reg_mean_sum / stat_steps,
            'reg_waiting_std':  ep_reg_std_sum  / stat_steps,
            'emg_waiting_mean': ep_emg_mean_sum / stat_steps,
            'emg_waiting_std':  ep_emg_std_sum  / stat_steps,
        }
    else:
        reward_stats = {
            'reg_waiting_mean': 0.0,
            'reg_waiting_std':  0.0,
            'emg_waiting_mean': 0.0,
            'emg_waiting_std':  0.0,
        }

    world = env.env
    ambulance_trip_times  = [t for vid, t in world.vehicles_trip_time.items()
                             if vid.startswith("ambulance_")]
    ambulance_duration    = float(np.mean(ambulance_trip_times)) if ambulance_trip_times else 0.0
    civilian_trip_times   = [t for vid, t in world.vehicles_trip_time.items()
                             if not vid.startswith("ambulance_")]
    civilian_avg_trip_time = np.mean(civilian_trip_times) if civilian_trip_times else 0.0

    return total_reward, steps, train_stats, ambulance_duration, civilian_avg_trip_time, reward_stats


# ============================================================================
# SUMO config builder
# ============================================================================

def create_sumo_config(scenario_dir, config_dir, gui=False): # Set gui=False for fast training, Warning: Libsumo on Windows does not work with GUI, falling back to plain libsumo.
    """Create the SUMO config JSON file for the experiment."""
    os.makedirs(config_dir, exist_ok=True)

    config = {
        "name": "emergency_mappo_ambulance",
        "dir": scenario_dir,
        "roadnetFile": "3_intersection_corridor_250long.net.xml", # Change if changing network
        # NEW FOR FYP: initial demand combo for the very first reset(), before
        # the curriculum's set_demand_files() calls take over each episode.
        # NOTE: "combined_file" removed on purpose - see set_demand_files()
        # in env.py / parlenv.py. A single .sumocfg can only point at one
        # fixed regular+EV combination, which doesn't work once you have
        # multiple demand pools to sample from per episode.
        "flowFile": "vtypes.rou.xml,demand_regular/reg_1800_v1.rou.xml,demand_ev/ev_100s_v1.rou.xml",
        "gui": gui, # Set "gui" = gui, Warning: Libsumo on Windows does not work with GUI, falling back to plain libsumo.
        "no_warning": True,
        "decision_interval": 5,
        "min_green": 5,
        "yellow_length": 4,
    }

    config_path = os.path.join(config_dir, 'sumo_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


# ============================================================================
# NEW FOR FYP: Curriculum / demand-pool config
# ============================================================================
#
# Each stage is (last_episode_of_stage, regular_pool, ev_pool). Stages are
# checked in order; the first stage whose last_episode >= current episode
# is used, so the final stage's last_episode should be >= max_episodes.
# Every episode, one file is sampled uniformly at random from that stage's
# regular_pool and one from its ev_pool - this is what gives you within-stage
# variability (your "2-3 versions of each demand level" idea), on top of the
# across-stage curriculum progression.
#
# Fill in your actual generated filenames here once you've built the pools.
# Paths are relative to scenario_dir, matching flowFile above.

# TODO: Fill this out with the demand files for the scenario - and set episode lengths for each segment of training
CURRICULUM = [
    # Stage 1: low regular demand, dense EVs - learn EV response mechanics
    # fast, without heavy background congestion complicating credit assignment.
    {
        "last_episode": 20,
        "regular_pool": [
            "demand_regular/reg_900_v1.rou.xml",
            "demand_regular/reg_900_v2.rou.xml",
        ],
        "ev_pool": [
            "demand_ev/ev_50s_v1.rou.xml",
            "demand_ev/ev_50s_v2.rou.xml",
        ],
    },
    # Stage 2: ramp regular demand up, ease EV frequency back down towards
    # realistic.
    {
        "last_episode": 160,
        "regular_pool": [
            "demand_regular/reg_1400_v1.rou.xml",
            "demand_regular/reg_1400_v2.rou.xml",
            
            "demand_regular/reg_1600_v1.rou.xml",
            "demand_regular/reg_1600_v2.rou.xml",
            
            "demand_regular/reg_1800_v1.rou.xml",
            "demand_regular/reg_1800_v2.rou.xml",
            "demand_regular/reg_1800_v3.rou.xml",
        ],
        "ev_pool": [
            "demand_ev/ev_50s_v2.rou.xml",
            
            "demand_ev/ev_100s_v1.rou.xml",
            "demand_ev/ev_100s_v2.rou.xml",
            "demand_ev/ev_100s_v3.rou.xml",
        ],
    },
    # Stage 3 (final): realistic demand + realistic EV frequency - this
    # should match whatever distribution you actually evaluate/deploy at,
    # so the policy ends training calibrated to it (this is what the 2x
    # experiment's train/test mismatch was missing).
    {
        "last_episode": 200,
        "regular_pool": [
            "demand_regular/reg_1800_v1.rou.xml",
            "demand_regular/reg_1800_v2.rou.xml",
            "demand_regular/reg_1800_v3.rou.xml",
            
            "demand_regular/reg_2200_v1.rou.xml",
            "demand_regular/reg_2200_v2.rou.xml",
        ],
        "ev_pool": [
            "demand_ev/ev_100s_v1.rou.xml",
            "demand_ev/ev_100s_v2.rou.xml",
            "demand_ev/ev_100s_v3.rou.xml",
            
            "demand_ev/ev_150s_v1.rou.xml",
            "demand_ev/ev_150s_v2.rou.xml",
        ],
    },
]


def sample_episode_demand(episode: int, curriculum=CURRICULUM, rng=None):
    """
    Pick (regular_file, ev_file) for this episode from the curriculum.

    Args:
        episode: current training episode number (1-indexed, matches the
            `episode` loop variable in the training loop below).
        curriculum: list of stage dicts, see CURRICULUM above.
        rng: optional random.Random instance for reproducible sampling
            (pass e.g. random.Random(seed + episode) if you want the choice
            itself to be reproducible per episode).

    Returns:
        (regular_file, ev_file) tuple, both relative to scenario_dir.
    """
    rng = rng or random
    for stage in curriculum:
        if episode <= stage["last_episode"]:
            reg_file = rng.choice(stage["regular_pool"])
            ev_file = rng.choice(stage["ev_pool"])
            return reg_file, ev_file
    # Should be unreachable if the last stage's last_episode is inf, but
    # fall back to the last stage just in case.
    last = curriculum[-1]
    return rng.choice(last["regular_pool"]), rng.choice(last["ev_pool"])


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train MAPPO agent with project1-style EMV-aware obs+reward')

    # Config / scenario
    parser.add_argument('--config', type=str, default='mappo_ambulance', # TODO: Use mappo_ambulance for BASELINE, mappo_fyp_config for FYP and mappo_fyp_lexicographic_config for lexicographic - from configs/tsc folder
                        help='Config name in configs/tsc/ (baseline is: mappo_ambulance)')
    parser.add_argument('--scenario-dir', type=str,
                        default='scenarios/3_intersection_corridor_250long', # TODO: change this when switching to a new scenario/network
                        help='SUMO scenario directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed') # normally we just use seed 42 for everything

    # Project-1 reward parameters (K and Z)
    parser.add_argument('--K', type=float, default=None,
                        help='Std-penalty weight in reward formula '
                             '(default: read from YAML ambulance.K, fallback 0.5)')
    parser.add_argument('--Z', type=float, default=None,
                        help='EMV penalty multiplier in reward formula '
                             '(default: read from YAML ambulance.Z, fallback 3.0)')

    # Training control
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Override config training.max_episodes')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Override config training.save_interval')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pretrained model to continue training')
    parser.add_argument('--start-episode', type=int, default=1,
                        help='Starting episode number (for resuming training)')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load YAML config
    # ------------------------------------------------------------------
    config_path = f'configs/tsc/{args.config}.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve K and Z: CLI > YAML > hardcoded default
    K = args.K if args.K is not None else config.get('algorithm', {}).get('ambulance', {}).get('K', 0.5) # to be safe - the fallback defaults should match the config being used
    Z = args.Z if args.Z is not None else config.get('algorithm', {}).get('ambulance', {}).get('Z', 3.0) # to be safe - the fallback defaults should match the config being used

    # Inject K and Z into the reward function's class-level config so that
    # _compute_project1_std_reward() picks them up at runtime.
    GetRewards.REWARD_CONFIGS['project1_std_reward']['K'] = K
    GetRewards.REWARD_CONFIGS['project1_std_reward']['Z'] = Z

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Experiment directory
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name  = f"mappo_ambulance_K{K}_Z{Z}_seed{seed}_{timestamp}"
    exp_dir   = os.path.join('experiments', exp_name)

    model_save_dir   = os.path.join(exp_dir, 'models')
    log_save_dir     = os.path.join(exp_dir, 'logs')
    configs_save_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(model_save_dir,   exist_ok=True)
    os.makedirs(log_save_dir,     exist_ok=True)
    os.makedirs(configs_save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Redirect stdout / stderr to log file
    # ------------------------------------------------------------------
    console_log_path = os.path.join(log_save_dir, 'console_output.log')
    log_file_handle  = open(console_log_path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = Tee(sys.stdout, log_file_handle)
    sys.stderr = Tee(sys.stderr, log_file_handle)

    print(f"\n{'='*70}")
    print("MAPPO-Ambulance Training - EMV Priority (project1-std obs+reward)")
    print(f"{'='*70}")
    print(f"Experiment dir : {exp_dir}")
    print("   |-- models/                (model checkpoints)")
    print("   |-- logs/")
    print("   |   |-- console_output.log (terminal output)")
    print("   |   |-- training.json      (per-episode metrics)")
    print("   |   `-- final_results.json (final summary)")
    print("   `-- configs/               (SUMO config)")
    print("Console log    : {console_log_path}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Build SUMO config
    # ------------------------------------------------------------------
    sumo_config_path = create_sumo_config(args.scenario_dir, configs_save_dir, gui=False) # set GUI to false for fast training - Warning: Libsumo on Windows does not work with GUI, falling back to plain libsumo.
    print(f"SUMO config    : {sumo_config_path}\n")

    # ------------------------------------------------------------------
    # Environment config
    # Key: algorithm_name = "project1_std_dqn" activates
    #      _compute_project1_std_observation()  (68-dim)
    #      _compute_project1_std_reward()        (std-aware EMV formula)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Environment config
    # Key: algorithm_name = "project1_std_dqn" activates
    #      _compute_project1_std_observation()  (68-dim)
    #      _compute_project1_std_reward()        (std-aware EMV formula)
    #
    # NEW FOR LEXICOGRAPHIC MORL: set config['algorithm']['lexicographic']
    # ['enabled'] = true to switch the reward path to
    # GetRewards._compute_lexicographic_reward_vector() (r1=EV, r2=regular)
    # and the agent to the Lagrangian LPPO actor loss. Observation space is
    # untouched either way - only reward/critic/actor change.
    # ------------------------------------------------------------------
    lex_cfg = config['algorithm'].get('lexicographic', {})
    lex_enabled = lex_cfg.get('enabled', False)
    n_objectives = lex_cfg.get('n_objectives', 2) if lex_enabled else 1

    base_algorithm_name = "final_year_project" # TODO: IMPORTANT: Sets what Observation state space is being used - change this to be fyp or final_year_project for FYP model (final_year_project_lane_mode for LANE FEATURES VERSION) OR project1_std_dqn for baseline model
    env_config = {
        "sumo_config":           os.path.abspath(sumo_config_path),
        "interface":             config.get('environment', {}).get('interface', 'libsumo'),
        "seed":                  seed,
        "sync_mode":             config.get('environment', {}).get('sync_mode', True),
        # These subscriptions satisfy the project1 reward path; the obs path
        # ignores them and queries SUMO directly for per-vehicle waiting times.
        "obs_to_subscribe":      config['algorithm']['observation']['obs_to_subscribe'],
        "reward_to_subscribe":   config['algorithm']['reward']['reward_to_subscribe'],
        # Activates project1-style observation AND reward routing.
        # Appending "_lexicographic" (rather than swapping the name outright)
        # keeps the FYP observation-space dispatch in Observations.py intact
        # (it string-matches 'final_year_project'/'fyp') while separately
        # triggering the new reward path in Rewards.py (it string-matches
        # 'lexicographic'/'lmorl' FIRST, checked ahead of the fyp branch).
        "algorithm_name":        base_algorithm_name + ("_lexicographic" if lex_enabled else ""),
        "normalize_observation": config['algorithm']['observation'].get('normalize', False),
        "norm_params":           config['algorithm']['observation'].get('norm_params', {}),
        "reward_weights":        config['algorithm']['reward'].get('reward_weights', [1.0]),
        "reward_scale":          config['algorithm']['reward'].get('scale', 1.0),
        "reward_clip_range":     config['algorithm']['reward'].get('clip_range', None),
    }

    env = PARLSumoEnv(env_config)

    agent_ids = env.get_agent_ids()
    n_agents  = len(agent_ids)
    obs_dim   = env.observation_space(agent_ids[0]).shape[0]
    act_dim   = env.action_space(agent_ids[0]).n

    print("Environment:")
    print(f"   Agent IDs : {agent_ids}")
    print(f"   N Agents  : {n_agents}")
    print(f"   Obs Dim   : {obs_dim}  (varies depending what was used)")
    print(f"   Act Dim   : {act_dim}")
    print("\nEMV reward parameters:")
    print(f"   K (std weight)         : {K}")
    print(f"   Z (EMV penalty mult.)  : {Z}")
    print("   Formula: reward = depends what is used")
    
    algorithm_name = env_config["algorithm_name"]
    print("=" * 70)

    # NEW: fail loudly here instead of silently falling through to the
    # wrong observation branch deep in Observations.py. base_algorithm_name
    # is the OBSERVATION-SPACE selector only - it must be exactly one of
    # these three. lexicographic.enabled appends "_lexicographic" onto
    # whichever of these you pick, automatically - do not put
    # "lexicographic" in base_algorithm_name itself.
    _valid_base_names = ("project1_std_dqn", "final_year_project", "final_year_project_lane_mode")
    if base_algorithm_name not in _valid_base_names:
        raise ValueError(
            f"base_algorithm_name = {base_algorithm_name!r} is not a recognised observation-space "
            f"selector - it must be exactly one of {_valid_base_names}. "
            f"lexicographic.enabled in the YAML appends '_lexicographic' onto this automatically; "
            f"you should not put 'lexicographic' here yourself. "
            f"(Full algorithm_name that would have been sent to the environment: {algorithm_name!r})"
        )

    # NOTE: these checks are against base_algorithm_name (pre-suffix), not
    # algorithm_name (post-suffix) - when lexicographic mode is on,
    # algorithm_name is e.g. "final_year_project_lexicographic", which will
    # never == "final_year_project" exactly.
    if base_algorithm_name == "project1_std_dqn":
        print("\nUsing BASELINE RL (Kodogoda-style) observation/state space\n")
        
    elif base_algorithm_name == "final_year_project":
        print("\nUsing FINAL YEAR PROJECT observation/state space\n")
        print_fyp_obs_config()
    elif base_algorithm_name == "final_year_project_lane_mode":
        print("\nUsing FINAL YEAR PROJECT !LANE! VERSION observation/state space\n")
    if lex_enabled:
        print(f"\nLEXICOGRAPHIC MORL ENABLED: n_objectives={n_objectives} "
              f"(r1=EV priority [system-wide], r2=regular vehicles [local])")
        print(f"   tolerance={lex_cfg.get('tolerance', 0.0)}  "
              f"dual_lr={lex_cfg.get('dual_lr', 0.05)}  "
              f"ema_rho={lex_cfg.get('ema_rho', 0.05)}  "
              f"base_weight_decay={lex_cfg.get('base_weight_decay', 0.1)}\n")
    print("=" * 70)

    # ------------------------------------------------------------------
    # MAPPO agent
    # ------------------------------------------------------------------
    device = 'cuda' if config.get('use_cuda', False) and torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    agent = MAPPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_agents=n_agents,
        lr=config['algorithm']['lr'],
        gamma=config['algorithm']['gamma'],
        eps_clip=config['algorithm']['ppo']['eps_clip'],
        epochs=config['algorithm']['ppo']['epochs'],
        hidden_dim=config['model']['network']['hidden_dim'],
        use_rnn=config['model']['network']['use_rnn'],
        entropy_coef=config['algorithm']['ppo']['entropy_coef'],
        q_nstep=config['algorithm']['ppo'].get('q_nstep', 5),
        max_grad_norm=config['algorithm']['ppo']['max_grad_norm'],
        device=device,
        standardise_rewards=True,
        standardise_returns=False,
        obs_agent_id=True,
        obs_last_action=False,
        target_update_interval_or_tau=0.01,
        common_reward=config['algorithm'].get('common_reward', False),
        # NEW FOR LEXICOGRAPHIC MORL
        n_objectives=n_objectives,
        lexicographic=lex_enabled,
        lex_tolerance=lex_cfg.get('tolerance', 0.0),
        lex_dual_lr=lex_cfg.get('dual_lr', 0.05),
        lex_ema_rho=lex_cfg.get('ema_rho', 0.05),
        lex_base_weight_decay=lex_cfg.get('base_weight_decay', 0.1),
    )

    if args.load_model:
        if os.path.exists(args.load_model):
            agent.load(args.load_model)
            print(f"\nLoaded pretrained model: {args.load_model}")
            print(f"   Resuming from episode {args.start_episode}\n")
        else:
            raise FileNotFoundError(f"Pretrained model not found: {args.load_model}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    max_episodes  = args.max_episodes  if args.max_episodes  is not None else config['training']['max_episodes']
    save_interval = args.save_interval if args.save_interval is not None else config['training']['save_interval']
    start_episode = args.start_episode if args.load_model else 1

    training_log        = []
    training_start_time = time.time()

    print(f"\n{'='*60}")
    if args.load_model:
        print(f"Resuming training: episodes {start_episode} to {max_episodes}")
    else:
        print(f"Starting training: {max_episodes} episodes")
    print(f"   Save interval : every {save_interval} episodes")
    print(f"{'='*60}\n")

    for episode in range(start_episode, max_episodes + 1):
        # NEW FOR FYP: curriculum training - pick this episode's regular +
        # EV demand files from the current curriculum stage, and vary the
        # SUMO seed too. Both setters must be called before reset() (which
        # happens as the first line of run_episode()) since SUMO only reads
        # its config/route files at process startup.
        reg_file, ev_file = sample_episode_demand(episode, rng=random.Random(seed + episode))        
        env.set_demand_files(reg_file, ev_file)
        env.set_seed(seed + episode)

        (total_reward, steps, train_stats,
         ambulance_duration, civilian_avg_trip_time,
         reward_stats) = run_episode(env, agent, obs_dim, train=True)

        # NEW FOR LEXICOGRAPHIC MORL: total_reward[aid] is now always an
        # array of shape (n_objectives,). avg_reward_per_objective[i] is
        # this episode's mean (across agents) return for objective i;
        # avg_reward keeps its old meaning (objective 0) for anything that
        # only expects a single number.
        avg_reward_per_objective = np.mean([total_reward[aid] for aid in agent_ids], axis=0)
        avg_reward = float(avg_reward_per_objective[0])

        log_entry = {
            'episode':              episode,
            'steps':                steps,
            # NEW FOR FYP: which demand files this episode used, for
            # correlating reward/behavior with curriculum stage later.
            'regular_demand_file':  reg_file,
            'ev_demand_file':       ev_file,
            'avg_reward':           avg_reward,
            'avg_reward_per_objective': [float(x) for x in avg_reward_per_objective],
            'actor_loss':           float(train_stats.get('actor_loss',
                                          train_stats.get('pg_loss', 0))),
            'critic_loss':          float(train_stats.get('critic_loss', 0)),
            'entropy':              float(train_stats.get('entropy', 0)),
            'td_error_abs':         float(train_stats.get('td_error_abs', 0)),
            'ambulance_duration':   float(ambulance_duration),
            'civilian_avg_trip_time': float(civilian_avg_trip_time),
            # project1-style reward stats
            'reg_waiting_mean':     float(reward_stats['reg_waiting_mean']),
            'reg_waiting_std':      float(reward_stats['reg_waiting_std']),
            'emg_waiting_mean':     float(reward_stats['emg_waiting_mean']),
            'emg_waiting_std':      float(reward_stats['emg_waiting_std']),
        }
        # per-objective advantage means, and (if lexicographic) the dual
        # variables - whatever MAPPOLearner.train() actually logged.
        for key, val in train_stats.items():
            if key.startswith('advantage_mean_r') or key.startswith('lex_'):
                log_entry[key] = float(val)
        training_log.append(log_entry)

        # NEW FOR LEXICOGRAPHIC MORL: show each objective's own average
        # reward separately (r1=EV, r2=regular, ...) instead of one number,
        # so you can see "this run got X reward for EVs, Y for regular
        # vehicles" directly in the console / training.json.
        if agent.args.get('lexicographic', False):
            reward_display = " | ".join(
                f"r{i+1}={avg_reward_per_objective[i]:7.2f}" for i in range(agent.n_objectives)
            )
        else:
            reward_display = f"AvgReward={avg_reward:7.2f}"

        lex_suffix = ""
        if agent.args.get('lexicographic', False):
            lam_str = ", ".join(f"lam_r{i+1}={train_stats.get(f'lex_lambda_r{i+1}', 0):.3f}"
                                 for i in range(agent.n_objectives - 1))
            lex_suffix = f" | {lam_str}"

        print(f"Episode {episode:4d}/{max_episodes} | "
              f"Steps={steps:4d} | {reward_display} | "
              f"EMV={ambulance_duration:.1f}s | Civilian={civilian_avg_trip_time:.1f}s | "
              f"RegWait={reward_stats['reg_waiting_mean']:.2f}s "
              f"(std={reward_stats['reg_waiting_std']:.2f}) | "
              f"EmgWait={reward_stats['emg_waiting_mean']:.2f}s | "
              f"ActorL={log_entry['actor_loss']:.4f} | "
              f"CriticL={log_entry['critic_loss']:.4f} | "
              f"Entropy={log_entry['entropy']:.4f}{lex_suffix}")

        if episode % save_interval == 0:
            save_path = os.path.join(model_save_dir, f"agent_episode_{episode}.pt")
            agent.save(save_path)

            log_file = os.path.join(log_save_dir, "training.json")
            with open(log_file, 'w') as f:
                json.dump(training_log, f, indent=2)

            print(f"   [Saved] model + log at episode {episode}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_model_path = os.path.join(model_save_dir, "agent_final.pt")
    agent.save(final_model_path)

    # NEW FOR LEXICOGRAPHIC MORL: how often (and how strongly) did each
    # constrained objective's Lagrange multiplier actually engage over the
    # whole run? "Final lambda" alone is close to useless - it's usually 0
    # at the end regardless, since lambda decays back down once a
    # regression is corrected. This instead scans every logged episode.
    lex_engagement_summary = {}
    if agent.args.get('lexicographic', False):
        for i in range(agent.n_objectives - 1):
            key = f'lex_lambda_r{i+1}'
            vals = [entry.get(key, 0.0) for entry in training_log]
            nonzero_count = sum(1 for v in vals if v > 0.0)
            peak_val = max(vals) if vals else 0.0
            peak_episode = training_log[vals.index(peak_val)]['episode'] if vals else None
            lex_engagement_summary[key] = {
                'nonzero_episode_count': nonzero_count,
                'total_episodes':        len(vals),
                'peak_value':            peak_val,
                'peak_episode':          peak_episode,
            }

    final_results = {
        "algorithm":         "MAPPO-Ambulance (project1-std obs+reward)",
        "K":                 K,
        "Z":                 Z,
        "training_episodes": max_episodes,
        "start_episode":     start_episode,
        "episodes_trained":  max_episodes - start_episode + 1,
        "total_time":        time.time() - training_start_time,
        "final_avg_reward":  float(avg_reward),
        "final_avg_reward_per_objective": [float(x) for x in avg_reward_per_objective],  # NEW FOR LEXICOGRAPHIC MORL
        "lexicographic_engagement_summary": lex_engagement_summary,  # NEW FOR LEXICOGRAPHIC MORL
        "loaded_from":       args.load_model if args.load_model else None,
        "training_log":      training_log,
    }

    final_results_file = os.path.join(log_save_dir, "final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    exp_config = {
        'args':              vars(args),
        'timestamp':         timestamp,
        'seed':              seed,
        'K':                 K,
        'Z':                 Z,
        'config_path':       config_path,
        'exp_dir':           exp_dir,
        'algorithm':         'MAPPO-Ambulance', # I dont think this needs to be renamed? don't think its used for any if statements - only algorithm_name at line ~300 is 100% used and important (normally I jsut leave this as 'MAPPO-Ambulance')
        'algorithm_name_env': 'final_year_project', # TODO: MAY? affect what Observation state space is being used (not 100% sure) - change this to be fyp or final_year_project for FYP model (or final_year_project_lane_mode for LANE FEATURES version) OR project1_std_dqn for baseline model
        'obs_dim':           obs_dim,
        'pretrained_model':  args.load_model if args.load_model else None,
        'start_episode':     start_episode,
        'max_episodes':      max_episodes,
        # Snapshot of the observation feature toggles used for this run
        # (only meaningful when algorithm_name == "final_year_project")
        'fyp_obs_config':    FYP_OBS_CONFIG,
        'fyp_obs_dims':      dict(zip(('intersection_dim', 'per_lane_dim'), get_fyp_observation_dims())),
        # NEW FOR LEXICOGRAPHIC MORL: snapshot of the lexicographic config
        # used for this run, plus the final Lagrangian dual state, so a run
        # can be inspected/resumed with full knowledge of what it was doing.
        'lexicographic_enabled': lex_enabled,
        'lexicographic_config':  lex_cfg,
        'lexicographic_final_lambda': getattr(agent.learner, 'lex_lambda', None),
        'lexicographic_final_k_ema':  getattr(agent.learner, 'lex_k_ema', None),
        'lexicographic_engagement_summary': lex_engagement_summary,  # NEW FOR LEXICOGRAPHIC MORL
    }
    with open(os.path.join(exp_dir, 'exp_config.json'), 'w') as f:
        json.dump(exp_config, f, indent=2)

    elapsed_h = (time.time() - training_start_time) / 3600
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    print(f"Experiment dir : {exp_dir}")
    print("   |-- models/agent_final.pt       (final model)")
    print("   |-- logs/training.json          (per-episode metrics)")
    print("   |-- logs/final_results.json     (final summary)")
    print("   |-- logs/console_output.log     (full console output)")
    print("   `-- exp_config.json             (experiment config)")
    print("\nPerformance summary:")
    if args.load_model:
        print(f"   Pretrained model  : {args.load_model}")
        print(f"   Episodes trained  : {start_episode} - {max_episodes} "
              f"({max_episodes - start_episode + 1} episodes)")
    else:
        print(f"   Episodes trained  : {max_episodes}")
    print(f"   K / Z             : {K} / {Z}")
    if agent.args.get('lexicographic', False):
        obj_summary = " | ".join(
            f"r{i+1}={avg_reward_per_objective[i]:.2f}" for i in range(agent.n_objectives)
        )
        print(f"   Final avg reward  : {obj_summary}   (r1=EV, r2=regular vehicles)")
        # NEW FOR LEXICOGRAPHIC MORL: nonzero-episode count and peak value
        # per constrained objective - "final lambda" alone is close to
        # useless since it decays back to 0 once a regression is corrected.
        for i in range(agent.n_objectives - 1):
            key = f'lex_lambda_r{i+1}'
            stats = lex_engagement_summary.get(key, {})
            n_nonzero = stats.get('nonzero_episode_count', 0)
            n_total   = stats.get('total_episodes', 0)
            peak_val  = stats.get('peak_value', 0.0)
            peak_ep   = stats.get('peak_episode', None)
            print(f"   Lambda {i+1} engaged  : {n_nonzero}/{n_total} episodes "
                  f"| peak={peak_val:.4f} (episode {peak_ep})")
    else:
        print(f"   Final avg reward  : {avg_reward:.2f}")
    print(f"   Total time        : {elapsed_h:.2f} hours")
    print(f"{'='*70}\n")
    
    # ------------------------------------------------------------------
    # Plot training progress
    # ------------------------------------------------------------------
    episodes = [x['episode'] for x in training_log]

    if agent.args.get('lexicographic', False):
        # NEW FOR LEXICOGRAPHIC MORL: one subplot per objective, so you can
        # see "EV reward went up, regular-vehicle reward went down" etc.
        # at a glance instead of one blended curve.
        n_obj = agent.n_objectives
        obj_labels = ['r1 (EV priority)', 'r2 (regular vehicles)'] + \
                     [f'r{i+1}' for i in range(2, n_obj)]
        fig, axes = plt.subplots(n_obj, 1, figsize=(10, 4 * n_obj), sharex=True)
        if n_obj == 1:
            axes = [axes]
        for i in range(n_obj):
            r_i = [x['avg_reward_per_objective'][i] for x in training_log]
            axes[i].plot(episodes, r_i, label=obj_labels[i])
            window = 20
            if len(r_i) >= window:
                moving_avg = np.convolve(r_i, np.ones(window) / window, mode='valid')
                axes[i].plot(episodes[window - 1:], moving_avg,
                             label=f'{window}-Episode Moving Average', linewidth=2)
            axes[i].set_ylabel('Average Reward')
            axes[i].set_title(obj_labels[i])
            axes[i].grid(True)
            axes[i].legend()
        axes[-1].set_xlabel('Episode')
        fig.suptitle('MAPPO Training Progress - Lexicographic Objectives')
        reward_plot_path = os.path.join(log_save_dir, 'reward_curve.png')
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved per-objective reward curves to: {reward_plot_path}")
    else:
        rewards  = [x['avg_reward'] for x in training_log]

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label='Average Reward')

        # Optional moving average smoothing
        window = 20
        if len(rewards) >= window:
            moving_avg = np.convolve(
                rewards,
                np.ones(window) / window,
                mode='valid'
            )
            plt.plot(
                episodes[window - 1:],
                moving_avg,
                label=f'{window}-Episode Moving Average',
                linewidth=2
            )

        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('MAPPO Training Progress')
        plt.grid(True)
        plt.legend()

        reward_plot_path = os.path.join(log_save_dir, 'reward_curve.png')
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved reward curve to: {reward_plot_path}")

    sys.stdout.flush()
    sys.stderr.flush()
    log_file_handle.close()

    env.close()


if __name__ == '__main__':
    main()