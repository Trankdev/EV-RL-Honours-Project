"""
MAPPO training script for emergency vehicle priority traffic signal control.

Observation / reward aligned with train_project1_dqn.py:
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
import numpy as np
import argparse
import yaml
import json
import time
import torch

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
    total_reward = {agent_id: 0 for agent_id in agent_ids}
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

        if agent.args['common_reward']:
            common_reward = sum(reward_dict.values())
            rewards = np.array([common_reward] * n_agents)
        else:
            rewards = np.array([reward_dict.get(aid, 0.0) for aid in agent_ids])

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

    # Episode-level averages of reward statistics
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

def create_sumo_config(scenario_dir, config_dir, gui=False):
    """Create the SUMO config JSON file for the experiment."""
    os.makedirs(config_dir, exist_ok=True)

    config = {
        "name": "emergency_mappo_ambulance",
        "dir": scenario_dir,
        "roadnetFile": "2_intersection_corridor.net.xml",
        "flowFile": "vtypes.rou.xml,2_intersection_corridor.rou.xml,ambulance.rou.xml",
        "combined_file": "2_intersection_corridor.sumocfg",
        "gui": gui,
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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train MAPPO agent with project1-style EMV-aware obs+reward')

    # Config / scenario
    parser.add_argument('--config', type=str, default='mappo_ambulance',
                        help='Config name in configs/tsc/ (default: mappo_ambulance)')
    parser.add_argument('--scenario-dir', type=str,
                        default='scenarios/2_intersection_corridor', # change this when using different scenarios
                        help='SUMO scenario directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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
    K = args.K if args.K is not None else config.get('algorithm', {}).get('ambulance', {}).get('K', 0.5)
    Z = args.Z if args.Z is not None else config.get('algorithm', {}).get('ambulance', {}).get('Z', 3.0)

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
    print(f"MAPPO-Ambulance Training - EMV Priority (project1-std obs+reward)")
    print(f"{'='*70}")
    print(f"Experiment dir : {exp_dir}")
    print(f"   |-- models/                (model checkpoints)")
    print(f"   |-- logs/")
    print(f"   |   |-- console_output.log (terminal output)")
    print(f"   |   |-- training.json      (per-episode metrics)")
    print(f"   |   `-- final_results.json (final summary)")
    print(f"   `-- configs/               (SUMO config)")
    print(f"Console log    : {console_log_path}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Build SUMO config
    # ------------------------------------------------------------------
    sumo_config_path = create_sumo_config(args.scenario_dir, configs_save_dir, gui=False)
    print(f"SUMO config    : {sumo_config_path}\n")

    # ------------------------------------------------------------------
    # Environment config
    # Key: algorithm_name = "project1_std_dqn" activates
    #      _compute_project1_std_observation()  (68-dim)
    #      _compute_project1_std_reward()        (std-aware EMV formula)
    # ------------------------------------------------------------------
    env_config = {
        "sumo_config":           os.path.abspath(sumo_config_path),
        "interface":             config.get('environment', {}).get('interface', 'libsumo'),
        "seed":                  seed,
        "sync_mode":             config.get('environment', {}).get('sync_mode', True),
        # These subscriptions satisfy the project1 reward path; the obs path
        # ignores them and queries SUMO directly for per-vehicle waiting times.
        "obs_to_subscribe":      config['algorithm']['observation']['obs_to_subscribe'],
        "reward_to_subscribe":   config['algorithm']['reward']['reward_to_subscribe'],
        # Activates project1-style observation AND reward routing
        "algorithm_name":        "project1_std_dqn",
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

    print(f"Environment:")
    print(f"   Agent IDs : {agent_ids}")
    print(f"   N Agents  : {n_agents}")
    print(f"   Obs Dim   : {obs_dim}  (project1_std: phase_onehot + 12 lanes × 5 features)")
    print(f"   Act Dim   : {act_dim}")
    print(f"\nEMV reward parameters:")
    print(f"   K (std weight)         : {K}")
    print(f"   Z (EMV penalty mult.)  : {Z}")
    print(f"   Formula: reward = 50 - [(reg_mean + {K}*reg_std) + {Z}*(emg_mean + {K}*emg_std)]")

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
        (total_reward, steps, train_stats,
         ambulance_duration, civilian_avg_trip_time,
         reward_stats) = run_episode(env, agent, obs_dim, train=True)

        avg_reward = np.mean([total_reward[aid] for aid in agent_ids])

        log_entry = {
            'episode':              episode,
            'steps':                steps,
            'avg_reward':           float(avg_reward),
            'actor_loss':           float(train_stats.get('actor_loss',
                                          train_stats.get('pg_loss', 0))),
            'critic_loss':          float(train_stats.get('critic_loss', 0)),
            'entropy':              float(train_stats.get('entropy', 0)),
            'advantage_mean':       float(train_stats.get('advantage_mean', 0)),
            'td_error_abs':         float(train_stats.get('td_error_abs', 0)),
            'ambulance_duration':   float(ambulance_duration),
            'civilian_avg_trip_time': float(civilian_avg_trip_time),
            # project1-style reward stats
            'reg_waiting_mean':     float(reward_stats['reg_waiting_mean']),
            'reg_waiting_std':      float(reward_stats['reg_waiting_std']),
            'emg_waiting_mean':     float(reward_stats['emg_waiting_mean']),
            'emg_waiting_std':      float(reward_stats['emg_waiting_std']),
        }
        training_log.append(log_entry)

        print(f"Episode {episode:4d}/{max_episodes} | "
              f"Steps={steps:4d} | AvgReward={avg_reward:7.2f} | "
              f"EMV={ambulance_duration:.1f}s | Civilian={civilian_avg_trip_time:.1f}s | "
              f"RegWait={reward_stats['reg_waiting_mean']:.2f}s "
              f"(std={reward_stats['reg_waiting_std']:.2f}) | "
              f"EmgWait={reward_stats['emg_waiting_mean']:.2f}s | "
              f"ActorL={log_entry['actor_loss']:.4f} | "
              f"CriticL={log_entry['critic_loss']:.4f} | "
              f"Entropy={log_entry['entropy']:.4f}")

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

    final_results = {
        "algorithm":         "MAPPO-Ambulance (project1-std obs+reward)",
        "K":                 K,
        "Z":                 Z,
        "training_episodes": max_episodes,
        "start_episode":     start_episode,
        "episodes_trained":  max_episodes - start_episode + 1,
        "total_time":        time.time() - training_start_time,
        "final_avg_reward":  float(avg_reward),
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
        'algorithm':         'MAPPO-Ambulance',
        'algorithm_name_env': 'project1_std_dqn',
        'obs_dim':           obs_dim,
        'pretrained_model':  args.load_model if args.load_model else None,
        'start_episode':     start_episode,
        'max_episodes':      max_episodes,
    }
    with open(os.path.join(exp_dir, 'exp_config.json'), 'w') as f:
        json.dump(exp_config, f, indent=2)

    elapsed_h = (time.time() - training_start_time) / 3600
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Experiment dir : {exp_dir}")
    print(f"   |-- models/agent_final.pt       (final model)")
    print(f"   |-- logs/training.json          (per-episode metrics)")
    print(f"   |-- logs/final_results.json     (final summary)")
    print(f"   |-- logs/console_output.log     (full console output)")
    print(f"   `-- exp_config.json             (experiment config)")
    print(f"\nPerformance summary:")
    if args.load_model:
        print(f"   Pretrained model  : {args.load_model}")
        print(f"   Episodes trained  : {start_episode} - {max_episodes} "
              f"({max_episodes - start_episode + 1} episodes)")
    else:
        print(f"   Episodes trained  : {max_episodes}")
    print(f"   K / Z             : {K} / {Z}")
    print(f"   Final avg reward  : {avg_reward:.2f}")
    print(f"   Total time        : {elapsed_h:.2f} hours")
    print(f"{'='*70}\n")

    sys.stdout.flush()
    sys.stderr.flush()
    log_file_handle.close()

    env.close()


if __name__ == '__main__':
    main()
