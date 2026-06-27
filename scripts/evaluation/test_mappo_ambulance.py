"""
Evaluation script for MAPPO-Ambulance models trained with
train_parl_mappo_ambulance.py. - NOT UPDATED FOR FYP

Key differences vs test_parl_mappo.py:
  - Forces algorithm_name = "project1_std_dqn" so the environment uses the
    68-dim project1-std observation and the std-aware EMV reward.
  - Injects Z into GetRewards.REWARD_CONFIGS at startup so the reward
    values printed match the training objective.
  - Reports per-episode reg/emg waiting-time statistics (mean + std).
  - Reads Z from the saved exp_config.json (when available) so you don't
    have to remember the exact values used during training.
"""

import os
import sys
import json
import numpy as np
import argparse
import yaml
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
# Single episode runner
# ============================================================================

def run_test_episode(env, agent, deterministic=True, verbose=False):
    """Run one test episode; return a dict of metrics."""
    obs = env.reset()
    agent.reset()

    agent_ids  = env.get_agent_ids()
    total_reward = {aid: 0 for aid in agent_ids}
    steps = 0

    # Accumulators for reward-component stats
    #ep_reg_mean_sum = 0.0 old historic code
    #ep_reg_std_sum  = 0.0 old historic code
    ep_reg_group1_mean_sum = 0.0
    ep_reg_group1_std_sum = 0.0
    ep_reg_group2_mean_sum = 0.0
    ep_reg_group2_std_sum = 0.0
    ep_emg_mean_sum = 0.0
    ep_emg_std_sum  = 0.0
    stat_steps = 0

    while True:
        steps += 1

        obs_list    = [obs[aid] for aid in agent_ids]
        actions     = agent.select_action(obs_list, deterministic=deterministic)
        action_dict = {aid: actions[i] for i, aid in enumerate(agent_ids)}

        next_obs, reward_dict, done, info = env.step(action_dict)

        for aid, r in reward_dict.items():
            total_reward[aid] += r

        # Collect waiting-time statistics from the first intersection
        try:
            world    = env.env
            first_ts = world.id2intersection[agent_ids[0]]
            stats    = first_ts.Rewards.get_reward_statistics()
            
            # Kept historic code
            #ep_reg_mean_sum += stats['regular_vehicles']['mean_waiting']
            #ep_reg_std_sum  += stats['regular_vehicles']['std_waiting']
            
            ep_reg_group1_mean_sum += stats['regular_group1_vehicles']['mean_waiting']
            ep_reg_group1_std_sum += stats['regular_group1_vehicles']['std_waiting']
            ep_reg_group2_mean_sum += stats['regular_group2_vehicles']['mean_waiting']
            ep_reg_group2_std_sum += stats['regular_group2_vehicles']['std_waiting']
            ep_emg_mean_sum += stats['emergency_vehicles']['mean_waiting']
            ep_emg_std_sum  += stats['emergency_vehicles']['std_waiting']
            stat_steps += 1
        except Exception:
            pass

        if verbose and steps % 50 == 0:
            avg_r = np.mean([total_reward[aid] for aid in agent_ids])
            print(f"    Step {steps:4d}: avg_reward={avg_r:.2f}")

        obs = next_obs
        if done:
            break

    # Episode-level averages
    if stat_steps > 0:
        reward_stats = {
            'reg_group1_waiting_mean': ep_reg_group1_mean_sum / stat_steps,
            'reg_group1_waiting_std':  ep_reg_group1_std_sum  / stat_steps,
            'reg_group2_waiting_mean': ep_reg_group2_mean_sum / stat_steps,
            'reg_group2_waiting_std':  ep_reg_group2_std_sum  / stat_steps,
            'emg_waiting_mean': ep_emg_mean_sum / stat_steps,
            'emg_waiting_std':  ep_emg_std_sum  / stat_steps,
        }
    else:
        reward_stats = {
            'reg_group1_waiting_mean': 0.0, 'reg_group1_waiting_std': 0.0,
            'reg_group2_waiting_mean': 0.0, 'reg_group2_waiting_std': 0.0,
            'emg_waiting_mean': 0.0, 'emg_waiting_std': 0.0,
        }

    world = env.env
    ambulance_trip_times   = [t for vid, t in world.vehicles_trip_time.items()
                               if vid.startswith("ambulance_")]
    ambulance_duration     = float(np.mean(ambulance_trip_times)) if ambulance_trip_times else 0.0
    civilian_trip_times    = [t for vid, t in world.vehicles_trip_time.items()
                               if not vid.startswith("ambulance_")]
    civilian_avg_trip_time = np.mean(civilian_trip_times) if civilian_trip_times else 0.0
    avg_reward             = np.mean([total_reward[aid] for aid in agent_ids])

    return {
        'total_reward':          total_reward,
        'avg_reward':            float(avg_reward),
        'steps':                 steps,
        'ambulance_duration':    float(ambulance_duration),
        'civilian_avg_trip_time': float(civilian_avg_trip_time),
        **{k: float(v) for k, v in reward_stats.items()},
    }


# ============================================================================
# Core test function
# ============================================================================

def test_model(
    model_path,
    config_path,
    scenario_dir,
    Z=None,
    num_episodes=10,
    deterministic=True,
    seed=42,
    gui=False,
    save_results=None,
    verbose=False,
):
    """
    Evaluate a trained MAPPO-Ambulance model.

    Z,  resolution order:
        1. CLI arguments (--Z)
        2. exp_config.json next to the model file
        3. mappo_ambulance.yaml  ambulance.Z
        4. Hard-coded defaults (1.0)
    """
    print("=" * 80)
    print("MAPPO-Ambulance Model Evaluation")
    print("=" * 80)
    print(f"Model path     : {model_path}")
    print(f"Config file    : {config_path}")
    print(f"Scenario dir   : {scenario_dir}")
    print(f"Episodes       : {num_episodes}")
    print(f"Policy         : {'deterministic' if deterministic else 'stochastic'}")
    print(f"GUI            : {'on' if gui else 'off'}")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Load YAML config
    # ------------------------------------------------------------------
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Resolve Z
    # Z priority: CLI > exp_config.json > YAML > hardcoded default
    # ------------------------------------------------------------------
    # Try reading from exp_config.json saved next to / above the model
    exp_Z = None
    for candidate in [
        os.path.join(os.path.dirname(model_path), '..', 'exp_config.json'),
        os.path.join(os.path.dirname(model_path), 'exp_config.json'),
    ]:
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            try:
                with open(candidate) as f:
                    exp_cfg = json.load(f)
                exp_Z = exp_cfg.get('Z')
                print(f"Loaded Z={exp_Z} from {candidate}")
            except Exception:
                pass
            break

    Z = (Z if Z is not None
         else exp_Z if exp_Z is not None
         else config.get('algorithm', {}).get('ambulance', {}).get('Z', 3.0))

    print(f"Using Z={Z}")
    print(f"Reward formula : -sum(lane_weights * lane_queues) # new reward function, where lane_weights = (1 + Z/EV_norm_tta)\n")

    # Inject Z so the reward function uses the correct values
    GetRewards.REWARD_CONFIGS['final_year_project_reward']['Z'] = Z 

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Build SUMO config
    # ------------------------------------------------------------------
    configs_dir = "tmp/test_mappo_ambulance_configs"
    os.makedirs(configs_dir, exist_ok=True)

    sumo_cfg = {
        "name":             "test_mappo_emergency_ambulance", # was 'test_emergency_ambulance' historically
        "dir":              scenario_dir,
        "roadnetFile":      "2_intersection_corridor.net.xml",
        "flowFile":         "vtypes.rou.xml,2_intersection_corridor.rou.xml,ambulance.rou.xml",
        "combined_file":    "2_intersection_corridor.sumocfg",
        "gui":              True, # Forces GUI on if set to 'True' otherwise set to 'gui' variable
        "no_warning":       True,
        "decision_interval": 5,
        "min_green":         5,
        "yellow_length":     4,
    }
    sumo_config_path = os.path.join(configs_dir, 'test_sumo_config.json')
    with open(sumo_config_path, 'w') as f:
        json.dump(sumo_cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Environment config
    # algorithm_name MUST be "final_year_project_dqn" to match training
    # ------------------------------------------------------------------
    env_config = {
        "sumo_config":           os.path.abspath(sumo_config_path),
        "interface":             "traci",
        "seed":                  seed,
        "sync_mode":             True,
        "obs_to_subscribe":      config['algorithm']['observation']['obs_to_subscribe'],
        "reward_to_subscribe":   config['algorithm']['reward']['reward_to_subscribe'],
        "algorithm_name":        "final_year_project_dqn",   # TODO: change this to be fyp or final_year_project for FYP model OR project1_std_dqn for baseline model
        "normalize_observation": config['algorithm']['observation'].get('normalize', False),
        "norm_params":           config['algorithm']['observation'].get('norm_params', {}),
        "reward_weights":        config['algorithm']['reward'].get('reward_weights', [1.0]),
        "reward_scale":          config['algorithm']['reward'].get('scale', 1.0),
        "reward_clip_range":     config['algorithm']['reward'].get('clip_range', None),
    }

    # ------------------------------------------------------------------
    # Probe env to get dimensions (then close)
    # ------------------------------------------------------------------
    print("Initialising environment to detect dimensions...")
    probe_env  = PARLSumoEnv(env_config)
    agent_ids  = probe_env.get_agent_ids()
    n_agents   = len(agent_ids)
    obs_dim    = probe_env.observation_space(agent_ids[0]).shape[0]
    act_dim    = probe_env.action_space(agent_ids[0]).n
    probe_env.close()

    print(f"Agents: {n_agents}  |  obs_dim: {obs_dim}  |  act_dim: {act_dim}\n")

    # ------------------------------------------------------------------
    # Build and load agent
    # ------------------------------------------------------------------
    print("Loading model...")
    device = 'cuda' if config.get('use_cuda', False) and torch.cuda.is_available() else 'cpu'

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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    agent.load(model_path)
    agent.mac.agent.eval()
    print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Test loop
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Running evaluation episodes...")
    print("=" * 80 + "\n")

    all_results = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env_config["seed"] = ep_seed
        env = PARLSumoEnv(env_config)

        print(f"Episode {ep+1:3d}/{num_episodes} (seed={ep_seed}) ... ", end='', flush=True)
        result = run_test_episode(env, agent, deterministic=deterministic, verbose=verbose)
        all_results.append(result)

        print(f"reward={result['avg_reward']:7.2f} | "
              f"EMV={result['ambulance_duration']:.1f}s | "
              f"civilian={result['civilian_avg_trip_time']:.1f}s | "
              f"reg_group1_wait={result['reg_group1_waiting_mean']:.1f}s | " 
              f"reg_group1_std={result['reg_group1_waiting_std']:.1f}s | " 
              f"reg_group2_wait={result['reg_group2_waiting_mean']:.1f}s | "
              f"reg_group2_std={result['reg_group2_waiting_std']:.1f}s | "
              f"emg_wait={result['emg_waiting_mean']:.1f}s | "
              f"steps={result['steps']}")

        env.close()

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    def _stats(key):
        vals = [r[key] for r in all_results]
        return float(np.mean(vals)), float(np.std(vals))

    avg_rew_mean,  avg_rew_std                = _stats('avg_reward')
    amb_mean,      amb_std                    = _stats('ambulance_duration')
    civ_mean,      civ_std                    = _stats('civilian_avg_trip_time')
    reg_group1_mean_mean, reg_group1_mean_std = _stats('reg_group1_waiting_mean')
    reg_group1_std_mean, reg_group1_std_std   = _stats('reg_group1_waiting_std')
    reg_group2_mean_mean, reg_group2_mean_std = _stats('reg_group2_waiting_mean')
    reg_group2_std_mean, reg_group2_std_std   = _stats('reg_group2_waiting_std')
    emg_mean_mean, emg_mean_std               = _stats('emg_waiting_mean')
    emg_std_mean,  emg_std_std                = _stats('emg_waiting_std')
    steps_mean,    steps_std                  = _stats('steps')

    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"  Avg reward          : {avg_rew_mean:8.2f} +/- {avg_rew_std:.2f}")
    print(f"  EMV trip time       : {amb_mean:8.2f}s +/- {amb_std:.2f}s")
    print(f"  Civilian trip time  : {civ_mean:8.2f}s +/- {civ_std:.2f}s")
    print(f"  Regular Group 1 wait mean   : {reg_group1_mean_mean:8.2f}s +/- {reg_group1_mean_std:.2f}s")
    print(f"  Regular Group 1 wait std    : {reg_group1_std_mean:8.2f}s +/- {reg_group1_std_std:.2f}s")
    print(f"  Regular Group 2 wait mean   : {reg_group2_mean_mean:8.2f}s +/- {reg_group2_mean_std:.2f}s")
    print(f"  Regular Group 2 wait std    : {reg_group2_std_mean:8.2f}s +/- {reg_group2_std_std:.2f}s")
    print(f"  EMG wait mean       : {emg_mean_mean:8.2f}s +/- {emg_mean_std:.2f}s")
    print(f"  Avg steps/episode   : {steps_mean:8.1f} +/- {steps_std:.1f}")
    print(f"{'='*80}\n")

    summary = {
        'model_path':            model_path,
        'config_path':           config_path,
        'algorithm_name_env':    'final_year_project_dqn', # TODO: change this to be fyp or final_year_project for FYP model OR project1_std_dqn for baseline model
        'Z':                     Z,
        'num_episodes':          num_episodes,
        'deterministic':         deterministic,
        'seed':                  seed,
        # aggregated metrics
        'avg_reward_mean':       avg_rew_mean,
        'avg_reward_std':        avg_rew_std,
        'ambulance_time_mean':   amb_mean,
        'ambulance_time_std':    amb_std,
        'civilian_time_mean':    civ_mean,
        'civilian_time_std':     civ_std,
        'reg_group1_waiting_mean_mean': reg_group1_mean_mean,
        'reg_group1_waiting_mean_std':  reg_group1_mean_std,
        'reg_group1_waiting_std_mean':  reg_group1_std_mean,
        'reg_group1_waiting_std_std':   reg_group1_std_std,
        'reg_group2_waiting_mean_mean': reg_group2_mean_mean,
        'reg_group2_waiting_mean_std':  reg_group2_mean_std,
        'reg_group2_waiting_std_mean':  reg_group2_std_mean,
        'reg_group2_waiting_std_std':   reg_group2_std_std,
        'emg_waiting_mean_mean': emg_mean_mean,
        'emg_waiting_mean_std':  emg_mean_std,
        'steps_mean':            steps_mean,
        'steps_std':             steps_std,
        # per-episode detail
        'all_results':           all_results,
    }

    if save_results:
        save_dir = os.path.dirname(save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_results, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to: {save_results}\n")

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained MAPPO-Ambulance model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (10 episodes, no GUI)
  python scripts/evaluation/test_mappo_ambulance.py \\
      --model-path experiments/mappo_ambulance_K0.5_Z3.0_seed42_XXXX/models/agent_final.pt \\
      --config configs/tsc/mappo_ambulance.yaml

  # With GUI, custom Z, save results
  python scripts/evaluation/test_mappo_ambulance.py \\
      --model-path experiments/.../models/agent_final.pt \\
      --config configs/tsc/mappo_ambulance.yaml \\
      --Z 5.0 --gui --num-episodes 5 \\
      --save-results evaluations/my_test.json
""")
    
    # TODO: Change this top arguments default '' to change the trained model that is being used
    parser.add_argument('--model-path', type=str, #required=True, removed 'required' and added default to run in IDE instead
                        default='experiments/*insert_model_folder_name*/models/agent_final.pt',
                        help='Path to the .pt model checkpoint')
    
    parser.add_argument('--config', type=str,
                        default='configs/tsc/mappo_fyp_config.yaml',
                        help='YAML config used during training (default: configs/tsc/mappo_fyp_config.yaml)')
    parser.add_argument('--scenario-dir', type=str,
                        default='scenarios/2_intersection_corridor', # change to scenario to be tested on
                        help='SUMO scenario directory')

    # Z override
    parser.add_argument('--Z', type=float, default=None,
                        help='EMV penalty multiplier Z (auto-detected from exp_config.json if omitted)')

    parser.add_argument('--num-episodes', type=int, default=2,
                        help='Number of test episodes (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (each episode uses seed+i)')
    parser.add_argument('--gui', action='store_true',
                        help='Show SUMO GUI')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print step-level progress every 50 steps')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save JSON result file')

    args = parser.parse_args()

    test_model(
        model_path=args.model_path,
        config_path=args.config,
        scenario_dir=args.scenario_dir,
        Z=args.Z,
        num_episodes=args.num_episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
        gui=args.gui,
        save_results=args.save_results,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
