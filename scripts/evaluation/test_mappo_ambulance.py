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

def run_test_episode(env, agent, deterministic=True, verbose=False, focus_agent_idx=0):
    """Run one test episode; return a dict of metrics.

    focus_agent_idx: index into agent_ids identifying which single
    intersection should be reported individually (in addition to the
    global/all-intersection metrics that are always computed).
    """
    obs = env.reset()
    agent.reset()

    agent_ids  = env.get_agent_ids()
    total_reward = {aid: 0 for aid in agent_ids}
    steps = 0

    # Per-intersection accumulators for reward-component stats (one set per agent)
    per_agent_sums = {
        aid: {
            'reg_group1_mean': 0.0, 'reg_group1_std': 0.0,
            'reg_group2_mean': 0.0, 'reg_group2_std': 0.0,
            'reg_all_mean': 0.0, 'reg_all_std': 0.0,
            'emg_mean': 0.0, 'emg_std': 0.0,
        } for aid in agent_ids
    }
    # Global (all-intersection, count-weighted) accumulators
    global_sums = {
        'reg_group1_mean': 0.0, 'reg_group1_std': 0.0,
        'reg_group2_mean': 0.0, 'reg_group2_std': 0.0,
        'reg_all_mean': 0.0, 'reg_all_std': 0.0,
        'emg_mean': 0.0, 'emg_std': 0.0,
    }
    stat_steps = 0

    while True:
        steps += 1

        obs_list    = [obs[aid] for aid in agent_ids]
        actions     = agent.select_action(obs_list, deterministic=deterministic)
        action_dict = {aid: actions[i] for i, aid in enumerate(agent_ids)}

        next_obs, reward_dict, done, info = env.step(action_dict)

        for aid, r in reward_dict.items():
            total_reward[aid] += r

        # Collect waiting-time statistics from EVERY intersection this step,
        # both per-agent and pooled into a single count-weighted "global" stat.
        try:
            world = env.env

            # per-group lists gathered across all agents THIS step, used to
            # build the pooled/global stat for this step
            step_counts  = {'reg_group1': [], 'reg_group2': [], 'reg_all': [], 'emg': []}
            step_means   = {'reg_group1': [], 'reg_group2': [], 'reg_all': [], 'emg': []}
            step_stds    = {'reg_group1': [], 'reg_group2': [], 'reg_all': [], 'emg': []}

            for aid in agent_ids:
                ts    = world.id2intersection[aid]
                stats = ts.Rewards.get_reward_statistics()

                per_agent_sums[aid]['reg_group1_mean'] += stats['regular_group1_vehicles']['mean_waiting']
                per_agent_sums[aid]['reg_group1_std']  += stats['regular_group1_vehicles']['std_waiting']
                per_agent_sums[aid]['reg_group2_mean'] += stats['regular_group2_vehicles']['mean_waiting']
                per_agent_sums[aid]['reg_group2_std']  += stats['regular_group2_vehicles']['std_waiting']
                per_agent_sums[aid]['reg_all_mean']     += stats['regular_vehicles']['mean_waiting']
                per_agent_sums[aid]['reg_all_std']      += stats['regular_vehicles']['std_waiting']
                per_agent_sums[aid]['emg_mean']         += stats['emergency_vehicles']['mean_waiting']
                per_agent_sums[aid]['emg_std']          += stats['emergency_vehicles']['std_waiting']

                for key, group in (('reg_group1', 'regular_group1_vehicles'),
                                    ('reg_group2', 'regular_group2_vehicles'),
                                    ('reg_all', 'regular_vehicles'),
                                    ('emg', 'emergency_vehicles')):
                    step_counts[key].append(stats[group]['count'])
                    step_means[key].append(stats[group]['mean_waiting'])
                    step_stds[key].append(stats[group]['std_waiting'])

            # Combine each group's per-agent (count, mean, std) into one
            # count-weighted pooled (mean, std) for this step. This is the
            # correct way to merge sub-population statistics (equivalent to
            # concatenating every vehicle's waiting time across intersections
            # and computing mean/std over the pooled set), and it's O(n_agents)
            # cheap arithmetic - no extra traci calls beyond the per-agent ones above.
            for key in ('reg_group1', 'reg_group2', 'reg_all', 'emg'):
                counts = step_counts[key]
                total_n = sum(counts)
                if total_n > 0:
                    g_mean = sum(c * m for c, m in zip(counts, step_means[key])) / total_n
                    g_var  = sum(c * (s ** 2 + (m - g_mean) ** 2)
                                 for c, m, s in zip(counts, step_means[key], step_stds[key])) / total_n
                    g_std  = g_var ** 0.5
                else:
                    g_mean, g_std = 0.0, 0.0
                global_sums[f'{key}_mean'] += g_mean
                global_sums[f'{key}_std']  += g_std

            stat_steps += 1
        except Exception:
            pass

        if verbose and steps % 50 == 0:
            avg_r = np.mean([total_reward[aid] for aid in agent_ids])
            print(f"    Step {steps:4d}: avg_reward={avg_r:.2f}")

        obs = next_obs
        if done:
            break

    # Episode-level averages: per-intersection, global (pooled), and the
    # single "focus" intersection (kept as top-level keys for backward compat)
    focus_aid = agent_ids[focus_agent_idx]
    per_intersection_stats = {}
    if stat_steps > 0:
        for aid in agent_ids:
            s = per_agent_sums[aid]
            per_intersection_stats[aid] = {
                'reg_group1_waiting_mean': s['reg_group1_mean'] / stat_steps,
                'reg_group1_waiting_std':  s['reg_group1_std']  / stat_steps,
                'reg_group2_waiting_mean': s['reg_group2_mean'] / stat_steps,
                'reg_group2_waiting_std':  s['reg_group2_std']  / stat_steps,
                'reg_all_waiting_mean': s['reg_all_mean'] / stat_steps,
                'reg_all_waiting_std':  s['reg_all_std']  / stat_steps,
                'emg_waiting_mean': s['emg_mean'] / stat_steps,
                'emg_waiting_std':  s['emg_std']  / stat_steps,
            }
        global_stats = {
            'reg_group1_waiting_mean': global_sums['reg_group1_mean'] / stat_steps,
            'reg_group1_waiting_std':  global_sums['reg_group1_std']  / stat_steps,
            'reg_group2_waiting_mean': global_sums['reg_group2_mean'] / stat_steps,
            'reg_group2_waiting_std':  global_sums['reg_group2_std']  / stat_steps,
            'reg_all_waiting_mean': global_sums['reg_all_mean'] / stat_steps,
            'reg_all_waiting_std':  global_sums['reg_all_std']  / stat_steps,
            'emg_waiting_mean': global_sums['emg_mean'] / stat_steps,
            'emg_waiting_std':  global_sums['emg_std']  / stat_steps,
        }
    else:
        empty = {
            'reg_group1_waiting_mean': 0.0, 'reg_group1_waiting_std': 0.0,
            'reg_group2_waiting_mean': 0.0, 'reg_group2_waiting_std': 0.0,
            'reg_all_waiting_mean': 0.0, 'reg_all_waiting_std': 0.0,
            'emg_waiting_mean': 0.0, 'emg_waiting_std': 0.0,
        }
        per_intersection_stats = {aid: dict(empty) for aid in agent_ids}
        global_stats = dict(empty)

    # Top-level keys stay as the focus intersection's stats (backward compatible)
    reward_stats = per_intersection_stats[focus_aid]

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
        # focus intersection (backward-compatible top-level keys)
        'focus_agent_id':        focus_aid,
        **{k: float(v) for k, v in reward_stats.items()},
        # NEW: global (all-intersection, count-weighted) stats
        **{f'global_{k}': float(v) for k, v in global_stats.items()},
        # NEW: full per-intersection breakdown, keyed by agent_id
        'per_intersection': {
            aid: {k: float(v) for k, v in d.items()}
            for aid, d in per_intersection_stats.items()
        },
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
    focus_agent_idx=0,
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
        "roadnetFile":      "3_intersection_corridor_250long.net.xml",
        "flowFile":         "vtypes.rou.xml,3_intersection_corridor_250long_1800.rou.xml,ambulance_var1.rou.xml", # TODO: Scenarios: Check right network and demand files are used (also check right scenario folder is used in other places)
        "combined_file":    "3_intersection_corridor_250long.sumocfg",
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
    # algorithm_name MUST be "final_year_project_dqn" to match training or "project1_std_dqn" for baseline
    # ------------------------------------------------------------------
    env_config = {
        "sumo_config":           os.path.abspath(sumo_config_path),
        "interface":             "traci",
        "seed":                  seed,
        "sync_mode":             True,
        "obs_to_subscribe":      config['algorithm']['observation']['obs_to_subscribe'],
        "reward_to_subscribe":   config['algorithm']['reward']['reward_to_subscribe'],
        "algorithm_name":        "final_year_project",   # TODO: IMPORTANT: Sets what Observation state space is being used - change this to be fyp or final_year_project for FYP model (final_year_project_lane_mode for LANE FEATURES VERSION) OR project1_std_dqn for baseline model
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
    
    algorithm_name = env_config["algorithm_name"]
    print("=" * 70)
    if algorithm_name == "project1_std_dqn":
        print("\nUsing BASELINE RL (Kodogoda-style) observation/state space\n")
    elif algorithm_name == "final_year_project":
        print("\nUsing FINAL YEAR PROJECT observation/state space\n")
    elif algorithm_name == "final_year_project_lane_mode":
         print("\nUsing FINAL YEAR PROJECT !LANE! VERSION observation/state space\n")
    else:
        print(f"\nPotential algorithm name mismatch: {algorithm_name}\n")
    print("=" * 70)

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
        result = run_test_episode(env, agent, deterministic=deterministic, verbose=verbose,
                                   focus_agent_idx=focus_agent_idx)
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
    steps_mean,    steps_std                  = _stats('steps')

    # focus intersection (backward-compatible keys) and global (pooled) metrics
    # use the same _stats() helper, just pointed at the 'global_*' keys for the latter
    def _print_table(title, prefix):
        rg1m_m, rg1m_s = _stats(f'{prefix}reg_group1_waiting_mean')
        rg1s_m, rg1s_s = _stats(f'{prefix}reg_group1_waiting_std')
        rg2m_m, rg2m_s = _stats(f'{prefix}reg_group2_waiting_mean')
        rg2s_m, rg2s_s = _stats(f'{prefix}reg_group2_waiting_std')
        ram_m,  ram_s  = _stats(f'{prefix}reg_all_waiting_mean')
        ras_m,  ras_s  = _stats(f'{prefix}reg_all_waiting_std')
        egm_m,  egm_s  = _stats(f'{prefix}emg_waiting_mean')
        egs_m,  egs_s  = _stats(f'{prefix}emg_waiting_std')
        print(f"  {title}")
        print(f"    EMG wait mean             : {egm_m:8.2f}s +/- {egm_s:.2f}s") # these metrics are all matching what the title says - focus intersection is just one isolated intersection (for ID in brackets), while global is all intersections - avg.
        print(f"    EMG wait std dev          : {egs_m:8.2f}s +/- {egs_s:.2f}s")
        print(f"    Regular Group 1 wait mean : {rg1m_m:8.2f}s +/- {rg1m_s:.2f}s") 
        print(f"    Regular Group 1 wait std  : {rg1s_m:8.2f}s +/- {rg1s_s:.2f}s")
        print(f"    Regular Group 2 wait mean : {rg2m_m:8.2f}s +/- {rg2m_s:.2f}s")
        print(f"    Regular Group 2 wait std  : {rg2s_m:8.2f}s +/- {rg2s_s:.2f}s")
        print(f"    Regular (all) wait mean   : {ram_m:8.2f}s +/- {ram_s:.2f}s")
        print(f"    Regular (all) wait std    : {ras_m:8.2f}s +/- {ras_s:.2f}s")
        return {
            'reg_group1_waiting_mean_mean': rg1m_m, 'reg_group1_waiting_mean_std': rg1m_s,
            'reg_group1_waiting_std_mean':  rg1s_m, 'reg_group1_waiting_std_std':  rg1s_s,
            'reg_group2_waiting_mean_mean': rg2m_m, 'reg_group2_waiting_mean_std': rg2m_s,
            'reg_group2_waiting_std_mean':  rg2s_m, 'reg_group2_waiting_std_std':  rg2s_s,
            'reg_all_waiting_mean_mean': ram_m, 'reg_all_waiting_mean_std': ram_s,
            'reg_all_waiting_std_mean':  ras_m, 'reg_all_waiting_std_std':  ras_s,
            'emg_waiting_mean_mean': egm_m, 'emg_waiting_mean_std': egm_s,
        }

    focus_aid = all_results[0]['focus_agent_id'] if all_results else None

    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"  Avg reward          : {avg_rew_mean:8.2f} +/- {avg_rew_std:.2f}") # averaged across all agents
    print(f"  EMV trip time       : {amb_mean:8.2f}s +/- {amb_std:.2f}s") # is for whole network
    print(f"  Civilian trip time  : {civ_mean:8.2f}s +/- {civ_std:.2f}s") # is for whole network
    print(f"  Avg steps/episode   : {steps_mean:8.1f} +/- {steps_std:.1f}") # is episode related
    print()
    focus_summary  = _print_table(f"Focus intersection ({focus_aid})", prefix='')
    print()
    global_summary = _print_table("Global (all intersections, averaged)", prefix='global_')
    print(f"{'='*80}\n")

    # per-intersection breakdown (mean over episodes for every intersection)
    per_intersection_summary = {}
    if all_results:
        for aid in all_results[0]['per_intersection'].keys():
            per_intersection_summary[aid] = {
                metric: float(np.mean([r['per_intersection'][aid][metric] for r in all_results]))
                for metric in all_results[0]['per_intersection'][aid].keys()
            }

    summary = {
        'model_path':            model_path,
        'config_path':           config_path,
        'algorithm_name_env':    'final_year_project', # TODO: MAY? affect what Observation state space is being used (not 100% sure) - change this to be fyp or final_year_project for FYP model (final_year_project_lane_mode for LANE FEATURES VERSION) OR project1_std_dqn for baseline model
        'Z':                     Z,
        'num_episodes':          num_episodes,
        'deterministic':         deterministic,
        'seed':                  seed,
        'focus_agent_id':        focus_aid,
        # aggregated metrics
        'avg_reward_mean':       avg_rew_mean,
        'avg_reward_std':        avg_rew_std,
        'ambulance_time_mean':   amb_mean,
        'ambulance_time_std':    amb_std,
        'civilian_time_mean':    civ_mean,
        'civilian_time_std':     civ_std,
        'steps_mean':            steps_mean,
        'steps_std':             steps_std,
        # focus-intersection waiting-time metrics (backward compatible names)
        **focus_summary,
        # NEW: global (all-intersection, count-weighted) waiting-time metrics
        **{f'global_{k}': v for k, v in global_summary.items()},
        # NEW: per-intersection waiting-time metrics, keyed by agent_id
        'per_intersection_summary': per_intersection_summary,
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
    
    parser.add_argument('--model-path', type=str, #required=True, removed 'required' and added default to run in IDE instead
                        default='experiments/7.FYPInt_EVSpeed_EVDist_3Inter250_1800_mappo_ambulance_K0.5_Z3.0_seed42_20260703_140250/models/agent_final.pt', # TODO: TRAINED AGENT: switch this to be path to the trained agent you wish to use (from root project folder)
                        help='Path to the .pt model checkpoint')
    
    parser.add_argument('--config', type=str,
                        default='configs/tsc/mappo_ambulance.yaml', # TODO: Rewards: Factors (e.g. Z and K) for the reward are taken from this - should align with reward being used - from configs/tsc folder - mappo_ambulance for BASELINE, mappo_fyp_config for FYP
                        help='YAML config used during training (default: configs/tsc/mappo_fyp_config.yaml)')
    parser.add_argument('--scenario-dir', type=str,
                        default='scenarios/3_intersection_corridor_250long', # change to scenario to be tested on
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
    parser.add_argument('--focus-agent-idx', type=int, default=0, # TODO: can change this default='x' x value to get a different intersection of interest in evaluation metric summary
                        help='Index (into agent_ids) of the single intersection to '
                             'report individually, in addition to the global metrics (default: 0)')

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
        focus_agent_idx=args.focus_agent_idx,
    )


if __name__ == '__main__':
    main()