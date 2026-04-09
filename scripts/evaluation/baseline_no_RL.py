"""
Evaluation script for a no-RL baseline for Ambulance traffic control.

This mirrors the DQN/MAPPO evaluation scripts but uses a
deterministic baseline policy (e.g., fixed timing or round-robin)
instead of an RL agent.
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

from src.core.parlenv import PARLSumoEnv
from src.core.Rewards import GetRewards

# ---------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------

class NoRLBaselineAgent:
    """A simple baseline agent using fixed rules (no learning)."""
    def __init__(self, act_dim):
        self.act_dim = act_dim

    def select_action(self, obs_dict):
        # Example baseline: always choose action 0 for all agents
        return {aid: 0 for aid in obs_dict.keys()}


def run_test_episode(env, agent, verbose=False):
    """Run one test episode; return metrics."""
    obs = env.reset()
    total_reward = {aid: 0 for aid in env.get_agent_ids()}
    steps = 0

    # Accumulators for reward-component stats
    ep_reg_mean_sum = 0.0
    ep_reg_std_sum  = 0.0
    ep_emg_mean_sum = 0.0
    ep_emg_std_sum  = 0.0
    stat_steps = 0

    while True:
        steps += 1

        action_dict = agent.select_action(obs)
        next_obs, reward_dict, done, info = env.step(action_dict)

        for aid, r in reward_dict.items():
            total_reward[aid] += r

        # Collect waiting-time statistics from the first intersection
        try:
            world    = env.env
            first_ts = world.id2intersection[list(obs.keys())[0]]
            stats    = first_ts.Rewards.get_reward_statistics()
            ep_reg_mean_sum += stats['regular_vehicles']['mean_waiting']
            ep_reg_std_sum  += stats['regular_vehicles']['std_waiting']
            ep_emg_mean_sum += stats['emergency_vehicles']['mean_waiting']
            ep_emg_std_sum  += stats['emergency_vehicles']['std_waiting']
            stat_steps += 1
        except Exception:
            pass

        if verbose and steps % 50 == 0:
            avg_r = np.mean([total_reward[aid] for aid in total_reward])
            print(f"    Step {steps:4d}: avg_reward={avg_r:.2f}")

        obs = next_obs
        if done:
            break

    # Episode-level averages
    reward_stats = {
        'reg_waiting_mean': ep_reg_mean_sum / stat_steps if stat_steps else 0.0,
        'reg_waiting_std':  ep_reg_std_sum / stat_steps if stat_steps else 0.0,
        'emg_waiting_mean': ep_emg_mean_sum / stat_steps if stat_steps else 0.0,
        'emg_waiting_std':  ep_emg_std_sum / stat_steps if stat_steps else 0.0,
    }

    world = env.env
    ambulance_trip_times   = [t for vid, t in world.vehicles_trip_time.items()
                               if vid.startswith("ambulance_")]
    ambulance_duration     = float(np.mean(ambulance_trip_times)) if ambulance_trip_times else 0.0
    civilian_trip_times    = [t for vid, t in world.vehicles_trip_time.items()
                               if not vid.startswith("ambulance_")]
    civilian_avg_trip_time = float(np.mean(civilian_trip_times)) if civilian_trip_times else 0.0
    avg_reward             = float(np.mean([total_reward[aid] for aid in total_reward]))

    return {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'steps': steps,
        'ambulance_duration': ambulance_duration,
        'civilian_avg_trip_time': civilian_avg_trip_time,
        **reward_stats,
    }


# ---------------------------------------------------------
# Core test function
# ---------------------------------------------------------

def test_baseline(
    config_path,
    scenario_dir,
    num_episodes=10,
    seed=42,
    gui=True,
    save_results=None,
    verbose=False,
):
    """
    Evaluate a no-RL baseline agent.
    """
    print("="*80)
    print("No-RL Baseline Evaluation")
    print("="*80)
    print(f"Config file    : {config_path}")
    print(f"Scenario dir   : {scenario_dir}")
    print(f"Episodes       : {num_episodes}")
    print(f"GUI            : {'on' if gui else 'off'}")
    print("="*80 + "\n")

    # ------------------------------------------------------------------
    # Load YAML config
    # ------------------------------------------------------------------
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Build SUMO config
    # ------------------------------------------------------------------
    configs_dir = "tmp/test_no_rl_baseline_configs"
    os.makedirs(configs_dir, exist_ok=True)

    sumo_cfg = {
        "name":             "test_no_rl_baseline",
        "dir":              scenario_dir,
        "roadnetFile":      "draft02.net.xml",
        "flowFile":         "vtypes.rou.xml,draft02.rou.xml,ambulance.rou.xml",
        "combined_file":    "draft02.sumocfg",
        "gui":              True,
        "no_warning":       True,
        "decision_interval": 5,
        "min_green":        5,
        "yellow_length":    4,
    }
    sumo_config_path = os.path.join(configs_dir, 'test_sumo_config.json')
    with open(sumo_config_path, 'w') as f:
        json.dump(sumo_cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Environment config
    # ------------------------------------------------------------------
    env_config = {
        "sumo_config":           os.path.abspath(sumo_config_path),
        "interface":             "traci",
        "seed":                  seed,
        "sync_mode":             True,
        "obs_to_subscribe":      config['algorithm']['observation']['obs_to_subscribe'],
        "reward_to_subscribe":   config['algorithm']['reward']['reward_to_subscribe'],
        "algorithm_name":        "project1_std_dqn",
        "normalize_observation": config['algorithm']['observation'].get('normalize', False),
        "norm_params":           config['algorithm']['observation'].get('norm_params', {}),
    }

    # ------------------------------------------------------------------
    # Probe env to get dimensions
    # ------------------------------------------------------------------
    print("Initialising environment to detect dimensions...")
    probe_env  = PARLSumoEnv(env_config)
    agent_ids  = probe_env.get_agent_ids()
    act_dim    = probe_env.action_space(agent_ids[0]).n
    probe_env.close()

    # ------------------------------------------------------------------
    # Build no-RL agent
    # ------------------------------------------------------------------
    agent = NoRLBaselineAgent(act_dim=act_dim)

    # ------------------------------------------------------------------
    # Run episodes
    # ------------------------------------------------------------------
    all_results = []
    for ep in range(num_episodes):
        ep_seed = seed + ep
        env_config["seed"] = ep_seed
        env = PARLSumoEnv(env_config)

        print(f"Episode {ep+1}/{num_episodes} (seed={ep_seed}) ... ", end='', flush=True)
        result = run_test_episode(env, agent, verbose=verbose)
        all_results.append(result)

        print(f"reward={result['avg_reward']:7.2f} | "
              f"EMV={result['ambulance_duration']:.1f}s | "
              f"civilian={result['civilian_avg_trip_time']:.1f}s | "
              f"reg_wait={result['reg_waiting_mean']:.1f}s"
              f"(std={result['reg_waiting_std']:.1f}) | "
              f"emg_wait={result['emg_waiting_mean']:.1f}s | "
              f"steps={result['steps']}")

        env.close()

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    def _stats(key):
        vals = [r[key] for r in all_results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        'num_episodes': num_episodes,
        'avg_reward_mean': _stats('avg_reward')[0],
        'avg_reward_std':  _stats('avg_reward')[1],
        'ambulance_time_mean': _stats('ambulance_duration')[0],
        'ambulance_time_std':  _stats('ambulance_duration')[1],
        'civilian_time_mean': _stats('civilian_avg_trip_time')[0],
        'civilian_time_std':  _stats('civilian_avg_trip_time')[1],
        'reg_waiting_mean_mean': _stats('reg_waiting_mean')[0],
        'reg_waiting_mean_std':  _stats('reg_waiting_mean')[1],
        'reg_waiting_std_mean': _stats('reg_waiting_std')[0],
        'reg_waiting_std_std':  _stats('reg_waiting_std')[1],
        'emg_waiting_mean_mean': _stats('emg_waiting_mean')[0],
        'emg_waiting_mean_std':  _stats('emg_waiting_mean')[1],
        'emg_waiting_std_mean': _stats('emg_waiting_std')[0],
        'emg_waiting_std_std':  _stats('emg_waiting_std')[1],
        'all_results': all_results,
    }

    if save_results:
        save_dir = os.path.dirname(save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_results, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to: {save_results}\n")

    return summary


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate no-RL baseline agent')
    parser.add_argument('--config', type=str,
                        default='configs/tsc/dqn_ambulance.yaml',
                        help='YAML config file used for environment')
    parser.add_argument('--scenario-dir', type=str,
                        default='scenarios/emergency_vehicle',
                        help='SUMO scenario directory')
    parser.add_argument('--num-episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-results', type=str, default=None)

    args = parser.parse_args()

    test_baseline(
        config_path=args.config,
        scenario_dir=args.scenario_dir,
        num_episodes=args.num_episodes,
        seed=args.seed,
        gui=args.gui,
        save_results=args.save_results,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()