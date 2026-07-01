"""
No-RL Baseline Evaluation (correct RL-mirrored version)

This version mirrors test_dqn_ambulance.py exactly,
but replaces the neural policy with a deterministic rule.
"""

import os
import sys
import json
import numpy as np
import argparse
import yaml

# ---------------------------------------------------------
# Project root setup (same as RL script)
# ---------------------------------------------------------
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Working directory set to: {os.getcwd()}")

from src.core.parlenv import PARLSumoEnv


# =========================================================
# Deterministic policy (REPLACES DQN)
# =========================================================
class FixedTimePolicy:
    def __init__(self, switch_every=10):
        self.switch_every = switch_every
        self.counters = {}
        self.phases = [0, 1, 2, 3]  # ✅ ONLY valid phases

    def act(self, obs_dict):
        actions = {}

        for aid in obs_dict.keys():
            self.counters[aid] = self.counters.get(aid, 0) + 1

            idx = (self.counters[aid] // self.switch_every) % len(self.phases)
            actions[aid] = self.phases[idx]

        return actions


# =========================================================
# Episode runner (IDENTICAL structure to RL script)
# =========================================================
def run_episode(env, policy, verbose=False):

    obs = env.reset()
    agent_ids = env.get_agent_ids()

    total_reward = {aid: 0 for aid in agent_ids}
    steps = 0

    #ep_reg_mean_sum = 0.0 kept old historic code
    #ep_reg_std_sum = 0.0 kept old historic code
    ep_reg_group1_mean_sum = 0.0
    ep_reg_group1_std_sum = 0.0
    ep_reg_group2_mean_sum = 0.0
    ep_reg_group2_std_sum = 0.0
    ep_emg_mean_sum = 0.0
    ep_emg_std_sum = 0.0
    stat_steps = 0

    while True:
        steps += 1

        # -----------------------------
        # ACTION (ONLY DIFFERENCE FROM RL SCRIPT)
        # -----------------------------
        action_dict = policy.act(obs)

        # CRITICAL: single env.step ONLY
        next_obs, reward_dict, done, info = env.step(action_dict)

        for aid, r in reward_dict.items():
            total_reward[aid] += r

        # -----------------------------
        # stats (same as RL script)
        # -----------------------------
        try:
            world = env.env
            first = world.id2intersection[agent_ids[0]]
            stats = first.Rewards.get_reward_statistics()

            # kept historic code for context
            #ep_reg_mean_sum += stats['regular_vehicles']['mean_waiting']
            #ep_reg_std_sum += stats['regular_vehicles']['std_waiting'] 
            #ep_emg_mean_sum += stats['emergency_vehicles']['mean_waiting']
            #ep_emg_std_sum += stats['emergency_vehicles']['std_waiting']
            
            # New definition that uses group 1 and group 2
            ep_reg_group1_mean_sum += stats['regular_group1_vehicles']['mean_waiting']
            ep_reg_group1_std_sum += stats['regular_group1_vehicles']['std_waiting']
            ep_reg_group2_mean_sum += stats['regular_group2_vehicles']['mean_waiting']
            ep_reg_group2_std_sum += stats['regular_group2_vehicles']['std_waiting']
            ep_emg_mean_sum += stats['emergency_vehicles']['mean_waiting']
            ep_emg_std_sum += stats['emergency_vehicles']['std_waiting']
            stat_steps += 1

        except Exception:
            pass

        obs = next_obs

        if done:
            break

    # -----------------------------
    # episode summary
    # -----------------------------
    reward_stats = {
        'reg_group1_waiting_mean': ep_reg_group1_mean_sum / stat_steps if stat_steps else 0.0,
        'reg_group1_waiting_std': ep_reg_group1_std_sum / stat_steps if stat_steps else 0.0,
        'reg_group2_waiting_mean': ep_reg_group2_mean_sum / stat_steps if stat_steps else 0.0,
        'reg_group2_waiting_std': ep_reg_group2_std_sum / stat_steps if stat_steps else 0.0,
        'emg_waiting_mean': ep_emg_mean_sum / stat_steps if stat_steps else 0.0,
        'emg_waiting_std': ep_emg_std_sum / stat_steps if stat_steps else 0.0,
    }

    world = env.env

    ambulance_times = [
        t for vid, t in world.vehicles_trip_time.items()
        if vid.startswith("ambulance_")
    ]
    civilian_times = [
        t for vid, t in world.vehicles_trip_time.items()
        if not vid.startswith("ambulance_")
    ]

    return {
        "avg_reward": float(np.mean(list(total_reward.values()))),
        "steps": steps,
        "ambulance_duration": float(np.mean(ambulance_times)) if ambulance_times else 0.0,
        "civilian_avg_trip_time": float(np.mean(civilian_times)) if civilian_times else 0.0,
        **reward_stats
    }


# =========================================================
# Main evaluation loop
# =========================================================
def test_baseline(config_path, scenario_dir, num_episodes=5, seed=42, gui=False):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    all_results = []

    for ep in range(num_episodes):

        env_config = {
            "sumo_config": "scenarios/3_intersection_corridor/3_intersection_corridor.sumocfg",  # IMPORTANT: your PARL wrapper builds this internally
            "interface": "traci",
            "seed": seed + ep,
            "sync_mode": True,
            "obs_to_subscribe": config["algorithm"]["observation"]["obs_to_subscribe"],
            "reward_to_subscribe": config["algorithm"]["reward"]["reward_to_subscribe"],
            "algorithm_name": "project1_std_dqn",
            "normalize_observation": config["algorithm"]["observation"].get("normalize", False),
            "norm_params": config["algorithm"]["observation"].get("norm_params", {}),
        }

        sumo_cfg = {
            "name": "baseline",
            "dir": scenario_dir,
            "roadnetFile": "3_intersection_corridor.net.xml",
            "flowFile": "3_intersection_corridor_1350.rou.xml", # TODO: YOU ALSO NEED TO MANUALLY CHANGE THE .SUMOCFG
            "combined_file": "3_intersection_corridor.sumocfg",
            "gui": True, # TODO: HAVE TO CHANGE THIS VALUE TO CHANGE IF USE SUMO UI OR NOT, MAKE THIS WORK WITH A VAR.
            "no_warning": True,
            "decision_interval": 5,
            "min_green": 5,
            "yellow_length": 4,
        }
        
        sumo_config_path = "tmp/baseline_config.json"
        
        with open(sumo_config_path, "w") as f:
            json.dump(sumo_cfg, f)
        
        env_config["sumo_config"] = os.path.abspath(sumo_config_path)

        env = PARLSumoEnv(env_config)
        
        policy = FixedTimePolicy(switch_every=args.switch_every)
        
        print("\n================ BASELINE CONFIG ================")
        print(f"Policy type        : FixedTimePolicy (round-robin)")
        print(f"Switch interval    : {policy.switch_every} decision steps")
        print("=================================================\n")

        print(f"\nEpisode {ep+1}/{num_episodes}")

        result = run_episode(env, policy)
        all_results.append(result)

        print(
            f"reward={result['avg_reward']:.2f} | "
            f"EMV={result['ambulance_duration']:.1f}s | "
            f"civilian={result['civilian_avg_trip_time']:.1f}s | "
            f"reg_group1_wait={result['reg_group1_waiting_mean']:.1f}s | " 
            f"reg_group2_wait={result['reg_group2_waiting_mean']:.1f}s | "
            f"emg_wait={result['emg_waiting_mean']:.1f}s | "
            f"steps={result['steps']}"
        )

        env.close()

    # summary
    def mean(key):
        return np.mean([r[key] for r in all_results])

    print("\n===== SUMMARY =====")
    print("Avg reward:", mean("avg_reward"))
    print("Ambulance time:", mean("ambulance_duration"))
    print("Civilian time:", mean("civilian_avg_trip_time"))
    
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
    print("Baseline Evaluation Summary")
    print(f"  Baseline policy     : FixedTime (switch_every={args.switch_every})")
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


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tsc/dqn_fyp_config.yaml")
    parser.add_argument("--scenario-dir", default="scenarios/3_intersection_corridor")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    
    parser.add_argument( # Parser for the cycle switch timings of the no RL model version
    "--switch-every",
    type=int,
    default=10,
    help="Number of decision steps before switching traffic phase"
)

    args = parser.parse_args()

    test_baseline(
        args.config,
        args.scenario_dir,
        args.episodes,
        args.seed,
        args.gui
    )
