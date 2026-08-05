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
import csv
import random
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
from src.core.Observations import FYP_OBS_CONFIG, get_fyp_observation_dims, print_fyp_obs_config


# ============================================================================
# NEW: Evaluation demand pools
# ============================================================================
# Default pools for the "realistic" 1800 regular / 100s EV combination (this
# matches CURRICULUM stage 3's reg_1800 / ev_100s files in
# CL_train_parl_mappo_ambulance.py). One (regular_file, ev_file) pair is
# sampled per test episode - rng below is reproducible per episode (seeded
# by seed+ep), same convention as sample_episode_demand() in the training
# script, so re-running with the same --seed reproduces the same sequence
# of demand-file combinations.
DEFAULT_REGULAR_POOL = [
    "demand_regular/reg_1800_v1.rou.xml",
    "demand_regular/reg_1800_v2.rou.xml",
    "demand_regular/reg_1800_v3.rou.xml",
]
DEFAULT_EV_POOL = [
    "demand_ev/ev_100s_v1.rou.xml",
    "demand_ev/ev_100s_v2.rou.xml",
    "demand_ev/ev_100s_v3.rou.xml",
]


def sample_episode_demand(episode: int, regular_pool, ev_pool, rng=None):
    """
    Pick (regular_file, ev_file) for one test episode, sampled uniformly and
    independently from the given pools. Mirrors sample_episode_demand() in
    CL_train_parl_mappo_ambulance.py, minus the curriculum-stage logic (test
    time always draws from one fixed pool - no ramping).
    """
    rng = rng or random
    reg_file = rng.choice(regular_pool)
    ev_file = rng.choice(ev_pool)
    return reg_file, ev_file


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
    total_reward = {aid: np.zeros(agent.n_objectives) for aid in agent_ids}  # NEW FOR LEXICOGRAPHIC MORL
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
            avg_r = np.mean([total_reward[aid] for aid in agent_ids], axis=0)[0]
            print(f"    Step {steps:4d}: avg_reward={avg_r:.2f}")

        obs = next_obs
        if done:
            break

    # Finalize any vehicles still in the sim at episode cutoff
    world = env.env
    world.finalize_stranded_vehicles()

    def _vehicle_stats(lst):
        if len(lst) == 0:
            return 0.0, 0.0, 0
        return float(np.mean(lst)), float(np.std(lst)), len(lst)

    g1_mean, g1_std, g1_n   = _vehicle_stats(world.finished_reg_group1_delays)
    g2_mean, g2_std, g2_n   = _vehicle_stats(world.finished_reg_group2_delays)
    all_mean, all_std, all_n = _vehicle_stats(world.finished_reg_all_delays)
    ev_mean, ev_std, ev_n   = _vehicle_stats(world.finished_ev_delays)

    # NEW: raw per-vehicle delay values for THIS episode, copied out before
    # the next env.reset() clears world.finished_*_delays. This is the
    # underlying data behind the mean/std above - kept so test_model() can
    # pool it across episodes for histogram/fitted-curve plots instead of
    # only ever seeing the reduced (mean, std, n) summary.
    ep_raw_delays = {
        'group1':  [float(x) for x in world.finished_reg_group1_delays],
        'group2':  [float(x) for x in world.finished_reg_group2_delays],
        'all_reg': [float(x) for x in world.finished_reg_all_delays],
        'ev':      [float(x) for x in world.finished_ev_delays],
    }

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
    # NEW FOR LEXICOGRAPHIC MORL: total_reward[aid] is (n_objectives,), so
    # average across agents per-objective rather than blending objectives
    # together into one meaningless number.
    avg_reward_per_objective = np.mean([total_reward[aid] for aid in agent_ids], axis=0)
    avg_reward              = float(avg_reward_per_objective[0])

    return {
        'total_reward':          {aid: r.tolist() for aid, r in total_reward.items()},
        'avg_reward':            avg_reward,
        'avg_reward_per_objective': [float(x) for x in avg_reward_per_objective],
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
        # NEW: per-vehicle trip-total delay stats (the ones you actually want)
        'group1_delay_mean': g1_mean, 'group1_delay_std': g1_std, 'group1_delay_n': g1_n,
        'group2_delay_mean': g2_mean, 'group2_delay_std': g2_std, 'group2_delay_n': g2_n,
        'all_reg_delay_mean': all_mean, 'all_reg_delay_std': all_std, 'all_reg_delay_n': all_n,
        'ev_delay_mean': ev_mean, 'ev_delay_std': ev_std, 'ev_delay_n': ev_n,
        # NEW: raw per-vehicle delay values for this episode (for histogram /
        # distribution plots downstream - see plot_delay_distributions.py).
        'raw_delays': ep_raw_delays,
    }


# ============================================================================
# NEW: raw per-vehicle delay CSV export
# ============================================================================
# Written automatically alongside the JSON (same --save-results run, no
# extra step) purely so you have something easy to open in Excel and eyeball
# - which demand files went with which episode, how many vehicles landed in
# each group, spot-checking values, etc. The JSON's 'raw_delays' arrays stay
# the actual input to plot_delay_distributions.py; this CSV is a read-only
# human-friendly view of the exact same numbers, not a separate source of
# truth - so there's nothing to keep in sync and nothing to manually type.
#
# Long/"tidy" format (one row per vehicle) rather than one column per
# vehicle: Group 2 alone can be ~300-400 vehicles in a single episode, so a
# wide layout would mean hundreds of columns. In Excel you can still filter
# or pivot-table this by 'group' and 'episode' to see any slice you want.
_CSV_GROUP_LABELS = {
    'ev':      'EV',
    'group1':  'Group1',
    'group2':  'Group2',
    'all_reg': 'AllReg',
}


def _write_raw_delays_csv(all_results, csv_path):
    """One row per finished vehicle: episode, demand files for that episode,
    which group it belongs to, and its total trip delay in seconds."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'regular_demand_file', 'ev_demand_file',
                          'group', 'delay_s'])
        for i, r in enumerate(all_results):
            ep = i + 1
            reg_file = r.get('regular_demand_file', '')
            ev_file = r.get('ev_demand_file', '')
            for group_key, group_label in _CSV_GROUP_LABELS.items():
                for delay in r['raw_delays'][group_key]:
                    writer.writerow([ep, reg_file, ev_file, group_label, delay])


# ============================================================================
# Core test function
# ============================================================================

def test_model(
    model_path,
    config_path,
    scenario_dir,
    Z=None,
    num_episodes=1,
    deterministic=True,
    seed=42,
    gui=False,
    save_results=None,
    verbose=False,
    focus_agent_idx=0,
    regular_pool=None,
    ev_pool=None,
):
    """
    Evaluate a trained MAPPO-Ambulance model.

    Z,  resolution order:
        1. CLI arguments (--Z)
        2. exp_config.json next to the model file
        3. mappo_ambulance.yaml  ambulance.Z
        4. Hard-coded defaults (1.0)
    """
    regular_pool = regular_pool if regular_pool else DEFAULT_REGULAR_POOL
    ev_pool = ev_pool if ev_pool else DEFAULT_EV_POOL

    print("=" * 80)
    print("MAPPO-Ambulance Model Evaluation")
    print("=" * 80)
    print(f"Model path     : {model_path}")
    print(f"Config file    : {config_path}")
    print(f"Scenario dir   : {scenario_dir}")
    print(f"Episodes       : {num_episodes}")
    print(f"Policy         : {'deterministic' if deterministic else 'stochastic'}")
    print(f"GUI            : {'on' if gui else 'off'}")
    print(f"Regular pool   : {regular_pool}")
    print(f"EV pool        : {ev_pool}")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Load YAML config
    # ------------------------------------------------------------------
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # NEW FOR LEXICOGRAPHIC MORL: must match the training run's setting or
    # the critic's saved state_dict won't load (see the note by env_config
    # below). Simplest is to keep this flag set the same in both YAML
    # configs used for training vs. evaluating a given model.
    lex_cfg = config['algorithm'].get('lexicographic', {})
    lex_enabled = lex_cfg.get('enabled', False)
    n_objectives = lex_cfg.get('n_objectives', 2) if lex_enabled else 1

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
    print("Reward formula : -sum(lane_weights * lane_queues) # new reward function, where lane_weights = (1 + Z/EV_norm_tta)\n")

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

    # NOTE: "combined_file" is deliberately omitted here, same as
    # create_sumo_config() in CL_train_parl_mappo_ambulance.py — with it
    # present, the env falls back to the pre-built .sumocfg on disk instead
    # of the route files env.set_demand_files() swaps in per episode below,
    # which is what silently pinned every test episode to the same demand
    # files regardless of seed. "flowFile" here is just the *initial* combo
    # used for env creation/dimension probing; set_demand_files() overrides
    # it before every episode's reset().
    initial_reg = (regular_pool if regular_pool else DEFAULT_REGULAR_POOL)[0]
    initial_ev = (ev_pool if ev_pool else DEFAULT_EV_POOL)[0]
    sumo_cfg = {
        "name":             "test_mappo_emergency_ambulance", # was 'test_emergency_ambulance' historically
        "dir":              scenario_dir,
        "roadnetFile":      "3_intersection_corridor_250long.net.xml",
        "flowFile":         f"vtypes.rou.xml,{initial_reg},{initial_ev}",
        "gui":              True, # Hardwired on — run via Spyder, not CLI, so --gui flag isn't convenient. Flip to False here (or back to `gui` param) if you want it off.
        "no_warning":       True,
        "decision_interval": 5, # was 5 
        "min_green":         5, # was 5 
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
        # NEW FOR LEXICOGRAPHIC MORL: must match whatever the checkpoint was
        # actually trained with, or the critic's state_dict won't load (its
        # output head is sized n_objectives wide) and the reward-vector
        # shape agent.n_objectives expects downstream won't match either.
        "algorithm_name":        "final_year_project" + ("_lexicographic" if lex_enabled else ""),   # TODO: IMPORTANT: Sets what Observation state space is being used - change this to be fyp or final_year_project for FYP model (final_year_project_lane_mode for LANE FEATURES VERSION) OR project1_std_dqn for baseline model
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
        # NEW FOR LEXICOGRAPHIC MORL - must match training, see note above
        n_objectives=n_objectives,
        lexicographic=lex_enabled,
        lex_tolerance=lex_cfg.get('tolerance', 0.0),
        lex_dual_lr=lex_cfg.get('dual_lr', 0.05),
        lex_ema_rho=lex_cfg.get('ema_rho', 0.05),
        lex_base_weight_decay=lex_cfg.get('base_weight_decay', 0.1),
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    agent.load(model_path)
    agent.mac.agent.eval()
    print("Model loaded successfully.\n")
    
    algorithm_name = env_config["algorithm_name"]
    print("=" * 70)
    # NOTE: substring checks here, matching how Observations.py itself
    # dispatches - exact equality would misfire whenever lexicographic mode
    # appends "_lexicographic" onto the name.
    if 'project1' in algorithm_name.lower() or 'std_dqn' in algorithm_name.lower():
        print("\nUsing BASELINE RL (Kodogoda-style) observation/state space\n")
    elif 'final_year_project_lane_mode' in algorithm_name.lower():
         print("\nUsing FINAL YEAR PROJECT !LANE! VERSION observation/state space\n")
    elif 'final_year_project' in algorithm_name.lower() or 'fyp' in algorithm_name.lower():
        print("\nUsing FINAL YEAR PROJECT observation/state space\n")
        print_fyp_obs_config()
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

    # NEW: single persistent env, reused across episodes — set_demand_files()
    # and set_seed() (both must be called before reset(), same requirement
    # as the training loop) swap in a fresh (regular, ev) demand combo and
    # seed each episode, rather than recreating PARLSumoEnv (with the same
    # fixed demand files) every time.
    env = PARLSumoEnv(env_config)

    for ep in range(num_episodes):
        ep_seed = seed + ep
        reg_file, ev_file = sample_episode_demand(
            ep, regular_pool, ev_pool, rng=random.Random(seed + ep)
        )
        env.set_demand_files(reg_file, ev_file)
        env.set_seed(ep_seed)

        print(f"Episode {ep+1:3d}/{num_episodes} (seed={ep_seed}, "
              f"reg={reg_file}, ev={ev_file}) ... ", end='', flush=True)
        result = run_test_episode(env, agent, deterministic=deterministic, verbose=verbose,
                                   focus_agent_idx=focus_agent_idx)
        result['regular_demand_file'] = reg_file
        result['ev_demand_file'] = ev_file
        all_results.append(result)

        # NEW FOR LEXICOGRAPHIC MORL: show each objective separately so you
        # can read off "this run got X for EVs, Y for regular vehicles"
        # directly, instead of one blended number.
        if agent.args.get('lexicographic', False):
            reward_str = " | ".join(
                f"r{i+1}={result['avg_reward_per_objective'][i]:7.2f}"
                for i in range(agent.n_objectives)
            )
        else:
            reward_str = f"reward={result['avg_reward']:7.2f}"

        print(f"{reward_str} | "
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
    # NEW: pool raw per-vehicle delays across all episodes of this run.
    # Each all_results[i]['raw_delays'][group] is one episode's list; for a
    # histogram/fitted-curve you want every vehicle from every episode of
    # this run pooled together (that's what turns "45 EV data points per
    # run" into a distribution worth plotting).
    # ------------------------------------------------------------------
    pooled_raw_delays = {
        group: [d for r in all_results for d in r['raw_delays'][group]]
        for group in ('group1', 'group2', 'all_reg', 'ev')
    }

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    def _stats(key):
        vals = [r[key] for r in all_results]
        return float(np.mean(vals)), float(np.std(vals))

    avg_rew_mean,  avg_rew_std  = _stats('avg_reward')
    amb_mean,      amb_std      = _stats('ambulance_duration')
    civ_mean,      civ_std      = _stats('civilian_avg_trip_time')
    steps_mean,    steps_std    = _stats('steps')

    # NEW FOR LEXICOGRAPHIC MORL: mean/std of each objective's reward
    # across the evaluation episodes, e.g. reward_per_objective_stats[0]
    # = (mean, std) for r1 (EV) across all episodes.
    n_objectives = agent.n_objectives
    reward_per_objective_stats = [
        (float(np.mean([r['avg_reward_per_objective'][i] for r in all_results])),
         float(np.std([r['avg_reward_per_objective'][i] for r in all_results])))
        for i in range(n_objectives)
    ]

    g1m_m, g1m_s   = _stats('group1_delay_mean')
    g1s_m, g1s_s   = _stats('group1_delay_std')
    g2m_m, g2m_s   = _stats('group2_delay_mean')
    g2s_m, g2s_s   = _stats('group2_delay_std')
    allm_m, allm_s = _stats('all_reg_delay_mean')
    alls_m, alls_s = _stats('all_reg_delay_std')
    evm_m, evm_s   = _stats('ev_delay_mean')
    evs_m, evs_s   = _stats('ev_delay_std')

    focus_aid = all_results[0]['focus_agent_id'] if all_results else None

    print(f"\n{'='*80}")
    print("Demand files used per episode")
    print(f"{'='*80}")
    for i, r in enumerate(all_results):
        print(f"  Episode {i+1:3d}: reg={r['regular_demand_file']:<35} ev={r['ev_demand_file']}")

    print(f"\n{'='*80}")
    print("Evaluation Summary  (mean +/- std. dev. across episodes, per-vehicle trip totals)")
    print(f"{'='*80}")
    if agent.args.get('lexicographic', False):
        obj_labels = ['r1 (EV priority)', 'r2 (regular vehicles)'] + \
                     [f'r{i+1}' for i in range(2, n_objectives)]
        for i in range(n_objectives):
            m, s = reward_per_objective_stats[i]
            print(f"  Avg. {obj_labels[i]:<20}: {m:8.2f} +/- {s:.2f}")
    else:
        print(f"  Avg. reward              : {avg_rew_mean:8.2f} +/- {avg_rew_std:.2f}")
    print(f"  EV trip time         (s) : {amb_mean:8.2f} +/- {amb_std:.2f}")
    print(f"  Civilian trip time   (s) : {civ_mean:8.2f} +/- {civ_std:.2f}")
    print(f"  Avg steps/episode        : {steps_mean:8.1f} +/- {steps_std:.1f}")
    print()
    print(f"  EV delay (mean)      (s) : {evm_m:8.2f} +/- {evm_s:.2f}")
    print(f"  EV delay (std. dev.) (s) : {evs_m:8.2f} +/- {evs_s:.2f}")
    print(f"  Group 1 avg. delay   (s) : {g1m_m:8.2f} +/- {g1m_s:.2f}")
    print(f"  Group 1 delay std.   (s) : {g1s_m:8.2f} +/- {g1s_s:.2f}")
    print(f"  Group 2 avg. delay   (s) : {g2m_m:8.2f} +/- {g2m_s:.2f}")
    print(f"  Group 2 delay std.   (s) : {g2s_m:8.2f} +/- {g2s_s:.2f}")
    print(f"  All Reg. mean delay  (s) : {allm_m:8.2f} +/- {allm_s:.2f}")
    print(f"  All Reg. delay std.  (s) : {alls_m:8.2f} +/- {alls_s:.2f}")
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
        # NEW: which demand-file pools were sampled from, and the exact
        # (regular, ev) combo drawn for each episode (also inside
        # all_results, kept here too for a quick top-level glance)
        'regular_pool':          regular_pool,
        'ev_pool':               ev_pool,
        'episode_demand_files':  [
            {'episode': i + 1, 'regular': r['regular_demand_file'], 'ev': r['ev_demand_file']}
            for i, r in enumerate(all_results)
        ],
        'focus_agent_id':        focus_aid,
        # aggregated metrics
        'avg_reward_mean':       avg_rew_mean,
        'avg_reward_std':        avg_rew_std,
        # NEW FOR LEXICOGRAPHIC MORL: mean/std per objective, e.g.
        # reward_per_objective_mean[0] = r1 (EV) mean across episodes.
        'reward_per_objective_mean': [m for m, s in reward_per_objective_stats],
        'reward_per_objective_std':  [s for m, s in reward_per_objective_stats],
        'ambulance_time_mean':   amb_mean,
        'ambulance_time_std':    amb_std,
        'civilian_time_mean':    civ_mean,
        'civilian_time_std':     civ_std,
        'steps_mean':            steps_mean,
        'steps_std':             steps_std,
        'group1_delay_mean': g1m_m, 'group1_delay_std': g1m_s,
        'group1_delay_std_mean': g1s_m, 'group1_delay_std_std': g1s_s,
        'group2_delay_mean': g2m_m, 'group2_delay_std': g2m_s,
        'group2_delay_std_mean': g2s_m, 'group2_delay_std_std': g2s_s,
        'all_reg_delay_mean': allm_m, 'all_reg_delay_std': allm_s,
        'all_reg_delay_std_mean': alls_m, 'all_reg_delay_std_std': alls_s,
        'ev_delay_mean': evm_m, 'ev_delay_std': evm_s,
        'ev_delay_std_mean': evs_m, 'ev_delay_std_std': evs_s,
        # NEW: per-intersection waiting-time metrics, keyed by agent_id
        'per_intersection_summary': per_intersection_summary,
        # Snapshot of the observation feature toggles used for this test run
        # (only meaningful when algorithm_name == "final_year_project")
        'fyp_obs_config':        FYP_OBS_CONFIG,
        'fyp_obs_dims':          dict(zip(('intersection_dim', 'per_lane_dim'), get_fyp_observation_dims())),
        # NEW: raw per-vehicle delays pooled across all episodes of this run,
        # one array per group - the input plot_delay_distributions.py expects.
        'raw_delays':            pooled_raw_delays,
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

        # NEW: companion CSV, same data, for opening in Excel (see
        # _write_raw_delays_csv() docstring above for why this is safe to
        # regenerate every run rather than something you hand-edit)
        csv_path = os.path.splitext(save_results)[0] + '_raw_delays.csv'
        _write_raw_delays_csv(all_results, csv_path)
        print(f"Raw per-vehicle delays (for Excel) saved to: {csv_path}\n")

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
                        default='experiments/bestmodel_mappo_ambulance_K0.5_Z40_seed42_20260804_210135/models/agent_final.pt', # TODO: TRAINED AGENT: switch this to be path to the trained agent you wish to use (from root project folder)
                        help='Path to the .pt model checkpoint')
    
    # This one is important for the Reward Function setup (parameters) AND agent model (mappo or lmorl) along with the relevant Hyperparameters - WHICH ARE CONFIGURED WITHIN THE CONFIG .yaml FILE ITSELF!!!
    parser.add_argument('--config', type=str,
                        default='configs/tsc/mappo_fyp_config.yaml', # TODO: Use mappo_ambulance for BASELINE, mappo_fyp_config for FYP and mappo_fyp_lexicographic_config for lexicographic - from configs/tsc folder
                        help='YAML config used during training (default: configs/tsc/mappo_fyp_config.yaml)')
    parser.add_argument('--scenario-dir', type=str,
                        default='scenarios/3_intersection_corridor_250long', # change to scenario to be tested on
                        help='SUMO scenario directory')

    # Z override
    parser.add_argument('--Z', type=float, default=None,
                        help='EMV penalty multiplier Z (auto-detected from exp_config.json if omitted)')

    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of test episodes (default: 5)')
    parser.add_argument('--regular-pool', type=str, nargs='+', default=None,
                        help='Regular-traffic demand files to sample from, one per episode '
                             '(relative to --scenario-dir; default: reg_1800_v1/v2/v3)')
    parser.add_argument('--ev-pool', type=str, nargs='+', default=None,
                        help='EV demand files to sample from, one per episode '
                             '(relative to --scenario-dir; default: ev_100s_v1/v2/v3)')
    parser.add_argument('--seed', type=int, default=42, # use seed 42 as default normally
                        help='Base random seed (each episode uses seed+i)')
    parser.add_argument('--gui', action='store_true',
                        help='Show SUMO GUI')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print step-level progress every 50 steps')
    
    #                                               default=None or default='evaluations/filename_to_save_as.json'
    parser.add_argument('--save-results', type=str, default='evaluations/z=40_best_core.json', # TODO: set 'default=None' for no results saved. Set 'evaluations/[test name].json' to save some results, e.g. 'evaluations/baseline.json' to save a run and indicate you used the baseline model
                        help='Path to save JSON result file')
    
    parser.add_argument('--focus-agent-idx', type=int, default=0, # can change this default='x' x value to get a different intersection of interest in evaluation metric summary
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
        regular_pool=args.regular_pool,
        ev_pool=args.ev_pool,
    )


if __name__ == '__main__':
    main()
