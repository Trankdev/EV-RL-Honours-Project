"""
analyze_ablation_results.py

Reusable analysis for feature_ablation_grid_results.csv (or any bigger sweep
of the same shape, e.g. the full 1024-combo grid). Usage:

    python analyze_ablation_results.py path/to/results.csv

What it does:
  1. Reconstructs `composite_cost` from the SAME formula Rewards.py's 'new'
     variant actually optimizes (reward = 50 - composite_cost), using the
     logged k_value/z_value for each row. Lower composite_cost = better.
     This gives a single, principled ranking metric instead of eyeballing
     12 separate delay columns.
  2. Prints the best combo per num_features_on group.
  3. Prints a feature-importance table: mean composite_cost with the
     feature ON vs OFF across the whole grid.
  4. Fits a quick linear regression of composite_cost ~ feature bits as a
     second, interaction-free importance check.
  5. Flags any rows with status != 'done' so failed runs don't quietly
     get included (or silently excluded) from the analysis.

Caveat printed at the end: every combo in this grid was trained with the
SAME base seed (seed=42), so demand variation across the 200 training /
5 eval episodes is well controlled for and comparable across combos - but
weight init + exploration trajectory (torch.manual_seed / np.random.seed,
set once per run, not per episode) was never resampled. Each combo's score
reflects one training trajectory's luck, not an average over independent
runs - re-run top candidates at 2-3 different base seeds before trusting a
specific combo's rank.
"""
import sys
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Force working directory to project root - same depth logic as
# run_ablation_grid.py. Adjust PROJECT_ROOT_DEPTH if you put this script
# somewhere other than <project_root>/scripts/automation/.
# ---------------------------------------------------------------------------
PROJECT_ROOT_DEPTH = 3
current_file = os.path.abspath(__file__)
project_root = current_file
for _ in range(PROJECT_ROOT_DEPTH):
    project_root = os.path.dirname(project_root)
os.chdir(project_root)
print(f"Working directory set to: {os.getcwd()}")

FEATURES = [
    'phase_onehot', 'vehicle_fullness', 'waiting_mean', 'waiting_std',
    'ev_max_wait', 'downstream_occupancy', 'ev_in_lane_indicator',
    'ev_dist_to_next_veh', 'ev_speed', 'ev_distance_to_intersection',
]


def print_formula_banner(df):
    """
    Print the exact composite_cost formula and the parameter values pulled
    from this file's own k_value/z_value columns, before anything else -
    so every table below is read with the right definition of "best"
    already in view, and any (K, Z) mismatch is obvious immediately.
    """
    k_vals = df['k_value'].unique()
    z_vals = df['z_value'].unique()
    K_display = k_vals[0] if len(k_vals) == 1 else f"NOT CONSTANT: {sorted(k_vals)}"
    Z_display = z_vals[0] if len(z_vals) == 1 else f"NOT CONSTANT: {sorted(z_vals)}"

    print("=" * 70)
    print("COMPOSITE COST FORMULA (mirrors Rewards.py's 'new' variant)")
    print("=" * 70)
    print("  reward         = 50 - composite_cost")
    print("  composite_cost = (reg_mean + K * reg_std) + Z * (ev_delay_mean + K * ev_delay_std)")
    print()
    print(f"  K (k_value) = {K_display}")
    print(f"  Z (z_value) = {Z_display}")
    print()
    print("  reg_mean/reg_std   <- all_reg_delay_mean / all_reg_delay_std columns")
    print("  ev_delay_mean/std  <- ev_delay_mean / ev_delay_std columns")
    print("")
    print("  Lower composite_cost = better (= higher reward).")


def load(path):
    df = pd.read_csv(path)

    bad = df[df['status'] != 'done']
    if len(bad):
        print(f"⚠️  {len(bad)} row(s) with status != 'done' - excluding from analysis:")
        print(bad[['combo_id', 'status', 'error']].to_string(index=False))
        df = df[df['status'] == 'done'].copy()

    if df['k_value'].nunique() > 1 or df['z_value'].nunique() > 1:
        print("⚠️  k_value/z_value are NOT constant across this file - "
              "composite_cost mixes runs optimized for different objectives. "
              "Filter to one (K, Z) pair before comparing costs.")

    K = df['k_value']
    Z = df['z_value']
    df['composite_cost'] = (
        (df['all_reg_delay_mean'] + K * df['all_reg_delay_std'])
        + Z * (df['ev_delay_mean'] + K * df['ev_delay_std'])
    )
    return df


def best_per_feature_count(df):
    print("\n" + "=" * 70)
    print("BEST COMBO PER FEATURE COUNT (lowest composite_cost)")
    print("Larger cost = Worse | Lower cost = Better")
    print("=" * 70)
    for n in sorted(df['num_features_on'].unique(), reverse=True):
        sub = df[df['num_features_on'] == n].sort_values('composite_cost')
        best = sub.iloc[0]
        on = [f for f in FEATURES if best[f] == 1]
        off = [f for f in FEATURES if best[f] == 0]
        print(f"\n-- {n} features on ({len(sub)} combos tested) --")
        print(f"  combo_id       : {best['combo_id']}")
        print(f"  composite_cost : {best['composite_cost']:.2f}")
        print(f"  ev_delay_mean  : {best['ev_delay_mean']:.3f}  "
              f"reg_delay_mean : {best['all_reg_delay_mean']:.3f}")
        print(f"  OFF            : {off if off else '(none)'}")


def feature_importance(df):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE: mean composite_cost, ON vs OFF (whole grid)")
    print("Larger cost = Worse | Lower cost = Better")
    print("=" * 70)
    rows = []
    for f in FEATURES:
        on_mean = df.loc[df[f] == 1, 'composite_cost'].mean()
        off_mean = df.loc[df[f] == 0, 'composite_cost'].mean()
        rows.append((f, on_mean, off_mean, on_mean - off_mean,
                      (df[f] == 1).sum(), (df[f] == 0).sum()))
    imp = pd.DataFrame(
        rows, columns=['feature', 'mean_cost_ON', 'mean_cost_OFF',
                        'delta_ON_minus_OFF', 'n_on', 'n_off']
    ).sort_values('delta_ON_minus_OFF')
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\n(negative delta = feature ON is associated with LOWER cost, i.e. helpful)")


def linear_regression_check(df):
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION: composite_cost ~ feature bits (no interactions)")
    print("Larger cost = Worse | Lower cost = Better")
    print("=" * 70)
    X = df[FEATURES].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = df['composite_cost'].values.astype(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coefs = pd.Series(coef[1:], index=FEATURES).sort_values()
    print(coefs.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n(negative coefficient = turning this feature ON is associated with "
          "lower cost, holding the other bits fixed - a rough, interaction-free "
          "importance ranking; sanity-check against the ON/OFF table above)")


def top_n_overall(df, n=10):
    print("\n" + "=" * 70)
    print(f"TOP {n} COMBOS OVERALL (any feature count)")
    print("Larger cost = Worse | Lower cost = Better")
    print("=" * 70)
    cols = ['combo_id', 'num_features_on', 'composite_cost',
            'ev_delay_mean', 'all_reg_delay_mean']
    print(df.sort_values('composite_cost')[cols].head(n).to_string(index=False))


def main(path):
    df = load(path)
    print_formula_banner(df)
    top_n_overall(df)
    best_per_feature_count(df)
    feature_importance(df)
    linear_regression_check(df)
    print("\n" + "=" * 70)
    print("⚠️  CAVEAT: every combo here was trained with the SAME base seed "
          "(seed=42 -> np.random.seed/torch.manual_seed, set ONCE at process "
          "start - see CL_train_parl_mappo_ambulance.py). The per-episode "
          "env.set_seed(seed + episode) calls only randomize traffic demand "
          "across the 200 training / 5 eval episodes, so demand variation IS "
          "well controlled for and fairly comparable across combos - that part "
          "is fine. What was never resampled is weight initialization and the "
          "exploration trajectory: each combo's composite_cost reflects ONE "
          "training run's luck under seed 42, not an average over independent "
          "seeds. Two combos a few points apart may just differ in how lucky "
          "seed 42 happened to be for each architecture, not in which feature "
          "set is genuinely better. Re-run your top 5-10 candidates at 2-3 "
          "different base seeds (e.g. 42, 123, 7) before trusting a specific "
          "combo's rank, especially near the top where gaps are small.")
    print("=" * 70)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'feature_ablation_results/feature_ablation_grid_results.csv'
    main(path)