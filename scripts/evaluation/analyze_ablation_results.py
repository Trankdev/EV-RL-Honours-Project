"""
analyze_ablation_results.py

Reusable analysis for feature_ablation_grid_results.csv - built for the full
1023/1024-combo feature-ablation grid (a near-complete 2^10 factorial over
the 10 FYP_OBS_CONFIG toggles), but works on any subset of the same shape.

Usage:
    python analyze_ablation_results.py path/to/results.csv

Sections, in print order:
  0. COMPOSITE COST FORMULA - the exact formula + (K, Z) actually used,
     printed first so every table below is read against the right
     definition of "best".
  1. TOP N OVERALL - raw sort by composite_cost, for a quick look.
  2. PER-FEATURE-COUNT SUMMARY - mean/median/std/min per num_features_on,
     NOT just the raw min - see the group-size-bias note below.
  3. BOOTSTRAPPED "BEST OF K DRAWS" PER FEATURE COUNT - corrects for the
     fact that num_features_on groups have wildly different combo counts
     (e.g. C(10,3)=120 vs C(10,9)=10 vs C(10,10)=1), so a raw min-per-group
     comparison is biased toward whichever group had more chances to draw
     a lucky low value. This equalizes the number of draws before
     comparing, giving a fairer "expected best" per feature count.
  4. LEAVE-ONE-OUT (LOO) ABLATION TABLE - uses the num_features_on == 9
     group directly: each of those 10 combos is the full 10-feature model
     with exactly one feature removed, which IS the standard ablation-table
     format used throughout the RL/traffic-signal-control literature
     (remove one component, measure the degradation). Sorted by impact.
  5. FEATURE IMPORTANCE (whole-grid ON/OFF means) - marginal effect of each
     feature across the full factorial design, unaffected by the
     group-size bias in section 2/3 since it marginalizes over everything.
  6. LINEAR REGRESSION CHECK - composite_cost ~ feature bits, a second,
     interaction-free importance ranking to sanity-check section 5 against.
  7. ENRICHMENT ANALYSIS - for the top-N combos (by composite_cost), is
     each feature ON more/less often than its ~50% base rate would predict
     by chance? Includes a binomial-test p-value. This is a more principled
     version of "look at the top N and eyeball trends".
  8. PARETO-FRONTIER FLAGGING - composite_cost collapses (ev_delay,
     reg_delay) into one number via a tuned Z; combos that are
     Pareto-optimal (not beaten on BOTH objectives simultaneously by any
     other combo) are defensible regardless of the exact Z chosen, which
     hedges against Z being imperfectly tuned.
  9. STAGE-3 RERUN SHORTLIST - combines sections 2-8 into a concrete,
     small list of candidates (not hundreds) worth re-training at a larger
     episode budget and multiple seeds, to resolve the two confounds this
     script cannot resolve on its own (see CAVEATS at the end).

CAVEATS (also printed at the end, read them):
  - Every combo was trained with the SAME base seed (seed=42 ->
    np.random.seed/torch.manual_seed, set ONCE at process start - see
    CL_train_parl_mappo_ambulance.py). Per-episode env.set_seed(seed+episode)
    only randomizes traffic demand across the 200 training / 5 eval
    episodes, so demand variation IS well controlled for and fairly
    comparable across combos. What was never resampled is weight
    initialization + exploration trajectory - each combo's score reflects
    ONE training run's luck under seed 42, not an average over independent
    seeds.
  - All combos were trained for a FIXED 200-episode budget. Larger
    observation spaces may need more samples to reach their own optimum
    ("state-space explosion" / sample-efficiency degradation is a known
    DRL phenomenon), so this grid measures "performance achievable within
    200 episodes", not each feature set's true ceiling. A combo that looks
    worse here might just be under-converged, not genuinely inferior.
  - composite_cost collapses two objectives via a single tuned Z - see the
    Pareto-frontier section for a Z-independent robustness check.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import binomtest

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

TOP_N = 10                  # for the raw top-N overall table
ENRICHMENT_TOP_N = 50       # how many top combos to check for feature enrichment
BOOTSTRAP_K = 10            # draws per group in the "best-of-k" correction
BOOTSTRAP_REPS = 2000
SHORTLIST_PER_GROUP = 3     # candidates per feature count for the Stage-3 shortlist
SHORTLIST_GROUPS = [5, 6, 7, 8, 9, 10]


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

    n_expected = 2 ** len(FEATURES)
    n_have = len(df)
    print(f"\nLoaded {n_have}/{n_expected} combos "
          f"({'FULL' if n_have == n_expected else 'PARTIAL'} 2^{len(FEATURES)} factorial design).")
    return df


def top_n_overall(df, n=TOP_N):
    print("\n" + "=" * 70)
    print(f"1. TOP {n} COMBOS OVERALL (any feature count)")
    print("Larger cost = Worse | Lower cost = Better")
    print("=" * 70)
    cols = ['combo_id', 'num_features_on', 'composite_cost',
            'ev_delay_mean', 'all_reg_delay_mean']
    print(df.sort_values('composite_cost')[cols].head(n).to_string(index=False))


def per_feature_count_summary(df):
    print("\n" + "=" * 70)
    print("2. PER-FEATURE-COUNT SUMMARY (mean/median, not just raw min)")
    print("=" * 70)
    print("⚠️  Groups have very different combo counts (e.g. C(10,3)=120 vs "
          "C(10,9)=10 vs C(10,10)=1). The raw MIN column below is biased "
          "toward large groups purely because they had more draws - use "
          "MEAN/MEDIAN as the primary comparison, and see section 3 for a "
          "group-size-corrected 'expected best' figure.\n")
    g = df.groupby('num_features_on')['composite_cost'].agg(
        n_combos='count', mean='mean', median='median', std='std', min='min'
    )
    print(g.to_string(float_format=lambda x: f"{x:.1f}"))


def bootstrapped_best_of_k(df, k=BOOTSTRAP_K, reps=BOOTSTRAP_REPS, seed=0):
    print("\n" + "=" * 70)
    print(f"3. GROUP-SIZE-CORRECTED COMPARISON: expected best of {k} random "
          f"draws per feature count ({reps} bootstrap reps)")
    print("=" * 70)
    print("Equalizes the number of draws per group before comparing, so "
          "groups with more total combos don't win purely by having had "
          "more chances at a lucky low value.\n")
    rng = np.random.default_rng(seed)
    rows = []
    for n in sorted(df['num_features_on'].unique()):
        vals = df.loc[df['num_features_on'] == n, 'composite_cost'].values
        if len(vals) <= k:
            rows.append((n, len(vals), None, vals.min()))
            continue
        mins = [rng.choice(vals, size=k, replace=False).min() for _ in range(reps)]
        rows.append((n, len(vals), np.mean(mins), vals.min()))
    out = pd.DataFrame(rows, columns=['num_features_on', 'n_combos',
                                       f'expected_best_of_{k}', 'raw_min'])
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}",
                         na_rep='n/a (too few combos to resample)'))


def leave_one_out_table(df):
    """
    (a) LOO ablation table, built directly from the num_features_on == 9
    group: each of those combos is the full 10-feature model with exactly
    one feature removed. This IS the standard ablation-table format used
    throughout the RL/traffic-signal-control literature.
    """
    print("\n" + "=" * 70)
    print("4. LEAVE-ONE-OUT (LOO) ABLATION TABLE")
    print("Full model (10 features) vs. each feature removed individually")
    print("=" * 70)

    full = df[df['num_features_on'] == 10]
    loo = df[df['num_features_on'] == 9]
    if len(full) != 1 or len(loo) == 0:
        print("⚠️  Need exactly 1 combo at num_features_on=10 and >=1 at "
              "num_features_on=9 for this table - skipping "
              f"(have {len(full)} and {len(loo)} respectively).")
        return

    full_cost = full.iloc[0]['composite_cost']
    print(f"Full-model (all 10 features) composite_cost = {full_cost:.2f}\n")

    rows = []
    for _, row in loo.iterrows():
        off_feats = [f for f in FEATURES if row[f] == 0]
        removed = off_feats[0] if len(off_feats) == 1 else f"AMBIGUOUS: {off_feats}"
        rows.append((removed, row['composite_cost'], row['composite_cost'] - full_cost))
    out = pd.DataFrame(rows, columns=['feature_removed', 'composite_cost', 'delta_vs_full'])
    out = out.sort_values('delta_vs_full', ascending=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\n(positive delta = removing this feature made things WORSE, i.e. "
          "the feature is helpful; negative delta = removing it IMPROVED "
          "the score, i.e. the feature may be net-harmful or redundant here)")


def feature_importance(df):
    print("\n" + "=" * 70)
    print("5. FEATURE IMPORTANCE: mean composite_cost, ON vs OFF (whole grid)")
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
    print("6. LINEAR REGRESSION: composite_cost ~ feature bits (no interactions)")
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
          "importance ranking; sanity-check against section 5 above)")


def enrichment_analysis(df, n=ENRICHMENT_TOP_N):
    """
    (b) For the top-N combos by composite_cost, is each feature ON more or
    less often than its ~50% base rate (guaranteed by the factorial design)
    would predict by chance? Binomial test p-value included.
    """
    print("\n" + "=" * 70)
    print(f"7. ENRICHMENT ANALYSIS: feature ON-rate in top {n} combos vs. "
          f"~50% base rate")
    print("=" * 70)
    top = df.sort_values('composite_cost').head(n)
    rows = []
    for f in FEATURES:
        base_rate = df[f].mean()  # should be ~0.5 in a full factorial design
        k_on = int(top[f].sum())
        observed_rate = k_on / n
        p = binomtest(k_on, n, base_rate, alternative='two-sided').pvalue
        rows.append((f, base_rate, observed_rate, k_on, n - k_on, p))
    out = pd.DataFrame(rows, columns=['feature', 'base_rate', 'top_n_rate',
                                       'n_on', 'n_off', 'p_value'])
    out = out.sort_values('p_value')
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n(top_n_rate >> base_rate with small p_value = feature is enriched "
          f"i.e. over-represented among the best {n} combos - a stronger "
          f"signal than eyeballing raw top-N rows; p_value < 0.05 is a rough "
          f"conventional threshold, treat as descriptive here given multiple "
          f"comparisons across 10 features)")


def pareto_frontier(df):
    """
    (c) Flag combos that are Pareto-optimal on (ev_delay_mean, all_reg_delay_mean)
    - not simultaneously beaten on BOTH objectives by any other combo. These
    are defensible choices regardless of the exact Z used to build
    composite_cost.
    """
    print("\n" + "=" * 70)
    print("8. PARETO-FRONTIER FLAGGING (ev_delay_mean vs. all_reg_delay_mean, "
          "both minimized)")
    print("=" * 70)
    ev = df['ev_delay_mean'].values
    reg = df['all_reg_delay_mean'].values
    n = len(df)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated_by_other = ((ev <= ev[i]) & (reg <= reg[i]) &
                               ((ev < ev[i]) | (reg < reg[i])))
        dominated_by_other[i] = False
        if dominated_by_other.any():
            is_dominated[i] = True

    frontier = df.loc[~is_dominated].sort_values('ev_delay_mean')
    print(f"{len(frontier)} of {n} combos are Pareto-optimal.\n")
    cols = ['combo_id', 'num_features_on', 'ev_delay_mean',
            'all_reg_delay_mean', 'composite_cost']
    print(frontier[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_composite_id = df.loc[df['composite_cost'].idxmin(), 'combo_id']
    on_frontier = best_composite_id in frontier['combo_id'].values
    print(f"\nComposite-cost-best combo ({best_composite_id}) is "
          f"{'ON' if on_frontier else 'NOT ON'} the Pareto frontier.")
    if not on_frontier:
        print("⚠️  This means some other combo beats it on BOTH objectives "
              "simultaneously - worth double-checking Z is well-tuned, or "
              "just note the composite-best still involves this tradeoff.")
    return frontier


def stage3_shortlist(df, frontier):
    """
    Combine per-feature-count bests with Pareto-frontier membership into a
    small, concrete shortlist worth re-training at a larger episode budget
    and multiple seeds - resolving the two confounds this script cannot
    resolve on its own (fixed 200-episode budget, single seed per combo).
    """
    print("\n" + "=" * 70)
    print("9. STAGE-3 RERUN SHORTLIST (candidates for multi-seed, "
          "larger-budget confirmatory reruns)")
    print("=" * 70)
    print(f"Top {SHORTLIST_PER_GROUP} per feature count in {SHORTLIST_GROUPS}, "
          "plus any Pareto-optimal combos not already included.\n")

    picks = []
    for n in SHORTLIST_GROUPS:
        sub = df[df['num_features_on'] == n].sort_values('composite_cost')
        picks.append(sub.head(SHORTLIST_PER_GROUP))
    shortlist = pd.concat(picks).drop_duplicates(subset='combo_id')

    frontier_extra = frontier[~frontier['combo_id'].isin(shortlist['combo_id'])]
    if len(frontier_extra):
        shortlist = pd.concat([shortlist, frontier_extra])

    shortlist = shortlist.sort_values(['num_features_on', 'composite_cost'])
    shortlist['on_pareto_frontier'] = shortlist['combo_id'].isin(frontier['combo_id'])
    cols = ['combo_id', 'num_features_on', 'composite_cost',
            'ev_delay_mean', 'all_reg_delay_mean', 'on_pareto_frontier']
    print(shortlist[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n{len(shortlist)} combos total - a feasible size for a "
          f"confirmatory rerun batch (e.g. 500 episodes x 2-3 seeds each), "
          f"vs. re-running all {len(df)}.")


def main(path):
    df = load(path)
    print_formula_banner(df)
    top_n_overall(df)
    per_feature_count_summary(df)
    bootstrapped_best_of_k(df)
    leave_one_out_table(df)
    feature_importance(df)
    linear_regression_check(df)
    enrichment_analysis(df)
    frontier = pareto_frontier(df)
    stage3_shortlist(df, frontier)

    print("\n" + "=" * 70)
    print("CAVEATS")
    print("=" * 70)
    print("⚠️  SEED: every combo here was trained with the SAME base seed "
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
          "set is genuinely better.")
    print()
    print("⚠️  TRAINING BUDGET: all combos were trained for a FIXED 200-episode "
          "budget. Larger observation spaces may need more samples to reach "
          "their own optimum (a known DRL phenomenon), so this grid measures "
          "performance achievable within 200 episodes, not each feature set's "
          "true ceiling - a combo that looks worse here might just be "
          "under-converged, not genuinely inferior.")
    print()
    print("⚠️  GROUP-SIZE BIAS: num_features_on groups have very different "
          "combo counts (120 at n=3 vs 1 at n=10) - raw 'best per group' "
          "comparisons are biased toward large groups; use sections 2-3 "
          "instead of picking on raw minimums alone.")
    print()
    print("→  See section 9 for a concrete, small shortlist to re-run at a "
          "larger budget and multiple seeds (e.g. 42, 123, 7) before "
          "trusting any specific combo's final rank.")
    print("=" * 70)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'feature_ablation_results_NewRewZ=35/feature_ablation_grid_results.csv'
    main(path)