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
  1. TOP N OVERALL - raw sort by composite_cost. Includes a compact
     'features_on' decode column (short codes - see the legend printed
     right after the formula banner, e.g. WM+WS+EIL) and an
     'on_pareto_frontier' flag (see 8) next to every combo_id, so you're
     never hand-decoding bits or cross-referencing section 8 to know if a
     top row is a real Pareto-optimal candidate or just Z-favoured.
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
  6b. PAIRWISE INTERACTION REGRESSION - composite_cost ~ feature bits +
     all 45 pairwise products. 1023 rows vs 55 terms (10 main + 45
     pairwise) is well-determined, so this formally tests interactions
     (e.g. phase_onehot/ev_in_lane_indicator redundancy) with an actual
     coefficient instead of inferring it by eyeballing sections 4/5/7.
  7. ENRICHMENT ANALYSIS - for the top-N combos (by composite_cost), is
     each feature ON more/less often than its ~50% base rate would predict
     by chance? Includes a binomial-test p-value. This is a more principled
     version of "look at the top N and eyeball trends".
  8. PARETO-FRONTIER FLAGGING - composite_cost collapses (ev_delay,
     reg_delay) into one number via a tuned Z; combos that are
     Pareto-optimal (not beaten on BOTH objectives simultaneously by any
     other combo) are defensible regardless of the exact Z chosen, which
     hedges against Z being imperfectly tuned.
  8b. CONSTRAINED / PRACTICAL PARETO FRONTIER - the same idea as 8, but
     first excludes combos with degenerate EV delay (default: > 15s)
     before computing dominance, so a policy that's essentially ignoring
     EVs to win on regular-traffic delay can't appear as a "Pareto-optimal"
     candidate.
  9. STAGE-3 RERUN SHORTLIST - combines sections 2-8b into a concrete,
     small list of candidates (not hundreds) worth re-training at a larger
     episode budget and multiple seeds, to resolve the two confounds this
     script cannot resolve on its own (see CAVEATS at the end).
  10. GRAPHS - saved as PNGs next to the input CSV (see PLOTS_DIRNAME
     below), and also shown via plt.show() when run somewhere interactive
     (e.g. Spyder's Plots pane): feature count vs composite_cost,
     ev_delay vs reg_delay for all combos, ev_delay vs reg_delay for the
     Pareto frontier only, feature-importance bar chart, and the LOO
     ablation bar chart.

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
    Pareto-frontier sections for a Z-independent robustness check.
"""

import sys
import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import binomtest

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

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
INTERACTION_TOP_N = 15      # how many pairwise interaction terms to print
CONSTRAINED_EV_DELAY_THRESHOLD = 15.0   # seconds - "degenerate" cutoff for section 8b
PLOTS_DIRNAME = 'ablation_analysis_plots'

# Short codes used for the 'features_on' column in tables - full feature
# names would blow tables out to 150+ chars/row (see FEATURE_LEGEND, printed
# once near the top of the output, for the full-name mapping).
FEATURE_CODE = {
    'phase_onehot':                 'PH',
    'vehicle_fullness':             'VF',
    'waiting_mean':                 'WM',
    'waiting_std':                  'WS',
    'ev_max_wait':                  'EMW',
    'downstream_occupancy':         'DO',
    'ev_in_lane_indicator':         'EIL',
    'ev_dist_to_next_veh':          'EDV',
    'ev_speed':                     'ESP',
    'ev_distance_to_intersection':  'EDI',
}


def decode_features_on(row):
    """Compact ON-feature code for one combo row, e.g. 'WM+WS+EMW+EIL' -
    see FEATURE_CODE / the legend printed near the top of the output for
    what each code means."""
    on = [FEATURE_CODE[f] for f in FEATURES if row[f] == 1]
    return '+'.join(on) if on else '(none)'


def print_feature_legend():
    print("\nFeature codes used in the 'features_on' column below:")
    for f in FEATURES:
        print(f"  {FEATURE_CODE[f]:<4s} = {f}")


def compact_for_display(df_subset):
    """
    Shrinks two wide columns for printing so rows fit on one terminal line
    instead of word-wrapping mid-row: 'num_features_on' -> 'n_feat', and
    'on_pareto_frontier' (True/False) -> 'pareto' (Y/N). Only touches
    columns that are actually present, and returns a copy - never mutates
    the original df.
    """
    out = df_subset.copy()
    if 'num_features_on' in out.columns:
        out = out.rename(columns={'num_features_on': 'n_feat'})
    if 'on_pareto_frontier' in out.columns:
        out['pareto'] = out['on_pareto_frontier'].map({True: 'Y', False: 'N'})
        out = out.drop(columns=['on_pareto_frontier'])
    return out


def add_computed_columns(df):
    """
    Adds two columns used throughout the rest of the script, computed once
    up front so later sections don't redo the O(n^2) Pareto-dominance pass
    or the decode:
      - 'features_on'        human-readable ON-feature list (see section 1/8/8b/9)
      - 'on_pareto_frontier' bool, True if not simultaneously beaten on both
                              ev_delay_mean AND all_reg_delay_mean by any
                              other combo in the full grid (see section 8)
    """
    df['features_on'] = df.apply(decode_features_on, axis=1)

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
    df['on_pareto_frontier'] = ~is_dominated
    return df


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
    cols = ['combo_id', 'features_on', 'num_features_on', 'composite_cost',
            'ev_delay_mean', 'all_reg_delay_mean', 'on_pareto_frontier']
    disp = compact_for_display(df.sort_values('composite_cost')[cols].head(n))
    print(disp.to_string(index=False))
    print("\n(pareto = Y means this combo is NOT beaten on both ev_delay_mean "
          "AND all_reg_delay_mean simultaneously by any other combo - see "
          "section 8. A top-by-composite-cost row with pareto = N is only "
          "'best' under this specific Z; some other combo dominates it on "
          "both objectives at once.)")


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
        n_combos='count',
        mean_composite_cost='mean',
        median_composite_cost='median',
        std_composite_cost='std',
        min_composite_cost='min',
    )
    print("(all stats below are of composite_cost, in composite_cost units - "
          "see the formula banner above)\n")
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
                                       f'expected_best_composite_cost_of_{k}',
                                       'raw_min_composite_cost'])
    print("(both columns are composite_cost values, in composite_cost units)\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}",
                         na_rep='n/a (too few combos to resample)'))


def leave_one_out_table(df):
    """
    (a) LOO ablation table, built directly from the num_features_on == 9
    group: each of those combos is the full 10-feature model with exactly
    one feature removed. This IS the standard ablation-table format used
    throughout the RL/traffic-signal-control literature.

    Returns the computed DataFrame (or None if the table was skipped) so
    section 10 can reuse it for the LOO bar chart without recomputing.
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
        return None

    full_cost = full.iloc[0]['composite_cost']
    print(f"Full-model (all 10 features) composite_cost = {full_cost:.2f}\n")

    rows = []
    for _, row in loo.iterrows():
        off_feats = [f for f in FEATURES if row[f] == 0]
        removed = off_feats[0] if len(off_feats) == 1 else f"AMBIGUOUS: {off_feats}"
        rows.append((removed, row['composite_cost'], row['composite_cost'] - full_cost))
    out = pd.DataFrame(rows, columns=['feature_removed', 'composite_cost',
                                       'delta_composite_cost_vs_full'])
    out = out.sort_values('delta_composite_cost_vs_full', ascending=False)
    print("(composite_cost = score of the 10-feature model with this one "
          "feature removed; delta = that minus the full-model score above)\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\n(positive delta = removing this feature made things WORSE, i.e. "
          "the feature is helpful; negative delta = removing it IMPROVED "
          "the score, i.e. the feature may be net-harmful or redundant here)")
    return out


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
        rows, columns=['feature', 'mean_composite_cost_ON', 'mean_composite_cost_OFF',
                        'delta_ON_minus_OFF', 'n_on', 'n_off']
    ).sort_values('delta_ON_minus_OFF')
    print("(mean_composite_cost_ON/OFF = average composite_cost across all "
          "combos with this feature ON vs OFF)\n")
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\n(negative delta = feature ON is associated with LOWER cost, i.e. helpful)")


def linear_regression_check(df):
    """Returns the fitted main-effect coefficients (Series indexed by
    FEATURES) so section 10 can reuse them for the feature-importance bar
    chart without refitting."""
    print("\n" + "=" * 70)
    print("6. LINEAR REGRESSION: composite_cost ~ feature bits (no interactions)")
    print("Larger cost = Worse | Lower cost = Better")
    print("=" * 70)
    X = df[FEATURES].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = df['composite_cost'].values.astype(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coefs = pd.Series(coef[1:], index=FEATURES).sort_values()
    print("(each value = estimated change in composite_cost from turning that "
          "feature ON, holding all other features fixed - same units as "
          "composite_cost, e.g. -845 means ~845 lower/better composite_cost)\n")
    print(coefs.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n(negative coefficient = turning this feature ON is associated with "
          "lower cost, holding the other bits fixed - a rough, interaction-free "
          "importance ranking; sanity-check against section 5 above)")
    return coefs


def interaction_regression(df, n_top=INTERACTION_TOP_N):
    """
    Second regression: composite_cost ~ 10 main effects + all 45 pairwise
    products (feature_i * feature_j for i<j). 1023 rows vs 55 terms is
    well-determined, so this puts an actual number on interaction stories
    (e.g. phase_onehot/ev_in_lane_indicator) instead of inferring them by
    eyeballing sections 4/5/7 the way the qualitative writeup did.
    """
    print("\n" + "=" * 70)
    print("6b. PAIRWISE INTERACTION REGRESSION "
          "(main effects + all 45 pairwise terms)")
    print("=" * 70)

    pairs = list(itertools.combinations(FEATURES, 2))
    n = len(df)
    X_main = df[FEATURES].values.astype(float)
    X_inter = np.column_stack([
        df[a].values.astype(float) * df[b].values.astype(float) for a, b in pairs
    ])
    X = np.column_stack([np.ones(n), X_main, X_inter])
    y = df['composite_cost'].values.astype(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    main_coefs = pd.Series(coef[1:1 + len(FEATURES)], index=FEATURES)
    inter_names = [f"{a}  x  {b}" for a, b in pairs]
    inter_coefs = pd.Series(coef[1 + len(FEATURES):], index=inter_names)

    print(f"Design: {n} rows, {1 + len(FEATURES) + len(pairs)} terms "
          f"(1 intercept + {len(FEATURES)} main effects + {len(pairs)} pairwise).\n")

    print("Main effects (co-estimated with interactions - compare to section 6):")
    print(main_coefs.sort_values().to_string(float_format=lambda x: f"{x:+.2f}"))

    top_inter = inter_coefs.reindex(inter_coefs.abs().sort_values(ascending=False).index).head(n_top)
    print(f"\nTop {n_top} pairwise interactions by |coefficient| (of {len(pairs)} total):")
    print(top_inter.to_string(float_format=lambda x: f"{x:+.2f}"))

    key = "phase_onehot  x  ev_in_lane_indicator"
    if key in inter_coefs.index:
        print(f"\n{key}: {inter_coefs[key]:+.2f}")
        print("(this is the formal test of the phase_onehot/ev_in_lane_indicator "
              "redundancy story from sections 4/5/7 - a positive value means "
              "having BOTH on together costs more than the sum of their two "
              "separate main effects, i.e. redundancy/diminishing returns once "
              "both are present; a negative value would mean synergy instead.)")

    print("\n(each interaction coefficient = the EXTRA change in composite_cost "
          "when both features in the pair are ON together, beyond what their "
          "two main effects alone would predict, holding every other term "
          "fixed. With 55 correlated terms on this many rows, treat individual "
          "coefficients as suggestive rather than precise - useful for "
          "confirming a direction/sign, not for citing an exact magnitude.)")
    return main_coefs, inter_coefs


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
    print("(base_rate/top_n_rate = fraction of combos with this feature ON, "
          "i.e. NOT composite_cost - base_rate is across all 1023 combos, "
          f"top_n_rate is within just the top {n} by composite_cost)\n")
    print("A low p_value (~0) corresponds to high confidence of an association/trend. While a high p_value (~1) means no detectable association/trend. Thus, partial p_values are more 'suggestive, not proven'\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n(top_n_rate >> base_rate with small p_value = feature is enriched "
          f"i.e. over-represented among the best {n} combos - a stronger "
          f"signal than eyeballing raw top-N rows; p_value < 0.05 is a rough "
          f"conventional threshold, treat as descriptive here given multiple "
          f"comparisons across 10 features)")


def pareto_frontier(df):
    """
    (c) Print the Pareto-frontier explanation and table. The dominance
    computation itself was already done once in add_computed_columns() -
    this just reads df['on_pareto_frontier'] rather than recomputing the
    O(n^2) pass a second time.
    """
    print("\n" + "=" * 70)
    print("8. PARETO-FRONTIER FLAGGING (ev_delay_mean vs. all_reg_delay_mean, "
          "both minimized)")
    print("=" * 70)
    n = len(df)
    frontier = df.loc[df['on_pareto_frontier']].sort_values('ev_delay_mean')
    print(f"{len(frontier)} of {n} combos are Pareto-optimal. Pareto-optimal refers to 'not beaten on both axes simultaneously.'\n")
    print("This is useful for seeing the trade-offs and how further decreases in delay of one group results in substantial delay increases for the other group.\n")
    cols = ['combo_id', 'features_on', 'num_features_on', 'ev_delay_mean',
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


def constrained_pareto_frontier(df, frontier, ev_threshold=CONSTRAINED_EV_DELAY_THRESHOLD):
    """
    (d) A second, "practical" frontier: first drop any combo with
    ev_delay_mean above ev_threshold (degenerate - essentially ignoring
    EVs to win on regular-traffic delay), THEN compute Pareto dominance
    among what's left. This is what addresses the two catastrophic-EV-delay
    frontier points flagged earlier (e.g. combos with 25-35s EV delay that
    are technically non-dominated but not real candidates).
    """
    print("\n" + "=" * 70)
    print("8b. CONSTRAINED / PRACTICAL PARETO FRONTIER "
          f"(excludes ev_delay_mean > {ev_threshold:.0f}s as degenerate)")
    print("=" * 70)

    sub = df[df['ev_delay_mean'] <= ev_threshold].copy()
    excluded = df[df['ev_delay_mean'] > ev_threshold]
    print(f"{len(excluded)} of {len(df)} combos excluded outright for "
          f"ev_delay_mean > {ev_threshold:.0f}s before dominance is even computed.\n")

    ev = sub['ev_delay_mean'].values
    reg = sub['all_reg_delay_mean'].values
    n = len(sub)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        dominated_by_other = ((ev <= ev[i]) & (reg <= reg[i]) &
                               ((ev < ev[i]) | (reg < reg[i])))
        dominated_by_other[i] = False
        if dominated_by_other.any():
            is_dominated[i] = True
    constrained = sub.loc[~is_dominated].sort_values('ev_delay_mean')

    print(f"{len(constrained)} of {n} threshold-qualifying combos are "
          f"Pareto-optimal within this constrained set.\n")
    cols = ['combo_id', 'features_on', 'num_features_on', 'ev_delay_mean',
            'all_reg_delay_mean', 'composite_cost']
    print(constrained[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    dropped = frontier[frontier['ev_delay_mean'] > ev_threshold]
    if len(dropped):
        print(f"\n{len(dropped)} combo(s) were on the FULL Pareto frontier (section 8) "
              f"but are excluded here as degenerate:")
        print(dropped[['combo_id', 'features_on', 'ev_delay_mean', 'all_reg_delay_mean']]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("\nNo full-frontier combos were excluded - the full and constrained "
              "frontiers agree on membership at this threshold.")

    return constrained


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
    # on_pareto_frontier is already a column on df (from add_computed_columns),
    # carried through automatically - no need to recompute via isin() here.
    cols = ['combo_id', 'features_on', 'num_features_on', 'composite_cost',
            'ev_delay_mean', 'all_reg_delay_mean', 'on_pareto_frontier']
    disp = compact_for_display(shortlist[cols])
    print(disp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n{len(shortlist)} combos total - a feasible size for a "
          f"confirmatory rerun batch (e.g. 500 episodes x 2-3 seeds each), "
          f"vs. re-running all {len(df)}.")


def make_plots(df, frontier, constrained_frontier, coefs, loo_df, csv_path):
    """
    Saves PNGs to <folder containing the CSV>/PLOTS_DIRNAME/, and also
    calls plt.show() so they appear in an interactive session (e.g.
    Spyder's Plots pane). Silently no-ops if matplotlib isn't installed.
    """
    print("\n" + "=" * 70)
    print("10. GRAPHS")
    print("=" * 70)
    if not HAVE_MPL:
        print("⚠️  matplotlib not installed - skipping graphs "
              "(pip install matplotlib to enable this section).")
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), PLOTS_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        saved.append(path)
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)

    # --- 1. Feature count vs composite cost ---------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = sorted(df['num_features_on'].unique())
    groups = [df.loc[df['num_features_on'] == n, 'composite_cost'].values for n in counts]
    bp = ax.boxplot(groups, tick_labels=[str(c) for c in counts], showmeans=True)
    ax.set_xlabel('Number of features ON')
    ax.set_ylabel('composite_cost (lower = better)')
    ax.set_title('Composite cost distribution by feature count')
    # Legend explaining the boxplot anatomy - matplotlib's defaults (orange
    # median line, green triangle mean marker) aren't self-explanatory.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    box_color = bp['boxes'][0].get_color()
    median_color = bp['medians'][0].get_color()
    mean_marker = bp['means'][0]
    legend_handles = [
        Patch(facecolor='none', edgecolor=box_color, label='Box = IQR (25th-75th percentile)'),
        Line2D([0], [0], color=median_color, label='Median'),
        Line2D([0], [0], marker=mean_marker.get_marker(), color='w',
               markerfacecolor=mean_marker.get_markerfacecolor(),
               markeredgecolor=mean_marker.get_markeredgecolor(),
               markersize=8, label='Mean'),
        Line2D([0], [0], color=box_color, linestyle='--', label='Whiskers (within 1.5x IQR)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=box_color, markersize=6, label='Outliers'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    _save(fig, 'feature_count_vs_composite_cost.png')

    # --- 2. EV delay vs reg delay, all combos --------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    non_frontier = df.loc[~df['on_pareto_frontier']]
    ax.scatter(non_frontier['ev_delay_mean'], non_frontier['all_reg_delay_mean'],
               s=10, alpha=0.25, color='gray', label='Dominated')
    ax.scatter(frontier['ev_delay_mean'], frontier['all_reg_delay_mean'],
               s=35, color='crimson', label='Pareto-optimal (full frontier)')
    ax.set_xlabel('EV delay mean (s)')
    ax.set_ylabel('All regular-vehicle delay mean (s)')
    ax.set_title('EV delay vs. regular-vehicle delay - all combos')
    ax.legend()
    _save(fig, 'ev_vs_reg_delay_all_combos.png')

    # --- 3. EV delay vs reg delay, Pareto frontier only ----------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    frontier_sorted = frontier.sort_values('ev_delay_mean')
    ax.plot(frontier_sorted['ev_delay_mean'], frontier_sorted['all_reg_delay_mean'],
            '-o', color='crimson', label='Full Pareto frontier')
    constrained_ids = set(constrained_frontier['combo_id'])
    degenerate = frontier_sorted[~frontier_sorted['combo_id'].isin(constrained_ids)]
    if len(degenerate):
        ax.scatter(degenerate['ev_delay_mean'], degenerate['all_reg_delay_mean'],
                   marker='x', s=90, color='black', linewidths=2,
                   label=f'Excluded as degenerate (>{CONSTRAINED_EV_DELAY_THRESHOLD:.0f}s EV delay)')
    ax.set_xlabel('EV delay mean (s)')
    ax.set_ylabel('All regular-vehicle delay mean (s)')
    ax.set_title('Pareto frontier: EV delay vs. regular-vehicle delay trade-off')
    ax.legend()
    _save(fig, 'ev_vs_reg_delay_pareto_frontier.png')
    
    # --- 3.1 EV delay vs reg delay, Pareto frontier only showing (0,0) ----------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    
    frontier_sorted = frontier.sort_values('ev_delay_mean')
    
    x = frontier_sorted['ev_delay_mean'].tolist()
    y = frontier_sorted['all_reg_delay_mean'].tolist()
    
    ax.plot(x, y, '-o', color='crimson', label='Full Pareto frontier')
    
    constrained_ids = set(constrained_frontier['combo_id'])
    degenerate = frontier_sorted[~frontier_sorted['combo_id'].isin(constrained_ids)]
    
    if len(degenerate):
        ax.scatter(degenerate['ev_delay_mean'], degenerate['all_reg_delay_mean'],
                   marker='x', s=90, color='black', linewidths=2,
                   label=f'Excluded as degenerate (>{CONSTRAINED_EV_DELAY_THRESHOLD:.0f}s EV delay)')
    
    ax.set_xlabel('EV delay mean (s)')
    ax.set_ylabel('All regular-vehicle delay mean (s)')
    ax.set_title('Pareto frontier: EV delay vs. regular-vehicle delay trade-off')
    ax.legend()
    
    # Make sure (0, 0) is visible
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    _save(fig, 'ev_vs_reg_delay_pareto_frontier_origin_shown.png')

    # --- 4. Feature importance (linear regression main effects) -------------
    fig, ax = plt.subplots(figsize=(9, 6))
    coefs_sorted = coefs.sort_values()
    colors = ['seagreen' if v < 0 else 'indianred' for v in coefs_sorted.values]
    ax.barh(coefs_sorted.index, coefs_sorted.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Regression coefficient (Δ composite_cost from turning ON)')
    ax.set_title('Feature importance - linear regression main effects\n'
                  '(green = helpful, red = harmful)')
    _save(fig, 'feature_importance_regression.png')

    # --- 5. LOO ablation bar chart -------------------------------------------
    if loo_df is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        loo_sorted = loo_df.sort_values('delta_composite_cost_vs_full')
        colors = ['indianred' if v > 0 else 'seagreen'
                  for v in loo_sorted['delta_composite_cost_vs_full']]
        ax.barh(loo_sorted['feature_removed'], loo_sorted['delta_composite_cost_vs_full'],
                color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Δ composite_cost when this feature is removed from the full model')
        ax.set_title('Leave-one-out ablation\n(red = feature is helpful, green = removing it improved the score)')
        _save(fig, 'loo_ablation.png')

    print(f"Saved {len(saved)} plot(s) to: {out_dir}")
    for p in saved:
        print(f"  - {os.path.basename(p)}")


def main(path):
    df = load(path)
    add_computed_columns(df)
    print_formula_banner(df)
    print_feature_legend()
    top_n_overall(df)
    per_feature_count_summary(df)
    bootstrapped_best_of_k(df)
    loo_df = leave_one_out_table(df)
    feature_importance(df)
    coefs = linear_regression_check(df)
    interaction_regression(df)
    enrichment_analysis(df)
    frontier = pareto_frontier(df)
    constrained_frontier = constrained_pareto_frontier(df, frontier)
    stage3_shortlist(df, frontier)
    make_plots(df, frontier, constrained_frontier, coefs, loo_df, path)

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