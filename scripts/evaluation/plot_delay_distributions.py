"""
plot_delay_distributions.py

Turns the per-vehicle delay data saved by var_test_mappo_ambulance.py
(--save-results ...json) into histogram + fitted-curve graphs, replacing the
old "mean +/- std" tables for the FYP report.

Compares exactly TWO runs at a time - baseline vs. one ablation variant -
and produces one PNG per delay group (EV, Group 1, Group 2, All Regular).
For a full ablation progression (baseline -> Core v1 -> Core v2 -> final),
just re-run this script once per step, changing VARIANT_LABEL/VARIANT_PATH
below each time - same as how you already swap MODEL_PATH's default in
var_test_mappo_ambulance.py between runs.

Run it exactly like your other scripts - hit Run in Spyder, no command line
needed. Just edit the config block below first.

Requires: numpy, scipy, matplotlib (matplotlib is already a project
dependency - see CL_train_parl_mappo_ambulance.py; scipy is only used here
for the KDE fit - `pip install scipy` if you don't already have it).
"""

import os
import json

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------
# Force working directory to project root (two levels up)
# ---------------------------------------------------------
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
os.chdir(project_root)        # change Python working directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # add root to Python path

print(f"Working directory set to: {os.getcwd()}\n")

# ============================================================================
# CONFIG - edit these for each ablation step, then hit Run
# ============================================================================
BASELINE_LABEL = "Baseline"
BASELINE_PATH  = "evaluations/baseline.json"          # TODO: path to baseline --save-results JSON

VARIANT_LABEL  = "Z=40_best_core"                             # TODO: name shown in the legend/filenames
VARIANT_PATH   = "evaluations/z=40_best_core.json"            # TODO: path to this variant's --save-results JSON

OUT_DIR = "evaluations/plots/baseline_vs_z=40_best_core"      # TODO: change per ablation step, or PNGs overwrite

MAX_BINS = 30            # bin count now scales down automatically for small-n
                          # groups (e.g. EV, ~30-70 points) instead of always
                          # using this many - this is just the ceiling for
                          # large-n groups (Group 2 / All Reg, ~2000 points)
MIN_BINS = 30
SHOW_MEAN_LINES = True   # dotted vertical line + mean value in the legend
CROP_PERCENTILE = 99     # x-axis view is zoomed to this percentile of the
                          # pooled data so a handful of extreme-delay vehicles
                          # don't stretch the whole plot flat; those vehicles
                          # are still counted in the histogram/KDE math, just
                          # not inside the visible window (noted on the plot)

# ============================================================================
# Group metadata: JSON key -> (display title, x-axis label)
# ============================================================================
GROUPS = {
    'ev':      ('Emergency Vehicle Delay',                 'EV delay (s)'),
    'group1':  ('Group 1 Delay (vehicles ahead of an EV)',  'Delay (s)'),
    'group2':  ('Group 2 Delay (other regular vehicles)',   'Delay (s)'),
    'all_reg': ('All Regular Vehicles - Delay',              'Delay (s)'),
}

BASELINE_COLOR = '#7f7f7f'   # neutral gray, dashed - stays consistent across
VARIANT_COLOR  = '#1f77b4'   # every comparison you make (baseline never changes look)


# ============================================================================
# Data loading
# ============================================================================

def load_run(label, path):
    """Load one saved evaluation JSON and pull out its pooled raw_delays."""
    with open(path, 'r') as f:
        data = json.load(f)
    if 'raw_delays' not in data:
        raise KeyError(
            f"'{path}' has no 'raw_delays' key - it was saved with an older "
            f"version of var_test_mappo_ambulance.py. Re-run evaluation with "
            f"the updated script (which now saves raw per-vehicle delays) "
            f"before plotting."
        )
    return {'label': label, 'raw_delays': data['raw_delays'], 'path': path}


# ============================================================================
# Plotting
# ============================================================================

def plot_group(ax, baseline, variant, group_key,
                max_bins=30, min_bins=8, show_mean_lines=True,
                crop_percentile=99):
    """Draw overlaid baseline vs. variant histogram + KDE curves for one
    delay group onto ax."""
    title, xlabel = GROUPS[group_key]

    base_vals = baseline['raw_delays'].get(group_key, [])
    var_vals  = variant['raw_delays'].get(group_key, [])
    all_vals  = base_vals + var_vals
    if not all_vals:
        ax.set_title(f"{title}\n(no data)")
        return
    pooled = np.array(all_vals)
    x_min, x_max = pooled.min(), pooled.max()

    # Figure out the visible window FIRST (crop_percentile of the full
    # pooled data), then build the bin edges / KDE grid only across that
    # window. Building them from the full x_min/x_max (including whatever
    # gets cropped out below) was the bug: bins spread across the true full
    # range, so only a handful ever landed inside the visible, cropped area
    # - looked like a fixed small bin count no matter what max_bins was set to.
    display_max = float(np.percentile(pooled, crop_percentile))
    cropped = display_max < x_max
    n_hidden = int((pooled > display_max).sum()) if cropped else 0
    view_max = display_max if cropped else x_max

    pad = 0.05 * (view_max - x_min) if view_max > x_min else 1.0
    x_grid = np.linspace(max(0.0, x_min - pad), view_max + pad, 400)

    # NEW: bin count scales with sample size (sqrt rule) instead of always
    # using max_bins. Fixes small-n groups like EV (~30-70 pooled points)
    # getting spread so thin across max_bins that the histogram looks nearly
    # empty/flat even though the data is there.
    n_bins = int(np.clip(np.sqrt(len(pooled)), min_bins, max_bins))

    for (label, vals, color, ls) in (
        (baseline['label'], base_vals, BASELINE_COLOR, '--'),
        (variant['label'],  var_vals,  VARIANT_COLOR,  '-'),
    ):
        n = len(vals)
        if n == 0:
            continue
        vals_arr = np.array(vals)
        mean_val = float(vals_arr.mean())

        ax.hist(vals, bins=n_bins, range=(x_grid[0], x_grid[-1]), density=True,
                 alpha=0.3, color=color, edgecolor='none')
        if n > 1 and np.ptp(vals) > 0:
            # KDE is still fit on the FULL vals (including any points beyond
            # the crop) so its shape reflects the true distribution - only
            # the drawing/view window is limited to x_grid.
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=color, linestyle=ls,
                     linewidth=2, label=f"{label} (n={n}, mean={mean_val:.1f}s)")
        else:
            ax.axvline(vals[0], color=color, linestyle=ls, linewidth=2,
                        label=f"{label} (n={n})")

        # NEW: dotted vertical line marking this run's mean, color-matched,
        # so the mean is visible on the graph itself, not just in the legend
        if show_mean_lines:
            ax.axvline(mean_val, color=color, linestyle=':', linewidth=1.3, alpha=0.9)

    # NEW: zoom the visible x-range to crop_percentile of the pooled data.
    # A couple of extreme-delay vehicles (e.g. one stuck for most of the
    # episode) otherwise stretch the axis out so far that the real
    # baseline-vs-variant differences near the bulk of the data become
    # invisible. The full data still goes into the histogram/KDE above -
    # this only changes what's shown, and the cropped point count is noted
    # in the title rather than silently hidden.
    ax.set_xlim(x_grid[0], x_grid[-1])
    title_note = f"\n({n_hidden} pt(s) beyond axis, max {x_max:.0f}s)" if n_hidden > 0 else ""

    ax.set_title(title + title_note, fontsize=11 if title_note else 12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def make_plots(baseline, variant, out_dir, max_bins=30, min_bins=8,
               show_mean_lines=True, crop_percentile=99):
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for group_key in GROUPS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_group(ax, baseline, variant, group_key, max_bins=max_bins,
                   min_bins=min_bins, show_mean_lines=show_mean_lines,
                   crop_percentile=crop_percentile)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'{group_key}_delay_distribution.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved: {out_path}")

    # NEW: all 4 groups on one combined 2x2 figure - same 4 plots above, just
    # also assembled into a single image for a report overview/summary slide.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, group_key in zip(axes.flat, GROUPS):
        plot_group(ax, baseline, variant, group_key, max_bins=max_bins,
                   min_bins=min_bins, show_mean_lines=show_mean_lines,
                   crop_percentile=crop_percentile)
    fig.suptitle(f"{baseline['label']} vs. {variant['label']} - Vehicle Delay Distributions",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(out_dir, 'all_groups_delay_distribution.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(combined_path)
    print(f"Saved: {combined_path}")

    return saved


# ============================================================================
# Run (Spyder: just hit Run - no CLI args needed)
# ============================================================================

def main():
    baseline = load_run(BASELINE_LABEL, BASELINE_PATH)
    variant  = load_run(VARIANT_LABEL, VARIANT_PATH)
    make_plots(baseline, variant, OUT_DIR, max_bins=MAX_BINS, min_bins=MIN_BINS,
               show_mean_lines=SHOW_MEAN_LINES, crop_percentile=CROP_PERCENTILE)


if __name__ == '__main__':
    main()