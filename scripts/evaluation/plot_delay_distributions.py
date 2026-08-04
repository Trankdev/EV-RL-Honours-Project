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
VARIANT_PATH   = "evaluations/z=40_best_core.json"     # TODO: path to this variant's --save-results JSON

OUT_DIR = "evaluations/plots/baseline_vs_z=40_best_core"      # TODO: change per ablation step, or PNGs overwrite
BINS = 30                                               # lower (e.g. 15) if EV histogram looks too jagged


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

def plot_group(ax, baseline, variant, group_key, bins=30):
    """Draw overlaid baseline vs. variant histogram + KDE curves for one
    delay group onto ax."""
    title, xlabel = GROUPS[group_key]

    base_vals = baseline['raw_delays'].get(group_key, [])
    var_vals  = variant['raw_delays'].get(group_key, [])
    all_vals  = base_vals + var_vals
    if not all_vals:
        ax.set_title(f"{title}\n(no data)")
        return

    x_min, x_max = min(all_vals), max(all_vals)
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_grid = np.linspace(max(0.0, x_min - pad), x_max + pad, 400)

    for (label, vals, color, ls) in (
        (baseline['label'], base_vals, BASELINE_COLOR, '--'),
        (variant['label'],  var_vals,  VARIANT_COLOR,  '-'),
    ):
        n = len(vals)
        if n == 0:
            continue
        ax.hist(vals, bins=bins, range=(x_grid[0], x_grid[-1]), density=True,
                 alpha=0.3, color=color, edgecolor='none')
        if n > 1 and np.ptp(vals) > 0:
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color=color, linestyle=ls,
                     linewidth=2, label=f"{label} (n={n})")
        else:
            ax.axvline(vals[0], color=color, linestyle=ls, linewidth=2,
                        label=f"{label} (n={n})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def make_plots(baseline, variant, out_dir, bins=30):
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for group_key in GROUPS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_group(ax, baseline, variant, group_key, bins=bins)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f'{group_key}_delay_distribution.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved: {out_path}")

    return saved


# ============================================================================
# Run (Spyder: just hit Run - no CLI args needed)
# ============================================================================

def main():
    baseline = load_run(BASELINE_LABEL, BASELINE_PATH)
    variant  = load_run(VARIANT_LABEL, VARIANT_PATH)
    make_plots(baseline, variant, OUT_DIR, bins=BINS)


if __name__ == '__main__':
    main()
