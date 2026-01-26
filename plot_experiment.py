import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import plots

# ======================================================
# CLI
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot results from a grid search JSON file"
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to grid_results_*.json"
    )
    parser.add_argument(
        "--feasible-only",
        action="store_true",
        help="Filters feasible-only solutions (default: False)"
    )
    return parser.parse_args()


# ======================================================
# Load results
# ======================================================
def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


args = parse_args()
data = load_results(args.results_json)

print(f"[OK] Loaded {len(data)} experiments from {args.results_json}")

df = pd.json_normalize(data)

if args.feasible_only is True:
    df = df[(df["slacks.cluster.sum"] <= 1e-9) & (df["slacks.raw.sum"] <= 1e-9)]


# ======================================================
# Lambda definitions (single source of truth)
# ======================================================
lambda_specs = [
    ("pos_lambda",     "λ_pos"),
    ("edge_lambda",    "λ_edge"),
    ("cluster_lambda", "λ_cluster"),
    ("raw_lambda",     "λ_raw"),
]

lambda_keys = [k for k, _ in lambda_specs]

# --------------------------------------------------
# FIGURE A
# --------------------------------------------------
plots.plot_solver_behavior(df)

# --------------------------------------------------
# FIGURE B — Ordering quality
# --------------------------------------------------
plots.plot_ordering_quality(df, ref="ground_truth")
plots.plot_ordering_quality(df, ref="initial")
#plots.plot_ordering_quality(df)

# --------------------------------------------------
# FIGURE C — Displacement
# --------------------------------------------------
plots.plot_displacement(df, ref="ground_truth")
plots.plot_displacement(df, ref="initial")
#plots.plot_displacement(df)

# ======================================================
# FIGURE 2a — Global vs local (vs GT)
# ======================================================
fig_gt, axes_gt = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

for ax, lam in zip(axes_gt.flat, lambda_keys):
    sc = plots.plot_global_vs_local_by_lambda(
        df, ax, lam_key=lam, ref="ground_truth"
    )
    plt.colorbar(sc, ax=ax, label=lam)

fig_gt.suptitle("Figure D — Global vs local displacement (vs ground truth)", fontsize=14)


# ======================================================
# FIGURE 2b — Global vs local (vs initial)
# ======================================================
fig_init, axes_init = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

for ax, lam in zip(axes_init.flat, lambda_keys):
    sc = plots.plot_global_vs_local_by_lambda(
        df, ax, lam_key=lam, ref="initial"
    )
    plt.colorbar(sc, ax=ax, label=lam)

fig_init.suptitle("Figure E — Global vs local displacement (vs initial)", fontsize=14)


# ======================================================
# FIGURE 3 — Ordering trade-off (sensitivity)
# ======================================================
fig3, axes3 = plt.subplots(
    2, 2, figsize=(14, 12), constrained_layout=True
)

for ax, lam in zip(axes3.flat, lambda_keys):
    sc = plots.plot_ordering_tradeoff_by_lambda(
        df, ax, lam_key=lam
    )

for ax in axes3.flat:
    sc = ax.collections[0]
    plt.colorbar(sc, ax=ax, label="λ value")

fig3.suptitle(
    "Figure F — Ordering trade-off: improvement vs drift",
    fontsize=14
)


# ======================================================
# Show all figures
# ======================================================
plt.show()

