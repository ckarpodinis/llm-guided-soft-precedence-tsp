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
    feasible = (df["slacks.cluster.sum"] <= 1e-9) & (df["slacks.raw.sum"] <= 1e-9)
    print(f"Excluded {len(df)-len(df[feasible])} infeasible experiment")
    df = df[feasible]


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
# FIGURE C — Permutation
# --------------------------------------------------
plots.plot_permutation(df, ref="ground_truth")
plots.plot_permutation(df, ref="initial")

# --------------------------------------------------
# FIGURE D — Displacement
# --------------------------------------------------
plots.plot_displacement(df, ref="ground_truth")
plots.plot_displacement(df, ref="initial")
#plots.plot_displacement(df)

# ======================================================
# FIGURE 2a — Global vs local (vs GT)
# ======================================================
fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    
plots.plot_global_vs_local(df, axes2.flat[0], ref="ground_truth")
plots.plot_global_vs_local(df, axes2.flat[1], ref="initial")
plots.plot_ordering_tradeoff(df, axes2.flat[2], "displacement.mean_abs_delta")
plots.plot_ordering_tradeoff(df, axes2.flat[3], "displacement.mean_delta_loc")
plots.plot_ordering_tradeoff(df, axes2.flat[4], "displacement.local_stability")

fig2.suptitle("Figure E — Global vs local displacement", fontsize=14)

# ======================================================
# FIGURE 3 — Ordering trade-off (sensitivity)
# ======================================================
fig3, axes3 = plt.subplots(
    3, 2, figsize=(14, 14), constrained_layout=True
)

metrics = [
    ("kendall_tau", "Kendall τ"),
    ("normalized_spearmanf", "Normalized Spearman F"),
    ("bigram", "Bigram overlap"),
    ("trigram", "Trigram overlap"),
    ("breakrate", "Break rate"),
    ("lcs", "LCS Lenght"),
]

for ax, (key, label) in zip(axes3.flat, metrics):
    sc = plots.plot_ordering_tradeoff(df, ax, key)

fig3.suptitle(
    "Figure F — Ordering trade-off: improvement vs drift",
    fontsize=14
)


# ======================================================
# Show all figures
# ======================================================
plt.show()
