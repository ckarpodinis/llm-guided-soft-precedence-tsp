import argparse
import json
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

# -----------------------------
# Extract rules with probabilities
# -----------------------------
def extract_rules(tree, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value

    rules = []

    def recurse(node, conditions):
        # leaf
        if feature[node] == _tree.TREE_UNDEFINED:
            n_samples = int(tree_.n_node_samples[node])
            counts = value[node][0]
            p_good = counts[1] / counts.sum() if counts.sum() > 0 else 0

            rules.append({
                "rule": " AND ".join(conditions) if conditions else "ALL",
                "P(good)": p_good,
                "n_samples": n_samples
            })
            return

        fname = feature_names[feature[node]]
        thresh = threshold[node]

        recurse(
            tree_.children_left[node],
            conditions + [f"{fname} <= {thresh:.2f}"]
        )
        recurse(
            tree_.children_right[node],
            conditions + [f"{fname} > {thresh:.2f}"]
        )

    recurse(0, [])
    return rules

# -----------------------------
# simplify rule conditions 
# (e.g. l4 > 1.5 AND l1 <= 1.5 AND l4 > 3.5 to l4 > 3.5 AND l1 <= 1.5)
# -----------------------------
def simplify_rule(rule: str):
    """
    Simplify a rule by collapsing multiple constraints
    on the same variable.

    Input:
        "l4 > 1.5 AND l1 <= 1.5 AND l4 > 3.5"

    Output:
        "l4 > 3.5 AND l1 <= 1.5"
    """
    if rule == "ALL":
        return rule

    # store bounds: var -> (lower_bound, upper_bound)
    bounds = defaultdict(lambda: [-float("inf"), float("inf")])

    for part in rule.split(" AND "):
        m = re.match(r"(\w+)\s*(<=|>)\s*([\d\.]+)", part)
        if not m:
            continue

        var, op, val = m.group(1), m.group(2), float(m.group(3))

        if op == ">":
            bounds[var][0] = max(bounds[var][0], val)
        elif op == "<=":
            bounds[var][1] = min(bounds[var][1], val)

    simplified = []
    for var, (lo, hi) in bounds.items():
        if lo > -float("inf"):
            simplified.append(f"{var} > {lo:.2f}")
        if hi < float("inf"):
            simplified.append(f"{var} <= {hi:.2f}")

    return " AND ".join(simplified)


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

# -----------------------------
# Define lambdas
# -----------------------------
df["pos_lambda"] = df["params.pos_lambda"]
df["edge_lambda"] = df["params.edge_lambda"]
df["cluster_lambda"] = df["params.cluster_lambda"]
df["raw_lambda"] = df["params.raw_lambda"]

# -----------------------------
# Define KPIs
# -----------------------------
df["kpi_cluster_slack"] = df["slacks.cluster.sum"]
df["kpi_raw_slack"]     = df["slacks.raw.sum"]
df["kendall_gt"]        = df["metrics.vs_ground_truth.kendall_tau"]
df["kendall_init"]      = df["metrics.vs_initial.kendall_tau"]
df["bigram_gt"]         = df["metrics.vs_ground_truth.bigram"]
df["bigram_init"]       = df["metrics.vs_initial.bigram"]

# -----------------------------
# Define feasibility and goodness
# -----------------------------
df["feasible"] = (
    (df["kpi_cluster_slack"] == 0) &
    (df["kpi_raw_slack"] == 0)
)

#df["good"] = (
#    df["feasible"] &
#    (df["kendall_gt"] > df["kendall_init"])
#)

df["good"] = (
    df["feasible"] &
    (df["bigram_gt"] > df["bigram_init"])
)

# optional: only meaningful when feasible
df.loc[~df["feasible"], "delta_kendall"] = float("nan")

# -----------------------------
# Decision tree
# -----------------------------
X = df[["pos_lambda", "edge_lambda", "cluster_lambda", "raw_lambda"]]
y = df["good"].astype(int)

tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf = max(10, int(0.03 * len(df))),
    random_state=0
)
tree.fit(X, y)

print(export_text(tree, feature_names=X.columns.tolist()))

rules = extract_rules(tree, X.columns.tolist())

for r in rules:
    #if r["n_samples"] >= 10:  # optional filter
        rule_clean = simplify_rule(r["rule"])
        print(f"{rule_clean}  →  P(good) = {r['P(good)']:.2f},  n = {r['n_samples']}")

#base_rate = df["good"].mean()
#for r in rules:
#    if r["n_samples"] >= 20 and abs(r["P(good)"] - base_rate) > 0.1:
#        rule_clean = simplify_rule(r["rule"])
#        print(f"{rule_clean}  →  P(good) = {r['P(good)']:.2f},  n = {r['n_samples']}")

