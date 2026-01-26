import itertools
import json
from soft_precedence_clustered_tsp import solver
from logging_utils import experiment_logger
from pathlib import Path
from datetime import datetime

GRID = {
    #"pos_lambda":     [0, 1, 2, 5],
    #"edge_lambda":    [0, 1, 3],
    #"cluster_lambda": [0, 1, 3, 10, 30],
    #"raw_lambda":     [0, 1, 10],
    #"pos_lambda":     [0.1, 0.5, 1, 2, 5],
    #"edge_lambda":    [0.1, 0.5, 1, 2, 5],
    #"cluster_lambda": [0.1, 1, 5, 10, 20],
    #"raw_lambda":     [0.1, 1, 5, 10, 20],
    "pos_lambda":     [0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 3, 5, 10],
    "edge_lambda":    [1],
    "cluster_lambda": [10],
    "raw_lambda":     [10],
}

grid_results = []

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = Path(f"logs/{timestamp}")
log_dir.mkdir(exist_ok=True)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

grid_combinations = list(itertools.product(
    GRID["pos_lambda"],
    GRID["edge_lambda"],
    GRID["cluster_lambda"],
    GRID["raw_lambda"]
))

total_experiments = len(grid_combinations)

for idx, (pos, edge, cluster, raw) in enumerate(grid_combinations, start=1):
    
    tag = f"pos{pos}_edge{edge}_cl{cluster}_raw{raw}"
    log_path = log_dir / f"{tag}.log"

    # console progress indicator
    print(f"[{idx:3d}/{total_experiments}] Running {tag}")

    with experiment_logger(log_path, also_print=False):
        print("=" * 80)
        print(f"Experiment: {tag}")
        print("=" * 80)

        res = solver(
            pos_lambda=pos,
            edge_lambda=edge,
            cluster_lambda=cluster,
            raw_lambda=raw,
            time_limit=10,
            heuristics=0.5,
            mipfocus=0,
            tag=tag
        )

    grid_results.append(res)

# -------------------------------------------------
# SAVE GRID RESULTS
# -------------------------------------------------
out_path = results_dir / f"grid_results_{timestamp}.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(grid_results, f, indent=2)
