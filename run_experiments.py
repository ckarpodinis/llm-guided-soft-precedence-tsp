import itertools
import json
import soft_precedence_clustered_tsp

#pos_lambdas = [0.5, 1, 2, 5]
#edge_lambdas = [0, 0.5, 1]
#cluster_lambdas = [1, 5, 10]
#raw_lambdas = [5, 10]
pos_lambdas = [2]
edge_lambdas = [1]
cluster_lambdas = [10]
raw_lambdas = [10]

experiments = itertools.product(
    pos_lambdas,
    edge_lambdas,
    cluster_lambdas,
    raw_lambdas
)

results = []

for idx, (pl, el, cl, rl) in enumerate(experiments):
    print(f"Running exp {idx}: "
          f"pos={pl}, edge={el}, cluster={cl}, raw={rl}")

    res = soft_precedence_clustered_tsp.solver(
        pos_lambda=pl,
        edge_lambda=el,
        cluster_lambda=cl,
        raw_lambda=rl,
        time_limit=10,
        heuristics=0.5,
        mipfocus=0,
        tag=f"exp_{idx}"
    )

    results.append(res)

with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

