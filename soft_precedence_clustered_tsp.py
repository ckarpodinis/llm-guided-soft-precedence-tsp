import gurobipy as gp
from gurobipy import GRB
import json
import math
import metrics
import helpers

def short(x):
    if isinstance(x, str):
        return x[:6]
    return str(x)

def debug_soft_constraints(u, s_cluster, s_raw, pos0):
    print("\n[DEBUG] Soft constraint violations summary\n")

    def print_block(title, slack_dict):
        print(f"\n--- {title} ---")

        any_printed = False
        for (i, j), var in slack_dict.items():
            start = var.start if var.start is not None else 0.0
            final = var.X

            if start > 1e-6 or final > 1e-6:
                any_printed = True
                print(
                    f"  {short(i)} → {short(j)} | "
                    f"start={start:.2f} → final={final:.2f}"
                )

        if not any_printed:
            print("  ✔ none violated")

    print_block("CLUSTER ORDER constraints", s_cluster)
    print_block("FREE / LOGICAL constraints", s_raw)

def compare_slack_start_vs_final(s_cluster=None, s_raw=None):
    def check(title, sdict):
        if not sdict:
            return
        print(f"\n-- {title} --")
        for (i, j), var in sdict.items():
            start = var.Start
            final = var.X
            if abs(final - start) > 1e-6:
                print(
                    f"  {short(i)} → {short(j)} : "
                    f"start={start:.2f} → final={final:.2f}"
                )

    check("CLUSTER slacks", s_cluster)
    check("RAW slacks", s_raw)

def debug_objective_components(m, travel_cost, position_penalty, edge_penalty,
                               cluster_penalty, raw_penalty):
    print("\n[DEBUG] Objective decomposition")

    print(f"  Travel cost       : {travel_cost.getValue():.2f}")
    print(f"  Position penalty  : {position_penalty.getValue():.2f}")
    print(f"  Edge penalty      : {edge_penalty.getValue():.2f}")
    print(f"  Cluster penalty   : {cluster_penalty.getValue():.2f}")
    print(f"  Raw penaltly      : {raw_penalty.getValue():.2f}")
    print(f"  TOTAL             : {m.ObjVal:.2f}")

def local_relative_displacement(i, pos0, u, k=2):
    """
    Counts how many local neighbor orderings of i were inverted.
    k = window size on each side
    """
    p0 = pos0[i]
    ui = u[i].X

    # initial neighbors
    neighbors = [
        j for j in pos0
        if j != i and abs(pos0[j] - p0) <= k
    ]

    inv = 0
    for j in neighbors:
        if (p0 - pos0[j]) * (ui - u[j].X) < 0:
            inv += 1

    return inv

def ascii_shift_plot(u, pos0, clusters=None, width=80, k=2):
    print("\n[DEBUG] Warm start → final ordering\n")
    
    # map node -> cluster id (optional)
    cluster_of = {}
    if clusters:
        for k, C in clusters.items():
            for i in C:
                cluster_of[i] = k

    N = len(pos0)
    scale = width / N

    # ranks
    final_rank = {
        i: rank for rank, i in enumerate(
            sorted(pos0, key=lambda x: u[x].X)
        )
    }
    initial_rank = {
        i: rank for rank, i in enumerate(
            sorted(pos0, key=lambda x: pos0[x])
        )
    }

    for i in sorted(pos0, key=lambda x: pos0[x]):
        p0 = pos0[i]
        p1 = u[i].X

        Δ = int(p1 - p0)
        Δrel = final_rank[i] - initial_rank[i]
        Δloc = local_relative_displacement(i, pos0, u, k=k)

        x0 = int(p0 * scale)
        x1 = int(p1 * scale)

        line = [" "] * (width + 1)
        line[x0] = "|"
        line[x1] = "*"

        for t in range(min(x0, x1) + 1, max(x0, x1)):
            line[t] = "-"

        ctag = f"[C{cluster_of[i]}]" if cluster_of and i in cluster_of else ""

        print(
            f"{short(i):6} {ctag:4}  "
            f"{''.join(line)}  "
            #f"Δ={Δ:+3d}  Δrel={Δrel:+3d}  Δloc={Δloc}"
            f"Δ={Δ:+3d}  Δloc={Δloc}"
        )


def print_route_table(route, unified_steps_path="unified_steps.json"):
    import pandas as pd

    with open(unified_steps_path, "r", encoding="utf-8") as f:
        steps_data = json.load(f)

    uuid_to_step = {entry["uuid"]: entry["step"] for entry in steps_data}

    def safe_name(x):
        if x is None:
            return "(none)"
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("name") or x.get("id") or "(unnamed)"
        return str(x)

    def describe(step):
        if not step:
            return "(missing step)"

        a = step.get("action")
        if a == "set":
            return f"Set {safe_name(step.get('target'))}.{step.get('property')} = {step.get('value')}"
        if a == "open":
            return f"Open {safe_name(step.get('target'))}"
        if a == "close":
            return f"Close {safe_name(step.get('target'))}"
        if a == "transfer":
            return f"Transfer {safe_name(step.get('target'))} {safe_name(step.get('from'))} → {safe_name(step.get('to'))}"
        return json.dumps(step, ensure_ascii=False)

    rows = []
    for i, uid in enumerate(route):
        rows.append((i, short(uid), describe(uuid_to_step.get(uid))))

    # console formatting
    U_W, UUID_W = 3, 8
    print(f"{'u':<{U_W}}  {'uuid':<{UUID_W}}  description")
    print("-" * (U_W + UUID_W + 60))
    for u, uid, desc in rows:
        print(f"{u:<{U_W}}  {uid:<{UUID_W}}  {desc}")

    return pd.DataFrame([{"u": u, "uuid": uid, "description": d} for u, uid, d in rows])

def evaluate_prediction(pred_path, base_path):
    pred = helpers.load_uuid_sequence(pred_path)
    base = helpers.load_uuid_sequence(base_path)

    kt, sp = metrics.rank_correlations(pred, base)
    ed, ops = metrics.edit_distance(pred, base)
    lcs = metrics.lcs_length(pred, base)

    print("\n=== GLOBAL METRICS ===")
    print(f"Kendall τ   : {kt:.3f}")
    print(f"Spearman ρ : {sp:.3f}")
    print(f"Edit dist  : {ed:.3f}")
    print(f"LCS length : {lcs}/{len(base)}")

    metrics.print_comparison(pred, base)

    return {
        "kendall_tau": kt,
        "spearman": sp,
        "edit_distance": ed,
        "lcs": lcs,
        "ops": ops
    }

# -------------------------------------------------
# DATA
# -------------------------------------------------
with open("unified_steps.json", "r", encoding="utf-8") as f:
    steps = json.load(f)

# (optional) precedence constraints file
try:
    with open("uuid_constraints.json", "r", encoding="utf-8") as f:
        raw_constraints = json.load(f)
except FileNotFoundError:
    raw_constraints = []
    
# Create UUID nodes
nodes = [entry["uuid"] for entry in steps]
depot = "DEPOT"

V = [depot] + nodes
N = nodes[:]  # tasks only (exclude depot)

# --- Your clusters (UNORDERED inside because they are sets) ---
C1 = {
    '91659360-dc37-ba91-1e1f-cecb57419fdf', '0cde32bc-95c2-9dd3-d5d5-f2b78aa2dd0c',
    'b24eae4f-6f3a-dbfe-c4db-b59d5f286485', '99ef758e-8f6f-a9c9-1e94-b4782910d0f8',
    'fb2fd835-9da6-6480-1a97-64bc66159535', '7c02710e-df26-39d6-1d81-cd11702ca1d8'
}
C2 = {
    '3c489084-c9be-4168-084a-6f402713fb71', '531a1fd1-cc33-1b06-b3cb-e63e714e4574',
    '664a99ff-45a1-9a9c-ff3b-c7f0c13030ce', 'b0352a80-f285-0aba-ee44-28d6c310c07e',
    'd97be2dc-a572-0268-39d8-93c54e63ce19', '127cffe5-4c5a-3ffd-ca92-3eb03c902a5d'
}
C3 = {'cf4a006c-3ffd-a552-f62e-dafbd48d296b', '95b51243-9aa9-1fbc-ef22-813467386422', 
      '47b449ab-b02a-81f0-217e-a3054497199c'}

# Cluster order: C3 -> C1 -> C2
cluster_order = {1: C3, 2: C1, 3: C2}
ordered_cluster_keys = list(cluster_order.keys())  # keep insertion order

# Map node -> cluster index (only for nodes that belong to a cluster)
node_cluster = {}
for k, Ck in cluster_order.items():
    for i in Ck:
        node_cluster[i] = k

# Precedence list from uuid_constraints.json (optional)
hard_precedence = []
for c in raw_constraints:
    ordered_nodes = [c[k] for k in sorted(c, key=lambda x: int(x))]
    hard_precedence.extend(zip(ordered_nodes[:-1], ordered_nodes[1:]))

# Initial tour (UUID-based)
initial_tour = [depot] + nodes + [depot]
pos0 = {node: idx for idx, node in enumerate(initial_tour) if node in N}
x0 = {(i, j): 1 for (i, j) in zip(initial_tour, initial_tour[1:])}

# -------------------------------------------------
# COST MATRIX (uniform for now)
# -------------------------------------------------
c = {(i,j): 1 for i in V for j in V if i != j}

# -------------------------------------------------
# MODEL
# -------------------------------------------------
m = gp.Model("clustered_tsp_uuid_soft_clusters")

# x[i,j] = 1 if arc i -> j is used
x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
     for i in V for j in V if i != j}

# Position variables (u can be CONTINUOUS -> much faster)
u = {i: m.addVar(lb=1, ub=len(N), vtype=GRB.CONTINUOUS, name=f"u_{i}") for i in N}

# Deviation vars to keep solution close to initial order
d = {i: m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in N}

m.update()

# -------------------------------------------------
# CONSTRAINTS
# -------------------------------------------------
# Degree constraints
for j in V:
    m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, name=f"in_{j}")
for i in V:
    m.addConstr(gp.quicksum(x[i, j] for j in V if i != j) == 1, name=f"out_{i}")

# MTZ subtour elimination (tasks only)
M = len(N)
for i in N:
    for j in N:
        if i == j:
            continue
        m.addConstr(u[i] - u[j] + M * x[i, j] <= M - 1, name=f"mtz_{i}_{j}")

# Optional hard precedence constraints (from uuid_constraints.json)
s_raw = {}
for (i, j) in hard_precedence:
    if i in u and j in u:
        s_raw[i, j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                name=f"s_raw_{i}_{j}")
        m.addConstr(u[i] + 1 <= u[j] + s_raw[i, j], 
                    name=f"raw_order_{i}_{j}")

# Position deviation linearization: d[i] >= abs(u[i] - pos0[i])
for i in N:
    m.addConstr(d[i] >= u[i] - pos0[i], name=f"dev_pos_{i}_plus")
    m.addConstr(d[i] >= pos0[i] - u[i], name=f"dev_pos_{i}_minus")

# -------------------------------------------------
# SOFT CLUSTER ORDER
# -------------------------------------------------
# One slack per consecutive cluster-pair (k1 -> k2), but we still enforce it for all (i,j) pairs.
# This avoids dozens of slack variables while keeping the semantics: violations are allowed but penalized.
s_cluster = {}           # s_cluster[k1,k2] >= 0
pair_constraints = []  # keep for debugging / warm-start slack init

for k1, k2 in zip(ordered_cluster_keys[:-1], ordered_cluster_keys[1:]):
    C_prev = cluster_order[k1]
    C_next = cluster_order[k2]

    s_cluster[k1, k2] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_cluster_{k1}_{k2}")

    for i in C_prev:
        for j in C_next:
            if i in u and j in u:
                # u[i] + 1 <= u[j] + s_cluster[k1,k2]
                constr = m.addConstr(u[i] + 1 <= u[j] + s_cluster[k1, k2],
                                     name=f"cluster_{k1}_{k2}_{i}_{j}")
                pair_constraints.append((k1, k2, i, j, constr))

m.update()

# -------------------------------------------------
# OBJECTIVE
# -------------------------------------------------
pos_lambda = 2         # weight for staying close to initial order (retain position)
edge_lambda = 1        # weight for staying close to initial order (retain edges)
raw_lambda = 10.0      # weight for cluster order
cluster_lambda = 10.0  # weight for raw order

travel_cost = gp.quicksum(c[i, j] * x[i, j] for (i, j) in x)
position_penalty = pos_lambda * gp.quicksum(d[i] for i in N)
edge_penalty = edge_lambda * gp.quicksum(1 - x[i,j] for (i, j) in x0)
cluster_penalty = cluster_lambda * gp.quicksum(s_cluster.values())
raw_penalty = raw_lambda * gp.quicksum(s_raw.values())

m.setObjective(travel_cost + position_penalty + edge_penalty + cluster_penalty + raw_penalty, GRB.MINIMIZE)

# -------------------------------------------------
# WARM START
# -------------------------------------------------
# x starts (only edges that exist; here all exist, but keep robust)
for (i, j) in x:
    x[i, j].start = 0

for (i, j) in zip(initial_tour, initial_tour[1:]):
    if (i, j) in x:
        x[i, j].start = 1

# u/d starts (consistent with initial_tour)
for i in N:
    #u[i].start = float(pos0[i])
    d[i].start = 0.0

# slack starts: compute maximum violation in the warm start for each cluster pair
# s_cluster[k1,k2] >= max_{i in Cprev, j in Cnext} (pos0[i] + 1 - pos0[j], 0)
for k1, k2 in s_cluster:
    worst = 0.0
    for i in cluster_order[k1]:
        for j in cluster_order[k2]:
            if i in pos0 and j in pos0:
                worst = max(worst, pos0[i] + 1 - pos0[j])
    s_cluster[k1, k2].start = max(0.0, float(worst))

for i, j in s_raw:
    worst = 0.0
    if i in pos0 and j in pos0:
        worst = max(worst, pos0[i] + 1 - pos0[j])
    s_raw[i, j].start = max(0.0, float(worst))

# -------------------------------------------------
# SOLVE
# -------------------------------------------------
m.setParam("TimeLimit", 10)
#m.setParam("MIPGap", 0.02)
#m.setParam("MIPFocus", 1)   # focus on finding incumbents
#m.setParam("Heuristics", 0.2)
#m.setParam("Cuts", 2)
#m.setParam("Presolve", 2)
m.setParam("MIPFocus", 0)      # balanced
m.setParam("Heuristics", 0.5)  # more aggressive incumbents

# --- DEBUG: evaluate warm-start cluster violations (based on u.start = pos0) ---
print("\n[Warm start check] cluster violations implied by pos0 order:")

for k1, k2 in s_cluster:
    worst = 0
    worst_pair = None
    for i in cluster_order[k1]:
        for j in cluster_order[k2]:
            if i in pos0 and j in pos0:
                viol = pos0[i] + 1 - pos0[j]
                if viol > worst:
                    worst = viol
                    worst_pair = (i, j, pos0[i], pos0[j])
    if worst > 0:
        i, j, pi, pj = worst_pair
        print(f"  pair {k1}->{k2}: worst viol = {worst}  (i={i}@{pi}  j={j}@{pj})")
    else:
        print(f"  pair {k1}->{k2}: no violation")

print("\n[Warm start] slack starts:")
for (k1, k2), var in s_cluster.items():
    print(f"  s_cluster[{k1}->{k2}].start = {var.start}")


m.optimize()

# -------------------------------------------------
# DEBUG & ANALYSIS
# -------------------------------------------------

# 1. Objective breakdown (who pays and what)
debug_objective_components(
    m,
    travel_cost,
    position_penalty,
    edge_penalty,
    cluster_penalty,
    raw_penalty
)

# 2. Soft constraint violations (cluster + raw)
debug_soft_constraints(
    u,
    s_cluster=s_cluster,
    s_raw=s_raw,
    pos0=pos0
)

# 3. Slack repair analysis (warm start → final)
compare_slack_start_vs_final(
    s_cluster=s_cluster,
    s_raw=s_raw
)

# 4. Positional shift visualization (ASCII)
ascii_shift_plot(
    u=u,
    pos0=pos0,
    width=80,
    clusters=cluster_order
)

# -------------------------------------------------
# REPORT
# -------------------------------------------------
print("\nSlack variables (cluster order violations):")
for (k1, k2), var in s_cluster.items():
    if var.X > 1e-6:
        print(f"s_cluster[{k1}->{k2}] = {var.X:.3f}")

print("\nOptimal tour edges:")
for (i, j), var in x.items():
    if var.X > 0.5:
        print(f"{short(i)} -> {short(j)}")

print("\nPositions (u):")
for i in N:
    print(short(i), u[i].X)

# -------------------------------------------------
# BUILD ROUTE FROM SOLUTION
# -------------------------------------------------
successor = {}
for (i, j), var in x.items():
    if var.X > 0.5:
        successor[i] = j

route = [depot]
cur = depot
while True:
    nxt = successor[cur]
    if nxt == depot:
        break
    route.append(nxt)
    cur = nxt

print_route_table(route, unified_steps_path="unified_steps.json")

# -------------------------------------------------
# WRITE THE SOLUTION
# -------------------------------------------------
helpers.write_solver_output(
    route,
    unified_steps_path="unified_steps.json",
    output_path="solver_output.json",
    depot=depot
)

# -------------------------------------------------
# EVALUATE THE SOLUTION
# -------------------------------------------------
evaluate_prediction(
    pred_path="solver_output.json",
    base_path="sample_steps_gt.json"
)

evaluate_prediction(
    pred_path="solver_output.json",
    base_path="unified_steps.json"
)
