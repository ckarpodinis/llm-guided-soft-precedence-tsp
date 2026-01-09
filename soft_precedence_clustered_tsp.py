import gurobipy as gp
from gurobipy import GRB
import json
import math


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
        rows.append((i, uid, describe(uuid_to_step.get(uid))))

    # console formatting
    U_W, UUID_W = 3, 36
    print(f"{'u':<{U_W}}  {'uuid':<{UUID_W}}  description")
    print("-" * (U_W + UUID_W + 60))
    for u, uid, desc in rows:
        print(f"{u:<{U_W}}  {uid:<{UUID_W}}  {desc}")

    return pd.DataFrame([{"u": u, "uuid": uid, "description": d} for u, uid, d in rows])


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
C3 = {'cf4a006c-3ffd-a552-f62e-dafbd48d296b', '95b51243-9aa9-1fbc-ef22-813467386422', '47b449ab-b02a-81f0-217e-a3054497199c'}

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
for (a, b) in hard_precedence:
    if a in u and b in u:
        m.addConstr(u[a] + 1 <= u[b], name=f"prec_{a}_{b}")

# Position deviation linearization
for i in N:
    m.addConstr(d[i] >= u[i] - pos0[i], name=f"dev_pos_{i}_plus")
    m.addConstr(d[i] >= pos0[i] - u[i], name=f"dev_pos_{i}_minus")

# -------------------------------------------------
# SOFT CLUSTER ORDER
# -------------------------------------------------
# One slack per consecutive cluster-pair (k1 -> k2), but we still enforce it for all (i,j) pairs.
# This avoids dozens of slack variables while keeping the semantics: violations are allowed but penalized.
s = {}           # s[k1,k2] >= 0
pair_constraints = []  # keep for debugging / warm-start slack init

for k1, k2 in zip(ordered_cluster_keys[:-1], ordered_cluster_keys[1:]):
    C_prev = cluster_order[k1]
    C_next = cluster_order[k2]

    s[k1, k2] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s_cluster_{k1}_{k2}")

    for i in C_prev:
        for j in C_next:
            if i in u and j in u:
                # u[i] + 1 <= u[j] + s[k1,k2]
                constr = m.addConstr(u[i] + 1 <= u[j] + s[k1, k2],
                                     name=f"cluster_{k1}_{k2}_{i}_{j}")
                pair_constraints.append((k1, k2, i, j, constr))

m.update()

# -------------------------------------------------
# OBJECTIVE
# -------------------------------------------------
lam = 0.1                 # weight for staying close to initial order
CLUSTER_PENALTY = 50.0     # make this bigger if you want cluster order to dominate

travel_cost = gp.quicksum(c[i, j] * x[i, j] for (i, j) in x)
soft_penalty = lam * gp.quicksum(d[i] for i in N)
cluster_penalty = CLUSTER_PENALTY * gp.quicksum(s.values())

m.setObjective(travel_cost + soft_penalty + cluster_penalty, GRB.MINIMIZE)

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
# s[k1,k2] >= max_{i in Cprev, j in Cnext} (pos0[i] + 1 - pos0[j], 0)
for k1, k2 in s:
    worst = 0.0
    for i in cluster_order[k1]:
        for j in cluster_order[k2]:
            if i in pos0 and j in pos0:
                worst = max(worst, pos0[i] + 1 - pos0[j])
    s[k1, k2].start = max(0.0, float(worst))

# -------------------------------------------------
# SOLVE
# -------------------------------------------------
m.setParam("TimeLimit", 30)
#m.setParam("MIPGap", 0.02)
#m.setParam("MIPFocus", 1)   # focus on finding incumbents
#m.setParam("Heuristics", 0.2)
#m.setParam("Cuts", 2)
#m.setParam("Presolve", 2)
m.setParam("MIPFocus", 0)      # balanced
m.setParam("Heuristics", 0.5)  # more aggressive incumbents

# --- DEBUG: evaluate warm-start cluster violations (based on u.start = pos0) ---
print("\n[Warm start check] cluster violations implied by pos0 order:")

for k1, k2 in s:
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
for (k1, k2), var in s.items():
    print(f"  s_cluster[{k1}->{k2}].start = {var.start}")


m.optimize()

# -------------------------------------------------
# REPORT
# -------------------------------------------------
print("\nObjective value:", m.ObjVal)
print("Travel cost:", travel_cost.getValue())
print("Soft penalty:", soft_penalty.getValue())
print("Cluster penalty:", cluster_penalty.getValue())

print("\nSlack variables (cluster order violations):")
for (k1, k2), var in s.items():
    if var.X > 1e-6:
        print(f"s_cluster[{k1}->{k2}] = {var.X:.3f}")

print("\nOptimal tour edges:")
for (i, j), var in x.items():
    if var.X > 0.5:
        print(f"{i} -> {j}")

print("\nPositions (u):")
for i in N:
    print(i, u[i].X)

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

