import gurobipy as gp
from gurobipy import GRB
import uuid
import math
import json

def print_route_table(route, unified_steps_path="unified_steps.json"):
    import json
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
    for i, uuid in enumerate(route):
        rows.append((i, uuid, describe(uuid_to_step.get(uuid))))

    # --------- MANUAL CONSOLE FORMATTING ----------
    U_W, UUID_W = 3, 36

    print(f"{'u':<{U_W}}  {'uuid':<{UUID_W}}  description")
    print("-" * (U_W + UUID_W + 60))

    for u, uuid, desc in rows:
        print(f"{u:<{U_W}}  {uuid:<{UUID_W}}  {desc}")

    return pd.DataFrame(
        [{"u": u, "uuid": uuid, "description": d} for u, uuid, d in rows]
    )

# -------------------------------------------------
# DATA
# -------------------------------------------------

with open("unified_steps.json", "r", encoding="utf-8") as f:
    steps = json.load(f)

with open("uuid_constraints.json", "r", encoding="utf-8") as f:
    raw_constraints = json.load(f)

# Create UUID nodes
nodes = [entry["uuid"] for entry in steps]

depot = "DEPOT"

V = [depot] + nodes
N = nodes[:]   # tasks only (exclude depot)

# Clusters (by UUID)
#C1 = set([nodes[0], nodes[1]])
#C2 = set(nodes[3:5])
#C3 = set(nodes[5:7])
C1 = {'91659360-dc37-ba91-1e1f-cecb57419fdf', '0cde32bc-95c2-9dd3-d5d5-f2b78aa2dd0c', 'b24eae4f-6f3a-dbfe-c4db-b59d5f286485', '99ef758e-8f6f-a9c9-1e94-b4782910d0f8', 'fb2fd835-9da6-6480-1a97-64bc66159535', '7c02710e-df26-39d6-1d81-cd11702ca1d8'}
C2 = {'3c489084-c9be-4168-084a-6f402713fb71', '531a1fd1-cc33-1b06-b3cb-e63e714e4574', '664a99ff-45a1-9a9c-ff3b-c7f0c13030ce', 'b0352a80-f285-0aba-ee44-28d6c310c07e', 'd97be2dc-a572-0268-39d8-93c54e63ce19', '127cffe5-4c5a-3ffd-ca92-3eb03c902a5d'}
C3 = {'cf4a006c-3ffd-a552-f62e-dafbd48d296b', '95b51243-9aa9-1fbc-ef22-813467386422', '47b449ab-b02a-81f0-217e-a3054497199c'}

clusters = [C1, C2, C3]

# Cluster order: 1 < 2 < 3
cluster_order = {1: C3, 2: C1, 3: C2}

# Map node -> cluster index
node_cluster_order = {}
for k, Ck in cluster_order.items():
    for i in Ck:
        node_cluster_order[i] = k

# Hard precedence
#hard_precedence = [(c["0"], c["1"]) for c in raw_constraints]
hard_precedence = []

for c in raw_constraints:
    ordered_nodes = [c[k] for k in sorted(c, key=lambda x: int(x))]
    hard_precedence.extend(
        zip(ordered_nodes[:-1], ordered_nodes[1:])
    )

## Coordinates (UUID -> (x,y))
#coords = {
#    depot: (0, 0),
#}
#coords.update({
#    nodes[0]: (1,1),
#    nodes[1]: (2,1),
#    nodes[2]: (3,1),
#    nodes[3]: (4,2),
#    nodes[4]: (5,2),
#    nodes[5]: (6,3),
#    nodes[6]: (7,3),
#})
#
#def dist(i, j):
#    xi, yi = coords[i]
#    xj, yj = coords[j]
#    return math.hypot(xi - xj, yi - yj)
#
## Cost matrix
#c = {(i,j): dist(i,j) for i in V for j in V if i != j}

# Cost matrix
c = {(i,j): 1 for i in V for j in V if i != j}

# Initial tour (UUID-based)
initial_tour = [depot] + nodes + [depot]

# Initial positions (only for tasks)
pos0 = {node: idx for idx, node in enumerate(initial_tour) if node in N}

# -------------------------------------------------
# MODEL
# -------------------------------------------------

m = gp.Model("clustered_tsp_uuid")

# x[i,j] = 1 if arc i -> j is used
x = {}

for i in V:
    for j in V:
        if i == j:
            continue

        ## HARD CLUSTER ORDER (forbid backward arcs) -> remove this because destroys the warm start
        #if i in node_cluster_order and j in node_cluster_order:
        #    if node_cluster_order[i] > node_cluster_order[j]:
        #        continue
        
        x[i,j] = m.addVar(
            vtype=GRB.BINARY,
            name=f"x_{i}_{j}"
        )

# Position variables (tasks only)
u = {
    i: m.addVar(lb=1, ub=len(N), vtype=GRB.INTEGER, name=f"u_{i}")
    for i in N
}

# Position deviation variables
d = {
    i: m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"d_{i}")
    for i in N
}

m.update()

# -------------------------------------------------
# OBJECTIVE
# -------------------------------------------------

lam = 0.5
#lam = 0.2

travel_cost = gp.quicksum(c[i,j] * x[i,j] for (i,j) in x)
soft_penalty = lam * gp.quicksum(d[i] for i in N)

m.setObjective(travel_cost + soft_penalty, GRB.MINIMIZE)

# -------------------------------------------------
# CONSTRAINTS
# -------------------------------------------------

# Degree constraints
for j in V:
    m.addConstr(
        gp.quicksum(x[i,j] for i in V if (i,j) in x) == 1,
        name=f"in_{j}"
    )

for i in V:
    m.addConstr(
        gp.quicksum(x[i,j] for j in V if (i,j) in x) == 1,
        name=f"out_{i}"
    )

# MTZ subtour elimination (tasks only)
for i in N:
    for j in N:
        if i != j and (i,j) in x:
            m.addConstr(
                u[i] - u[j] + len(N) * x[i,j] <= len(N) - 1,
                name=f"mtz_{i}_{j}"
            )

# HARD CLUSTER ORDER (GLOBAL, via positions)
ordered_cluster_keys = sorted(cluster_order.keys())

for k1, k2 in zip(ordered_cluster_keys[:-1], ordered_cluster_keys[1:]):
    C_prev = cluster_order[k1]
    C_next = cluster_order[k2]

    for i in C_prev:
        for j in C_next:
            m.addConstr(
                u[i] + 1 <= u[j],
                name=f"cluster_order_{k1}_{k2}_{i}_{j}"
            )

# Hard precedence inside clusters
#eps = 1e-4
for (a,b) in hard_precedence:
    m.addConstr(
        u[a] + 1 <= u[b],
        name=f"prec_{a}_{b}"
    )

# Position deviation linearization
for i in N:
    m.addConstr(d[i] >= u[i] - pos0[i])
    m.addConstr(d[i] >= pos0[i] - u[i])

# -------------------------------------------------
# WARM START
# -------------------------------------------------

# Initialize all arcs to 0
for (i,j) in x:
    x[i,j].start = 0

# Activate arcs from initial tour
for (i,j) in zip(initial_tour, initial_tour[1:]):
    if (i,j) in x:
        x[i,j].start = 1

# Initial positions
for i in N:
    #u[i].start = pos0[i] #don't set u starts if start tour might still be inconsistent
    d[i].start = 0.0

# -------------------------------------------------
# SOLVE
# -------------------------------------------------
m.setParam("TimeLimit", 30)
#m.setParam("TimeLimit", 900)
m.setParam("MIPGap", 0.02)
# Debug: check which initial edges are missing in x
#missing = []
#for (i, j) in zip(initial_tour, initial_tour[1:]):
#    if (i, j) not in x:
#        missing.append((i, j))
#print("Missing edges from initial_tour (not created due to forbids):", len(missing))
#for e in missing[:10]:
#    print("  missing:", e)
m.optimize()

print("\nOptimal tour:")
for (i,j), var in x.items():
    if var.X > 0.5:
        print(f"{i} -> {j}")

print("\nPositions:")
for i in N:
    print(i, u[i].X)

print("\nObjective value:", m.ObjVal)
print("Travel cost:", travel_cost.getValue())
print("Soft penalty:", soft_penalty.getValue())

# -----------------------------
# BUILD ROUTE FROM SOLUTION
# -----------------------------

successor = {}
for (i, j), var in x.items():
    if var.X > 0.5:
        successor[i] = j

# Reconstruct ordered route starting from depot
route = [depot]
current = depot

while True:
    nxt = successor[current]
    if nxt == depot:
        break
    route.append(nxt)
    current = nxt

# Optional: include depot at end
# route.append(depot)

# Call your function
print_route_table(route, unified_steps_path="unified_steps.json")
