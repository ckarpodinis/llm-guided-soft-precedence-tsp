import gurobipy as gp
from gurobipy import GRB
import math

# -----------------------------
# DATA
# -----------------------------
V = list(range(8))              # 0..7 nodes
depot = 0
N = [i for i in V if i != depot]

# Clusters
C1 = {1,2,3}
C2 = {4,5}
C3 = {6,7}
clusters = [C1, C2, C3]

# cluster ordering: cluster k must come before cluster k+1
cluster_order = {1: C1, 2: C2, 3: C3}

# Create node -> cluster_order mapping
node_cluster_order = {}
for k, Ck in cluster_order.items():
    for i in Ck:
        node_cluster_order[i] = k

# Hard precedence constraints INSIDE clusters
# Example: in cluster C1: 1 must come before 3
#          in cluster C2: 4 must come before 5
hard_precedence = [(1,3), (4,5)]

# Coordinates (change as needed)
coords = {
    0:(0,0),
    1:(1,1), 2:(2,1), 3:(3,1),
    4:(4,2), 5:(5,2),
    6:(6,3), 7:(7,3)
}

def dist(i,j):
    (xi,yi) = coords[i]
    (xj,yj) = coords[j]
    return math.hypot(xi-xj, yi-yj)

# Cost matrix
c = {(i,j): dist(i,j) for i in V for j in V if i!=j}

# Initial solution (your given sequence)
initial_tour = [0,1,2,3,4,5,6,7,0]

# Position of nodes in the initial tour (for N only)
pos0 = {i: idx for idx, i in enumerate(initial_tour) if i in N}

# -----------------------------
# MODEL
# -----------------------------
m = gp.Model("clustered_tsp_softpos")

# x[i,j] = 1 if go from i to j
x = {}
for i in V:
    for j in V:
        if i == j:
            continue

        # Hard cluster order: forbid backwards transitions (only for non-depot nodes)
        if i in node_cluster_order and j in node_cluster_order:
            if node_cluster_order[i] > node_cluster_order[j]:
                continue  # do not even create this variable

        x[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# Position variables (1..|N|) for tasks only (not depot)
u = {i: m.addVar(lb=1, ub=len(N), vtype=GRB.CONTINUOUS, name=f"u_{i}") for i in N}

# Position deviation variables: d[i] >= |u[i] - pos0[i]| for tasks only
d = {i: m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"d_{i}") for i in N}

m.update()

# -----------------------------
# OBJECTIVE
# -----------------------------
lam = 0.5   # tune as needed

travel_cost = gp.quicksum(c[i,j] * x[i,j] for (i,j) in x)
soft_penalty = lam * gp.quicksum(d[i] for i in N)

m.setObjective(travel_cost + soft_penalty, GRB.MINIMIZE)

# -----------------------------
# CONSTRAINTS
# -----------------------------

# 1. Degree constraints: κάθε node (συμπεριλ. depot) 1 είσοδο, 1 έξοδο
for j in V:
    m.addConstr(gp.quicksum(x[i,j] for i in V if (i,j) in x) == 1,
                name=f"in_{j}")

for i in V:
    m.addConstr(gp.quicksum(x[i,j] for j in V if (i,j) in x) == 1,
                name=f"out_{i}")

# 2. MTZ subtour elimination (για tasks μόνο)
for i in N:
    for j in N:
        if i != j and (i,j) in x:
            m.addConstr(u[i] - u[j] + len(N)*x[i,j] <= len(N)-1,
                        name=f"mtz_{i}_{j}")

# 3. Hard precedence constraints INSIDE clusters: u[a] < u[b]
for (a,b) in hard_precedence:
    m.addConstr(u[a] + 0.0001 <= u[b], name=f"prec_{a}_{b}")

# 4. Linearization για position deviation: d[i] >= |u[i] - pos0[i]|
for i in N:
    m.addConstr(d[i] >= u[i] - pos0[i], name=f"d_pos_{i}_1")
    m.addConstr(d[i] >= pos0[i] - u[i], name=f"d_pos_{i}_2")

# -----------------------------
# WARM START
# -----------------------------

# Αρχικοποίηση x από την initial_tour
for (i,j) in x:
    x[i,j].start = 0

initial_edges = list(zip(initial_tour, initial_tour[1:]))
for (i,j) in initial_edges:
    if (i,j) in x:
        x[i,j].start = 1

# Αρχικοποίηση u από pos0
for i in N:
    u[i].start = pos0[i]

# d[i] αρχικά 0
for i in N:
    d[i].start = 0.0

# -----------------------------
# SOLVE
# -----------------------------
m.optimize()

print("\nOptimal tour edges:")
for (i,j), var in x.items():
    if var.X > 0.5:
        print(f"{i} -> {j}")

print("\nPositions (u[i]):")
for i in N:
    print(i, u[i].X)

print("\nTotal objective:", m.ObjVal)
print("Travel cost component:", travel_cost.getValue())
print("Soft penalty component:", soft_penalty.getValue())

