import json
import uuid
import hashlib
from pathlib import Path

# ---------------------------------------------------------
# Deterministic UUID from JSON structure
# ---------------------------------------------------------
def deterministic_uuid(step_obj):
    s = json.dumps(step_obj, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------
steps = json.loads(Path("sample_steps.json").read_text())["steps"]
constraints = json.loads(Path("constraints.json").read_text())["constraints"]


# ---------------------------------------------------------
# Maps:
#   step_index → list of required steps (before)
#   step_index → list of follows steps (after)
# ---------------------------------------------------------
requires_by_step = {}
follows_by_step = {}

for c in constraints:
    trigger_index = c["step_index"]

    reqs = c.get("requires", [])
    if isinstance(reqs, dict):
        reqs = [reqs]
    requires_by_step.setdefault(trigger_index, []).extend(reqs)

    foll = c.get("follows", [])
    if isinstance(foll, dict):
        foll = [foll]
    follows_by_step.setdefault(trigger_index, []).extend(foll)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def canonical(obj):
    return json.dumps(obj, sort_keys=True)

uuid_cache = {}

def get_uuid(step_obj):
    key = canonical(step_obj)
    if key not in uuid_cache:
        uuid_cache[key] = deterministic_uuid(step_obj)
    return uuid_cache[key]


# ---------------------------------------------------------
# Final unified steps + dedup
# ---------------------------------------------------------
seen = set()
final_steps = []


# ---------------------------------------------------------
# Constraint chains for each rule application
# ---------------------------------------------------------
uuid_constraints = []  # list of objects: { "0": "...", "1": "...", ... }


def add_step_if_new(step_obj):
    """Dedup add into final_steps."""
    key = canonical(step_obj)
    if key not in seen:
        seen.add(key)
        final_steps.append(step_obj)

# ---------------------------------------------------------
# Insert:
#   requires BEFORE trigger
#   follows AFTER trigger
# Build chain: requires... -> trigger -> follows...
# ---------------------------------------------------------
for idx, step in enumerate(steps):
    reqs = requires_by_step.get(idx, [])
    foll = follows_by_step.get(idx, [])

    if reqs or foll:
        chain = {}
        pos = 0

        # requires (before)
        for r in reqs:
            chain[str(pos)] = get_uuid(r)
            pos += 1

        # trigger
        chain[str(pos)] = get_uuid(step)
        pos += 1

        # follows (after)
        for f in foll:
            chain[str(pos)] = get_uuid(f)
            pos += 1

        uuid_constraints.append(chain)

    # ---------- LINEAR ORDERING FIX ----------

    # 1. add requires BEFORE trigger
    for r in reqs:
        add_step_if_new(r)

    # 2. add trigger
    add_step_if_new(step)

    # 3. add follows AFTER trigger
    for f in foll:
        add_step_if_new(f)


# ---------------------------------------------------------
# Construct unified_steps.json
# ---------------------------------------------------------
unified_steps = [{"uuid": get_uuid(step), "step": step} for step in final_steps]


# ---------------------------------------------------------
# Write outputs
# ---------------------------------------------------------
Path("unified_steps.json").write_text(json.dumps(unified_steps, indent=2))
Path("uuid_constraints.json").write_text(json.dumps(uuid_constraints, indent=2))

print("Wrote unified_steps.json and uuid_constraints.json")

