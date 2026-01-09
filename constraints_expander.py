import json
import re
from pathlib import Path

def get_value_by_path(obj, path):
    """Retrieve dotted-path value, return None if missing."""
    parts = path.split(".")
    cur = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def match_condition(step, cond, bindings):
    """Match one condition rule object."""
    path = cond.get("path")
    val = get_value_by_path(step, path)

    # ----- exists -----
    if "exists" in cond:
        should_exist = cond["exists"]
        exists = val is not None
        return exists == should_exist

    # ----- equals -----
    if "equals" in cond:
        expected = cond["equals"]

        # variable reference (e.g. "$container_id")
        if isinstance(expected, str) and expected.startswith("$"):
            varname = expected[1:]

            # Bind if not bound yet
            if varname not in bindings:
                if val is None:
                    return False
                bindings[varname] = val

            # Must match previously bound value
            return bindings[varname] == val

        else:
            return val == expected

    # ----- matches.glob -----
    if "matches" in cond:
        m = cond["matches"]
        if "glob" in m:
            glob = m["glob"]
            regex = "^" + re.escape(glob).replace("\\*", ".*") + "$"
            return val is not None and re.match(regex, val) is not None

    return False


def match_trigger(step, trigger):
    """Return (matched, bindings)."""
    conds = trigger["match"]["all"]
    bindings = {}

    for c in conds:
        if not match_condition(step, c, bindings):
            return False, {}
    return True, bindings


def substitute(obj, bindings):
    """Recursively substitute $variables in any string value."""
    if isinstance(obj, dict):
        return {k: substitute(v, bindings) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute(x, bindings) for x in obj]
    elif isinstance(obj, str):
        for k, v in bindings.items():
            obj = obj.replace(f"${k}", str(v))
        return obj
    else:
        return obj


def normalize_to_list(value):
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def expand_rules(steps, rules):
    """
    Expand rules into constraints with:
      - requires: steps before trigger
      - follows:  steps after trigger
    """
    results = []

    for idx, step in enumerate(steps):
        for rule in rules:
            matched, bindings = match_trigger(step, rule["trigger"])
            if not matched:
                continue

            requires = normalize_to_list(rule.get("requires"))
            follows = normalize_to_list(rule.get("follows"))

            expanded_requires = substitute(requires, bindings)
            expanded_follows = substitute(follows, bindings)

            constraint = {
                "rule_id": rule["id"],
                "description": rule.get("description", ""),
                "step_index": idx,
                "step": step
            }

            if expanded_requires:
                constraint["requires"] = expanded_requires

            if expanded_follows:
                constraint["follows"] = expanded_follows

            results.append(constraint)

    return {"constraints": results}


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
steps = json.loads(Path("sample_steps.json").read_text())
rules = json.loads(Path("sample_rules.json").read_text())

constraints = expand_rules(steps["steps"], rules["rules"])

Path("constraints.json").write_text(json.dumps(constraints, indent=2))

print("Wrote constraints.json")

