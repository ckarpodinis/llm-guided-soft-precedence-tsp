import json

# load uuid → step mapping
def load_steps_by_uuid(path="unified_steps.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["uuid"]: entry["step"] for entry in data}

# load uuid sequence
def load_uuid_sequence(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [e["uuid"] for e in data]


# Build solver_output.json from route
def write_solver_output(
    route,
    unified_steps_path="unified_steps.json",
    output_path="solver_output.json",
    depot="DEPOT"
):
    import json

    uuid_to_step = load_steps_by_uuid(unified_steps_path)

    output = []
    for u in route:
        if u == depot:
            continue

        step = uuid_to_step.get(u)
        if step is None:
            raise KeyError(f"UUID {u} not found in unified_steps.json")

        output.append({
            "uuid": u,
            "step": step
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(output)} steps to {output_path}")
