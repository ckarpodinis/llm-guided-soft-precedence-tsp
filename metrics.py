# Position-based comparison (Δ, Δrel)
def position_differences(pred, gt):
    gt_pos = {u: i for i, u in enumerate(gt)}
    pred_pos = {u: i for i, u in enumerate(pred)}

    rows = []
    for u in pred:
        if u not in gt_pos:
            continue
        Δ = pred_pos[u] - gt_pos[u]
        rows.append((u, gt_pos[u], pred_pos[u], Δ))

    return rows

# Relative displacement (local Δrel)
def relative_displacement(pred, gt):
    gt_pos = {u: i for i, u in enumerate(gt)}
    pred_pos = {u: i for i, u in enumerate(pred)}

    deltas = {}
    for u in pred:
        if u not in gt_pos:
            continue

        i = gt_pos[u]
        neigh = []
        if i > 0:
            neigh.append(gt[i - 1])
        if i < len(gt) - 1:
            neigh.append(gt[i + 1])

        Δrel = 0
        for v in neigh:
            if v in pred_pos:
                Δrel += abs(
                    (pred_pos[u] - pred_pos[v]) -
                    (gt_pos[u] - gt_pos[v])
                )

        deltas[u] = Δrel

    return deltas

# Kendall tau & Spearman
from scipy.stats import kendalltau, spearmanr

def rank_correlations(pred, gt):
    common = [u for u in gt if u in pred]
    gt_r = list(range(len(common)))
    pr_r = [pred.index(u) for u in common]

    kt, _ = kendalltau(gt_r, pr_r)
    sp, _ = spearmanr(gt_r, pr_r)

    return kt, sp

# Edit distance (sequence-level)
import difflib

def edit_distance(pred, gt):
    sm = difflib.SequenceMatcher(a=gt, b=pred)
    return 1 - sm.ratio(), sm.get_opcodes()

# LCS length
def lcs_length(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]

def local_relative_displacement_pos(i, pos_ref, u_pos, k=2):
    """
    Counts how many local neighbor orderings of i were inverted.
    k = window size on each side
    """
    p0 = pos_ref[i]
    ui = u_pos[i]

    neighbors = [
        j for j in pos_ref
        if j != i and abs(pos_ref[j] - p0) <= k
    ]

    inv = 0
    for j in neighbors:
        if (p0 - pos_ref[j]) * (ui - u_pos[j]) < 0:
            inv += 1

    return inv


def displacement(u_pos, pos_ref, k=2, clusters=None):
    deltas = {}
    deltas_loc = {}

    for i in pos_ref:
        if i not in u_pos:
            continue

        Δ = u_pos[i] - pos_ref[i]
        Δloc = local_relative_displacement_pos(i, pos_ref, u_pos, k)

        deltas[i] = Δ
        deltas_loc[i] = Δloc

    abs_deltas = [abs(v) for v in deltas.values()]
    loc_vals = list(deltas_loc.values())

    metrics = {
        "mean_abs_delta": sum(abs_deltas) / len(abs_deltas),
        "max_abs_delta": max(abs_deltas),
        "mean_delta_loc": sum(loc_vals) / len(loc_vals),
        "max_delta_loc": max(loc_vals),
        "frac_delta_loc_positive": sum(v > 0 for v in loc_vals) / len(loc_vals),
        "local_stability": 1 - (sum(loc_vals) / len(loc_vals)) / (2*k), # 1 - mean_delta_loc / (2*k)
    }

    if clusters:
        metrics["by_cluster"] = {}
        for ck, C in clusters.items():
            C = [i for i in C if i in deltas]
            if not C:
                continue
            metrics["by_cluster"][ck] = {
                "mean_abs_delta": sum(abs(deltas[i]) for i in C) / len(C),
                "mean_delta_loc": sum(deltas_loc[i] for i in C) / len(C),
            }

    return metrics


# Pretty comparison table
def print_comparison(pred, gt):
    gt_pos = {u: i for i, u in enumerate(gt)}
    pred_pos = {u: i for i, u in enumerate(pred)}

    print("\nUUID     gt  pred   Δ   Δrel")
    print("-" * 40)

    Δrel = relative_displacement(pred, gt)

    for u in pred:
        if u not in gt_pos:
            continue
        print(
            f"{u[:6]}  "
            f"{gt_pos[u]:>3}  "
            f"{pred_pos[u]:>4}  "
            f"{pred_pos[u]-gt_pos[u]:>+3}  "
            f"{Δrel[u]:>+3}"
        )
