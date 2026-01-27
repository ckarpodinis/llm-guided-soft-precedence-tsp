from typing import List, Tuple

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


# Local Relative displacement
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


# Absolute and Local Relative displacement
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


def ranks(seq):
    return {item: i for i, item in enumerate(seq)}

# adjacency overlap
def adjacency_pairs(seq: List) -> List[Tuple]:
    """Directed adjacency pairs (a->b) for consecutive elements."""
    return list(zip(seq[:-1], seq[1:]))

def adjacency_overlap(a: List, b: List) -> float:
    """
    Fraction of adjacencies in a that are also present in b.
    (a is the reference, e.g., ground truth; b is prediction)
    """
    A = set(adjacency_pairs(a))
    B = set(adjacency_pairs(b))
    return len(A & B) / len(A) if A else 1.0


# n-gram overlap
def ngrams(seq: List, n: int) -> List[Tuple]:
    """All contiguous n-grams as tuples."""
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def ngram_overlap(a: List, b: List, n: int) -> float:
    """
    Fraction of n-grams in a that appear in b.
    (block-move tolerant when n is small, like 2 or 3)
    """
    A = set(ngrams(a, n))
    B = set(ngrams(b, n))
    return len(A & B) / len(A) if A else 1.0


# Adjacency break rate
def adjacency_break_rate(candidate: List, reference: List) -> float:
    """
    1 - adjacency_overlap(reference, candidate)

    Interpretable as: "What fraction of reference adjacencies got broken?"
    Example: reference=initial, candidate=prediction
    """
    return 1.0 - adjacency_overlap(reference, candidate)


# Number of breakpoints
def num_breakpoints(candidate: List, reference: List) -> int:
    """
    Breakpoints measured by mapping candidate into reference-index space
    and counting how many consecutive elements are NOT consecutive indices.

    This is common in genome rearrangement / permutation distance settings.

    For permutations:
      - 0 breakpoints => candidate is identical to reference
      - Small number => mostly the same, possibly via block moves
    """
    pos_ref = {item: i for i, item in enumerate(reference)}
    mapped = [pos_ref[item] for item in candidate]

    # Count adjacent mapped positions that are not consecutive
    bp = 0
    for x, y in zip(mapped[:-1], mapped[1:]):
        if y != x + 1:
            bp += 1
    return bp


# Spearman Footrule
def spearman_footrule(candidate: List, reference: List) -> float:
    """
    Spearman footrule distance between two permutations reference and candidate.
    Lower = more similar, 0 = identical.
    """
    ra = ranks(reference)
    rb = ranks(candidate)
    return sum(abs(ra[x] - rb[x]) for x in reference)


# Normalized Spearman
def spearman_footrule_normalized(candidate: List, reference: List) -> float:
    n = len(reference)
    F = spearman_footrule(reference, candidate)
    return F / (n*n//2)
