import matplotlib.pyplot as plt
import mplcursors
import numpy as np

# --------------------------------------------------
# Helper: Add jitter to reveal hidden points
# --------------------------------------------------
def jitter(x, scale=0.01):
    return x + np.random.uniform(-scale, scale, size=len(x))

# --------------------------------------------------
# Helper: experiment index + hover annotation
# --------------------------------------------------
def _add_hover(scatter, df, value_key):
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def _(sel):
        i = sel.index
        row = df.iloc[i]

        sel.annotation.set_text(
            f"exp={i}\n"
            f"pos={row['params.pos_lambda']}, "
            f"edge={row['params.edge_lambda']}, "
            f"cluster={row['params.cluster_lambda']}, "
            f"raw={row['params.raw_lambda']}\n"
            f"{value_key.rsplit('.', 1)[-1]}={sel.target[1]:.3f}"
        )


# --------------------------------------------------
# Helper: marginal aggregation
# --------------------------------------------------
def _group_mean(df, lam_key, value_key):
    return (
        df.groupby(f"params.{lam_key}")[value_key]
        .mean()
        .sort_index()
    )

# --------------------------------------------------
# Helper: structured annotation string
# --------------------------------------------------
def format_other_lambdas(row, lam_key):
    parts = []
    for k in ["pos_lambda", "edge_lambda", "cluster_lambda", "raw_lambda"]:
        if k != lam_key:
            short = k.replace("_lambda", "")
            parts.append(f"{short}={row[f'params.{k}']}")
    return ", ".join(parts)


# --------------------------------------------------
# Helper: dual series plot (GT vs Initial)
# --------------------------------------------------
def _plot_dual_series(ax, df, ykey_gt, ykey_init, ylabel):
    x = range(len(df))

    y_gt = df[ykey_gt]
    y_init = df[ykey_init]

    # ---- Ground truth ----
    ax.plot(
        x, y_gt,
        color="tab:blue",
        marker="o",
        linestyle="-",
        linewidth=1.5,
        label="vs GT"
    )
    sc_gt = ax.scatter(x, y_gt, color="tab:blue", s=40)

    # ---- Initial draft ----
    ax.plot(
        x, y_init,
        color="tab:orange",
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="vs initial"
    )
    sc_init = ax.scatter(x, y_init, color="tab:orange", s=40)

    # ---- Hover ----
    _add_hover(sc_gt, df, ykey_gt)
    _add_hover(sc_init, df, ykey_init)

    ax.set_ylabel(ylabel)
    ax.grid(True)


# --------------------------------------------------
# Global vs local displacement — structured + hover
# --------------------------------------------------
def plot_global_vs_local(df, ax, ref="ground_truth"):
    x = df[f"metrics.vs_{ref}.displacement.mean_abs_delta"]
    y = df[f"metrics.vs_{ref}.displacement.mean_delta_loc"]

    sc = ax.scatter(
        jitter(x, scale=0.005),
        jitter(y, scale=0.005),
        s=90,
        edgecolor="black",
        alpha=0.2,
    )

    ax.set_xlabel("Mean |Δ position|")
    ax.set_ylabel("Mean Δloc")
    ax.set_title(f"vs {'GT' if ref=='ground_truth' else 'initial'}")
    ax.grid(True)

#    # -------- Structured annotation (static) --------
#    for _, row in df.iterrows():
#        ax.annotate(
#            format_other_lambdas(row, lam_key),
#            (
#                row[f"metrics.vs_{ref}.displacement.mean_abs_delta"],
#                row[f"metrics.vs_{ref}.displacement.mean_delta_loc"],
#            ),
#            fontsize=8,
#            alpha=0.7,
#            xytext=(5, 5),
#            textcoords="offset points",
#        )

    # -------- Interactive hover --------
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        row = df.iloc[sel.index]
        sel.annotation.set_text(
            f"exp={i}\n"
            f"pos={row['params.pos_lambda']}\n"
            f"edge={row['params.edge_lambda']}\n"
            f"cluster={row['params.cluster_lambda']}\n"
            f"raw={row['params.raw_lambda']}\n\n"
            f"|Δ|={row[f'metrics.vs_{ref}.displacement.mean_abs_delta']:.2f}\n"
            f"Δloc={row[f'metrics.vs_{ref}.displacement.mean_delta_loc']:.2f}"
        )

    return sc

# --------------------------------------------------
# Ordering trade-off (for kendall tau only)
# --------------------------------------------------
def plot_ordering_tradeoff_by_lambda(df, ax, lam_key):
    x = df["metrics.vs_initial.kendall_tau"]
    y = df["metrics.vs_ground_truth.kendall_tau"]
    c = df[f"params.{lam_key}"]

    # scatter
    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="plasma",
        alpha=0.2,
        s=90,
        edgecolor="black",
    )

    # diagonal reference
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("Kendall τ vs initial (draft adherence)")
    ax.set_ylabel("Kendall τ vs ground truth (correctness)")
    ax.set_title(f"Ordering trade-off — colored by {lam_key}")
    ax.grid(True)

#    # --------------------------------------------------
#    # Structured static annotation (other lambdas)
#    # --------------------------------------------------
#    for _, row in df.iterrows():
#        ax.annotate(
#            format_other_lambdas(row, lam_key),
#            (
#                row["metrics.vs_initial.kendall_tau"],
#                row["metrics.vs_ground_truth.kendall_tau"],
#            ),
#            fontsize=8,
#            alpha=0.6,
#            xytext=(5, 5),
#            textcoords="offset points",
#        )

    # --------------------------------------------------
    # Interactive hover (reuse shared helper)
    # --------------------------------------------------
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        row = df.iloc[i]

        sel.annotation.set_text(
            f"exp={i}\n"
            f"pos={row['params.pos_lambda']}, "
            f"edge={row['params.edge_lambda']}, "
            f"cluster={row['params.cluster_lambda']}, "
            f"raw={row['params.raw_lambda']}\n\n"
            f"τ(initial)={row['metrics.vs_initial.kendall_tau']:.3f}\n"
            f"τ(GT)={row['metrics.vs_ground_truth.kendall_tau']:.3f}"
        )

    return sc

# --------------------------------------------------
# Ordering trade-off (GT vs Initial)
# --------------------------------------------------
def plot_ordering_tradeoff_gt_vs_init(
    df,
    ax,
    metric_key,
    *,
    title=None,
    cmap="plasma"
):
    """
    Plot ordering trade-off for a given metric.

    x-axis: metric vs initial
    y-axis: metric vs ground truth

    Parameters
    ----------
    metric_key : str
        e.g. "kendall_tau", "lcs", "bigram", "trigram",
             "breakpoints", "breakrate", "normalized_spearmanf"
    """

    x_col = f"metrics.vs_initial.{metric_key}"
    y_col = f"metrics.vs_ground_truth.{metric_key}"

    if x_col not in df or y_col not in df:
        raise KeyError(f"Metric '{metric_key}' not found in DataFrame")

    x = df[x_col]
    y = df[y_col]

    # scatter
    sc = ax.scatter(
        jitter(x, scale=0.005),
        jitter(y, scale=0.005),
        alpha=0.2,
        s=90,
        edgecolor="black",
    )

    # Optional diagonal reference (only meaningful for bounded similarity metrics)
    #if x.min() >= 0 and x.max() <= 1 and y.min() >= 0 and y.max() <= 1:
    #    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot([min(x.min(), y.min()), max(x.max(), y.max())], [min(x.min(), y.min()), max(x.max(), y.max())], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel(f"{metric_key.split('.')[-1]} vs initial")
    ax.set_ylabel(f"{metric_key.split('.')[-1]} vs ground truth")
    ax.set_title(title or f"Ordering trade-off — {metric_key.split('.')[-1]}")
    ax.grid(True)

    # --------------------------------------------------
    # Interactive hover
    # --------------------------------------------------
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        row = df.iloc[i]

        sel.annotation.set_text(
            f"exp={i}\n"
            f"pos={row['params.pos_lambda']}, "
            f"edge={row['params.edge_lambda']}, "
            f"cluster={row['params.cluster_lambda']}, "
            f"raw={row['params.raw_lambda']}\n\n"
            f"{metric_key} (initial)={row[x_col]:.3f}\n"
            f"{metric_key} (GT)={row[y_col]:.3f}"
        )

    return sc

# --------------------------------------------------
# Ordering trade-off (generic)
# --------------------------------------------------
def plot_ordering_tradeoff_generic(
    df,
    ax,
    metric1_key,
    metric2_key,
    *,
    title=None,
    cmap="plasma"
):
    """
    Plot ordering trade-off for a given metrics.

    x-axis: metric1
    y-axis: metric2

    Parameters
    ----------
    metric1_key : str
    metric2_key : str
        e.g. "kendall_tau", "lcs", "bigram", "trigram",
             "breakpoints", "breakrate", "normalized_spearmanf"
    """

    x_col = f"metrics.{metric1_key}"
    y_col = f"metrics.{metric2_key}"

    if x_col not in df or y_col not in df:
        raise KeyError(f"Metrics '{metric1_key}' or '{metric2_key}' not found in DataFrame")

    x = df[x_col]
    y = df[y_col]

    # scatter
    sc = ax.scatter(
        jitter(x, scale=0.005),
        jitter(y, scale=0.005),
        alpha=0.2,
        s=90,
        edgecolor="black",
    )

    ax.set_xlabel(f"{metric1_key.split('.')[-1]}")
    ax.set_ylabel(f"{metric2_key.split('.')[-1]}")
    ax.set_title(title or f"Trade-off — {metric1_key.split('.')[-1]}, {metric2_key.split('.')[-1]}")
    ax.grid(True)

    # --------------------------------------------------
    # Interactive hover
    # --------------------------------------------------
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        row = df.iloc[i]

        sel.annotation.set_text(
            f"exp={i}\n"
            f"pos={row['params.pos_lambda']}, "
            f"edge={row['params.edge_lambda']}, "
            f"cluster={row['params.cluster_lambda']}, "
            f"raw={row['params.raw_lambda']}\n\n"
            f"{metric1_key}={row[x_col]:.3f}\n"
            f"{metric2_key}={row[y_col]:.3f}"
        )

    return sc

def _plot_series(ax, df, ykey, ylabel):
    x = range(len(df))
    y = df[ykey]

    sc = ax.plot(x, y, marker="o", linewidth=1)[0]
    scatter = ax.scatter(x, y, s=40)

    ax.set_ylabel(ylabel)
    ax.grid(True)

    _add_hover(scatter, df, ykey)


def plot_solver_behavior(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    _plot_series(
        axes[0, 0], df, "solver.gap", "Solver MIP gap"
    )
    axes[0, 0].set_title("Solver convergence")

    _plot_series(
        axes[0, 1], df, "slacks.cluster.sum", "Cluster slack (sum)"
    )
    axes[0, 1].set_title("Cluster feasibility")

    _plot_series(
        axes[1, 0], df, "slacks.raw.sum", "Raw slack (sum)"
    )
    axes[1, 0].set_title("Logical feasibility")

    _plot_series(
        axes[1, 1], df, "solver.nodes", "B&B nodes explored"
    )
    axes[1, 1].set_title("Search effort (nodes)")

    fig.suptitle("Figure A — Solver behavior & feasibility", fontsize=14)
    return fig


def plot_ordering_quality(df, ref):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    prefix = f"metrics.vs_{ref}"

    metrics = [
        ("kendall_tau", "Kendall τ"),
        ("spearman", "Spearman ρ"),
        ("normalized_spearmanf", "Normalized Spearman F"),
        ("lcs", "LCS length"),
    ]

    for ax, (key, label) in zip(axes.flat, metrics):
        _plot_series(
            ax, df, f"{prefix}.{key}", label
        )
        ax.set_title(label)

    fig.suptitle(
        f"Figure B — Ordering quality vs {ref.replace('_', ' ')}",
        fontsize=14
    )
    return fig
#def plot_ordering_quality(df):
#    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
#
#    metrics = [
#        ("kendall_tau", "Kendall τ"),
#        ("spearman", "Spearman ρ"),
#        ("edit_distance", "Edit distance"),
#        ("lcs", "LCS length"),
#    ]
#
#    for ax, (key, label) in zip(axes.flat, metrics):
#        _plot_dual_series(
#            ax,
#            df,
#            f"metrics.vs_ground_truth.{key}",
#            f"metrics.vs_initial.{key}",
#            label,
#        )
#        ax.set_title(label)
#
#    axes[0, 0].legend()
#    fig.suptitle(
#        "Figure B — Ordering quality: correctness vs draft adherence",
#        fontsize=14
#    )
#    return fig


def plot_displacement(df, ref):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

    prefix = f"metrics.vs_{ref}.displacement"

    metrics = [
        ("mean_abs_delta", "Mean |Δ position|"),
        ("max_abs_delta", "Max |Δ position|"),
        ("mean_delta_loc", "Mean Δloc"),
        ("max_delta_loc", "Max Δloc"),
        ("frac_delta_loc_positive", "Frac Δloc > 0"),
        ("local_stability", "Local stability"),
    ]

    for ax, (key, label) in zip(axes.flat, metrics):
        _plot_series(
            ax, df, f"{prefix}.{key}", label
        )
        ax.set_title(label)

    fig.suptitle(
        f"Figure D — Displacement vs {ref.replace('_', ' ')}",
        fontsize=14
    )
    return fig
#def plot_displacement(df):
#    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
#
#    metrics = [
#        ("mean_abs_delta", "Mean |Δ position|"),
#        ("max_abs_delta", "Max |Δ position|"),
#        ("mean_delta_loc", "Mean Δloc"),
#        ("max_delta_loc", "Max Δloc"),
#        ("frac_delta_loc_positive", "Frac Δloc > 0"),
#        ("local_stability", "Local stability"),
#    ]
#
#    for ax, (key, label) in zip(axes.flat, metrics):
#        _plot_dual_series(
#            ax,
#            df,
#            f"metrics.vs_ground_truth.displacement.{key}",
#            f"metrics.vs_initial.displacement.{key}",
#            label,
#        )
#        ax.set_title(label)
#
#    axes[0, 0].legend()
#    fig.suptitle(
#        "Figure C — Displacement: global change vs local coherence",
#        fontsize=14
#    )
#    return fig

def plot_permutation(df, ref):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    prefix = f"metrics.vs_{ref}"

    metrics = [
        ("bigram", "bi-gram Overlap"),
        ("trigram", "tri-gram Overlap"),
        ("breakpoints", "Breakpoints"),
        ("breakrate", "Adjacency Break Rate"),
    ]

    for ax, (key, label) in zip(axes.flat, metrics):
        _plot_series(
            ax, df, f"{prefix}.{key}", label
        )
        ax.set_title(label)

    fig.suptitle(
        f"Figure C — Permutations vs {ref.replace('_', ' ')}",
        fontsize=14
    )
    return fig
