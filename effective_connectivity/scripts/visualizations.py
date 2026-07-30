import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

# Tableau-20 muted palette, reordered as 10 strong hues then their 10 soft variants,
# so small groups get maximally-distinct colours and large factor groups still get a
# unique, understated colour per row.
_TAB20 = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D",
          "#B6992D", "#F1CE63", "#499894", "#86BCB6", "#E15759", "#FF9D9A",
          "#79706E", "#BAB0AC", "#D37295", "#FABFD2", "#B07AA1", "#D4A6C8",
          "#9D7660", "#D7B5A6"]
MUTED = [to_rgb(c) for c in _TAB20[0::2] + _TAB20[1::2]]
N_BINS = 600                      # x-resolution of the density strips
# --- ridgeline height/transparency knobs -------------------------------------
PER_UNIT = 0.9                    # row-heights per 1x "as-expected" density (nominal vertical scale)
MAX_RISE = 5.0                    # global cap: the tallest mountain across ALL rows is limited to this
                                  # many row-heights; when exceeded, every row is rescaled by the same
                                  # factor, so heights stay comparable and nothing runs off the panel
TOP_ROOM = 0.6                    # fixed headroom (row-heights) reserved above the top row. The panel top
                                  # is pinned here instead of tracking the tallest spike, so one big
                                  # mountain no longer inflates the whole figure; taller ones overflow
                                  # upward past the panel into the scatter plot above (see _ridgeline).
RIDGE_ALPHA = 0.28                # fill transparency (lower = more see-through); heavy overlap stays legible
RIDGE_EDGE_ALPHA = 0.7            # outline transparency; drawn in the row's own hue, not black
LABEL_LIFT = 0.5                  # row-units to raise a level's label off its baseline, so it sits in the
                                  # body of its own upward-rising mountain instead of at its foot
SMOOTH_SIGMA = 4.0                # gaussian smoothing of per-bin counts (in bins)


def _row_color(idx):
    """One muted colour per row within a factor group, cycled if a group ever
    exceeds the palette. Height already encodes density-relative-to-expected, so
    colour here is pure row identity."""
    return MUTED[idx % len(MUTED)]


def _ridgeline(ax, xs, row, rise, color):
    """Draw one level's 'mountain' at baseline `row`, rising upward on the
    y-flipped axis (toward smaller y). `rise` is the height in row units (density
    relative to expected, times the global scale). clip_on=False lets a ridge
    taller than the reserved headroom overflow past the panel top into the plot
    above -- ridges only rise, never dip below baseline, so there's no spill."""
    top = row - rise
    ax.fill_between(xs, row, top, color=color, alpha=RIDGE_ALPHA, lw=0, zorder=2, clip_on=False)
    ax.plot(xs, top, color=color, lw=0.9, alpha=RIDGE_EDGE_ALPHA, zorder=2.5, clip_on=False)


def plot_specification_curve(spec_df, results_df, output_file=None, title="Specification Curve Analysis",
                            figsize=(13, None), plot_type='barcode'):
    """
    Creates a specification-curve visualization: top panel is the sorted contrast
    effect per specification; bottom panel has one ridgeline per factor level, where
    ridge height is that level's density relative to expected along the sorted curve
    (peaks = over-represented, dips = under-represented). Colour is row identity --
    one hue per row, shared by the ridge, its baseline and its label.

    Parameters:
        spec_df (DataFrame): DataFrame containing specification factors
        results_df (DataFrame): DataFrame containing contrast results
        output_file (str, optional): Path to save the plot. If None, displays the plot.
        title (str, optional): Title for the plot.
        figsize (tuple): Figure width and height. If height is None, it is calculated
                         from the number of factor levels.
        plot_type: unused, kept for call-site compatibility (the ridgeline density
                   approach replaces the old barcode/density switch).

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    if len(spec_df) == 0:
        print("No data available for specification curve analysis.")
        return None

    df = pd.concat([spec_df, results_df[["contrast"]]], axis=1)
    df = df.sort_values("contrast").reset_index(drop=True)
    factors = [col for col in spec_df.columns if col != "node_pair"]

    n = len(df)
    x = np.arange(n)
    bin_idx = np.minimum((x * N_BINS) // n, N_BINS - 1)
    bin_counts = np.bincount(bin_idx, minlength=N_BINS)

    # row layout: a bold header per factor, then one row per level
    row_pos, levels, pos = {}, {}, 0.0
    for f in factors:
        row_pos[(f, "header")] = pos
        pos += 1.3                     # clear gap so the header never sits on the first level's label
        levels[f] = sorted(df[f].dropna().unique())
        for i, lvl in enumerate(levels[f]):
            row_pos[(f, lvl)] = pos + i
        pos += len(levels[f]) + 0.5
    total_rows = pos

    # pass 1: per-level smoothed density relative to expected (1.0 = as-expected).
    # gmax is the tallest peak across every row; the global `scale` below caps it at
    # MAX_RISE row-heights so one spike can't run off the panel, rescaling all rows
    # together (heights stay comparable between rows).
    cells, gmax = {}, 0.0
    for f in factors:
        for lvl in levels[f]:
            in_lvl = (df[f].to_numpy() == lvl)
            lvl_per_bin = np.bincount(bin_idx[in_lvl], minlength=N_BINS)
            with np.errstate(invalid="ignore", divide="ignore"):
                local_frac = np.where(bin_counts > 0, lvl_per_bin / bin_counts, 0.0)
            h = gaussian_filter1d(local_frac / (in_lvl.mean() or 1.0), SMOOTH_SIGMA)
            cells[(f, lvl)] = (h, in_lvl)
            gmax = max(gmax, h.max())
    scale = min(PER_UNIT, MAX_RISE / gmax) if gmax > 0 else PER_UNIT

    fig_height = 5 + total_rows * 0.28 if figsize[1] is None else figsize[1]
    fig = plt.figure(figsize=(figsize[0], fig_height))
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan=5)

    yv = df["contrast"].to_numpy()
    ax1.scatter(x, yv, s=2, c=yv, cmap="viridis")
    mean_lbl = "0" if abs(yv.mean()) < 1e-9 else f"{yv.mean():.3g}"
    ax1.axhline(yv.mean(), color="r", lw=0.9, alpha=0.5, label=f"mean = {mean_lbl}")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("Contrast Effect")
    ax1.set_title(title)
    ax1.set_xticks([])
    ax1.margins(x=0.01, y=0.1)

    ax2.set_xlim(0, n)
    # flip y (factors top-down). The panel top is pinned a fixed TOP_ROOM above the
    # topmost row (the first header at the smallest y), NOT at the tallest peak, so a
    # single big spike can't inflate the figure; taller mountains overflow upward past
    # this edge into the plot above (_ridgeline drops the axes-box clip).
    top_row = min(row_pos.values())
    ax2.set_ylim(total_rows + 0.5, top_row - TOP_ROOM)
    ax2.set_yticks([])
    ax2.grid(False)
    label_x = -0.01 * n
    bin_x = (np.arange(N_BINS) + 0.5) * n / N_BINS      # bin centers in x (spec-rank) coords

    # bottom axis in *contrast* units: nice ticks placed at their rank via the
    # sorted curve (nonlinear map), so x reads in the top panel's units. The dashed
    # line at the rank where the contrast crosses 0 marks the sign boundary.
    ticks = [v for v in MaxNLocator(nbins=7).tick_values(yv.min(), yv.max()) if yv.min() <= v <= yv.max()]
    tick_pos, tick_lbl, last = [], [], -np.inf
    for p, v in zip(np.searchsorted(yv, ticks), ticks):
        if p - last >= 0.04 * n:            # the steep curve tail crams extreme ticks together; keep one per cluster
            tick_pos.append(p); tick_lbl.append(f"{v:g}"); last = p
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl)
    ax2.set_xlabel("Contrast Effect   (specifications sorted low → high)")
    zero_rank = int(np.searchsorted(yv, 0.0))
    if 0 < zero_rank < n:
        ax2.axvline(zero_rank, color="0.35", lw=0.8, ls="--", zorder=1)

    for fi, f in enumerate(factors):
        # alternating block background separates the factor groups
        if fi % 2 == 0 and levels[f]:
            ax2.axhspan(row_pos[(f, "header")] - 0.5, row_pos[(f, levels[f][-1])] + 0.5,
                        color="0.9", alpha=0.25, zorder=0)
        ax2.text(label_x, row_pos[(f, "header")], f, ha="right", va="center",
                 fontweight="bold", fontsize=11, zorder=6, clip_on=False)
        for i, lvl in enumerate(levels[f]):
            row = row_pos[(f, lvl)]
            color = _row_color(i)
            # label, baseline line and ridge share one hue -> unambiguous which mountain
            # each label names, even where ridges overlap heavily. The label uses a
            # darkened hue so pale colours (yellow/cyan) stay readable on white.
            label_color = tuple(0.55 * c for c in color[:3])
            ax2.axhline(row, color=color, lw=0.9, alpha=0.55, zorder=0.5)
            ax2.text(label_x, row - LABEL_LIFT, str(lvl), ha="right", va="center", fontsize=9,
                     color=label_color, fontweight="bold")

            h, in_lvl = cells[(f, lvl)]
            _ridgeline(ax2, bin_x, row, h * scale, color)

            # median-rank tick: where half this level's specifications fall (== its
            # median effect, since x is sorted by contrast); short mark into the mountain
            if in_lvl.any():
                mx = np.median(x[in_lvl])
                ax2.plot([mx, mx], [row, row - 0.16], color="k", lw=1.0, zorder=4)

    plt.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.06, hspace=0.02)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig

# For future
def visualize_force_directed_graph(nodes_list, contrasts_list, node_labels=None):
    """
    Creates a force-directed graph visualization using NetworkX and Matplotlib.
    Uses node_labels to color nodes by network if available.
    
    Parameters:
    nodes_list: List of node pairs (tuples)
    contrasts_list: List of contrast values corresponding to node pairs
    node_labels: Dictionary mapping nodes to their network labels (e.g., 'DMN', 'SN', 'CEN')
    
    Returns:
    None (displays the graph)
    """
    # Create a graph
    G = nx.Graph()
    
    # Add edges with contrast values as weights
    for (node_pair, contrast) in zip(nodes_list, contrasts_list):
        # Add edge with weight attribute
        G.add_edge(node_pair[0], node_pair[1], weight=contrast)
    
    # Create positions using force-directed layout
    # k is the optimal distance between nodes
    # Higher weights = stronger attraction
    pos = nx.spring_layout(G, k=0.15, iterations=100, 
                           weight='weight', seed=42)
    
    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize weights for visualization (larger contrast = thicker line)
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [3 * w / max_weight for w in edge_weights]
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    
    # Set up colors for networks
    network_colors = {
        'DMN': 'royalblue',
        'SN': 'crimson',
        'CEN': 'forestgreen',
        None: 'gray'  # Default color for unlabeled nodes
    }
    
    # Assign colors to nodes based on their network
    if node_labels:
        node_colors = [network_colors.get(node_labels.get(node, None), 'gray') for node in G.nodes()]
        
        # Add legend for networks
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, label=network)
                           for network, color in network_colors.items() if network]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        # Default color if no labels
        node_colors = ['skyblue' for _ in G.nodes()]
    
    # Draw edges with varying thickness based on weight
    nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                          alpha=0.7, edge_color='gray')
    
    # Draw nodes with colors based on network
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, 
                          alpha=0.8, edgecolors='black')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    # Draw edge labels (contrast values) with reduced visibility
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8,
                                font_color='gray',
                                alpha=0.6,
                                bbox=dict(boxstyle="round,pad=0.1", 
                                         alpha=0.2,
                                         ec="none",
                                         fc="white"))
    
    plt.title("Force-Directed Graph with Contrast Values", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()