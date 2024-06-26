import numpy as np
import anndata as ad

from .utils.figure_common import (
    cluster_on_rows,
    plot_umap,
    pool_palette,
    expt_palette,
    add_legend_in_plot
)
from .figure_constants import (
    FIGURE_1A_LFC_THRESHOLD,
    FIGURE_1A_MINMAX,
    FIGURE_1A_PADJ_THRESHOLD,
    FIG1B_FILE_NAME,
    FIG_RAPA_LEGEND_VERTICAL_FILE_NAME,
    MAIN_FIGURE_DPI,
    FIGURE_1_FILE_NAME
)
from .utils.figure_data import load_rapa_bulk_data, rapa_bulk_times
from .utils.Figure_deseq import run_deseq

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_figure_1(sc_data, save=True):
    # LOAD AND PROCESS BULK RAPAMYCIN DATA FOR HEATMAP PANEL #
    rapa_bulk, rapa_bulk_meta = load_rapa_bulk_data()
    rapa_bulk = ad.AnnData(
        rapa_bulk,
        obs=rapa_bulk_meta
    )

    print("Running DESeq2")
    res = run_deseq(
        rapa_bulk,
        'Time',
        "0.0",
        quiet=True,
        lfc_null=FIGURE_1A_LFC_THRESHOLD
    )

    plot_genes = res.loc[
        res["padj"] < FIGURE_1A_PADJ_THRESHOLD,
        :
    ].index.unique()

    plot_genes = res.loc[res.index.isin(plot_genes), :]
    plot_genes = plot_genes.pivot(
        columns="Time",
        values="log2FoldChange"
    ).reindex(
        rapa_bulk_times(include_0=True),
        axis=1
    ).fillna(0)

    print("Clustering Results")
    plot_y_order = cluster_on_rows(plot_genes)
    plot_genes = plot_genes.reindex(plot_y_order, axis=0)

    plot_x_labels = list(
        map(str, map(int, map(float, rapa_bulk_times(include_0=True))))
    )
    plot_x_labels[1] = "2.5"
    plot_x_labels[3] = "7.5"

    # BUILD FIGURE #
    fig_refs = {}
    fig = plt.figure(figsize=(6, 3.25), dpi=MAIN_FIGURE_DPI)

    axd = {
        "hm": fig.add_axes([0.05, 0.375, 0.25, 0.525]),
        "hm_cbar": fig.add_axes([0.05, 0.15, 0.25, 0.025]),
        "image": fig.add_axes([0.325, 0.6, 0.675, 0.3]),
        "umap_1": fig.add_axes([0.35, 0.08, 0.25, 0.45]),
        "umap_1_legend": fig.add_axes([0.6, 0.08, 0.075, 0.45]),
        "umap_2": fig.add_axes([0.725, 0.08, 0.25, 0.45]),
        "umap_2_legend": fig.add_axes([0.735, 0.125, 0.1, 0.1]),
    }

    # PLOT HEATMAP #
    fig_refs.update(
        _draw_bulk_heatmap(
            plot_genes,
            axd["hm"],
            cbar_ax=axd["hm_cbar"],
            cbar_label="Log${}_2$ FC",
            x_labels=plot_x_labels,
        )
    )

    axd["hm"].set_ylabel("Genes", size=8)
    axd["hm"].set_xlabel("Time (minutes)", size=8)
    axd["hm"].set_title("Rapamycin Treatment", size=8)
    axd["hm"].set_title("A", loc="left", x=-0.1, weight="bold")

    # DRAW SCHEMATIC #
    axd["image"].imshow(plt.imread(FIG1B_FILE_NAME), aspect="equal")
    axd["image"].axis("off")
    axd["image"].set_title("B", loc="left", weight="bold")

    # DRAW UMAPS #
    fig_refs["umap_1"] = plot_umap(
        sc_data.all_data,
        axd["umap_1"],
        color="Pool",
        palette=pool_palette(),
        alpha=0.2,
        size=1,
    )

    fig_refs["umap_2"] = plot_umap(
        sc_data.all_data,
        axd["umap_2"],
        color="Experiment",
        palette=expt_palette(),
        alpha=0.2,
        size=1,
    )

    axd["umap_1"].set_xlabel("UMAP1", size=8)
    axd["umap_1"].set_ylabel("UMAP2", size=8)
    axd["umap_1"].set_title("C", loc="left", weight="bold")
    axd["umap_1"].annotate(
        f"n = {sc_data.all_data.shape[0]}",
        xy=(0.45, 0.05),
        xycoords="axes fraction",
        size=7,
    )

    axd["umap_2"].tick_params(labelsize=8)
    axd["umap_2"].set_xlabel("UMAP1", size=8)
    axd["umap_2"].set_ylabel("UMAP2", size=8)
    axd["umap_2"].set_title("D", loc="left", weight="bold")

    axd["umap_1_legend"].imshow(
        plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME), aspect="equal"
    )
    axd["umap_1_legend"].axis("off")

    add_legend_in_plot(
        axd["umap_2_legend"],
        expt_palette(),
        sc_data.all_data.obs["Experiment"].dtype.categories.values,
        loc="center",
        bbox_to_anchor=None,
        title="Replicate"
    )
    axd["umap_2_legend"].axis("off")

    if save:
        fig.savefig(FIGURE_1_FILE_NAME + ".png", facecolor="white")

    return fig


def _draw_bulk_heatmap(
    heatmap_data,
    ax,
    cbar_ax=None,
    cbar_label=None,
    x_labels=None
):
    fig_refs = {}

    fig_refs["hm_im"] = ax.imshow(
        heatmap_data,
        cmap="bwr",
        aspect="auto",
        interpolation="nearest",
        vmin=-1 * FIGURE_1A_MINMAX,
        vmax=FIGURE_1A_MINMAX,
    )

    if cbar_ax is None:
        fig_refs["hm_divider"] = make_axes_locatable(ax)
        fig_refs["hm_toppad"] = fig_refs["hm_divider"].append_axes(
            "top", size="3%", pad=0.1
        )
        cbar_ax = fig_refs["hm_divider"].append_axes(
            "bottom",
            size="2.5%",
            pad=0.6
        )
        fig_refs["hm_bottompad"] = fig_refs["hm_divider"].append_axes(
            "bottom", size="6%", pad=0.1
        )
        fig_refs["hm_toppad"].axis("off")
        fig_refs["hm_bottompad"].axis("off")

    fig_refs["hm_cbar"] = ax.figure.colorbar(
        fig_refs["hm_im"], cax=cbar_ax, orientation="horizontal", fraction=1.0
    )

    if cbar_label is not None:
        fig_refs["hm_cbar"].set_label(cbar_label)

    if x_labels is not None:
        ax.set_xticks(
            np.arange(len(x_labels)), labels=x_labels, rotation=90, ha="center"
        )

    ax.set_yticks([], labels=[])

    return fig_refs
