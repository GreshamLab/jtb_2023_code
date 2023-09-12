import itertools
import gc
import sys

import pandas as pd
import scanpy as sc
import numpy as np
import anndata as ad
import statsmodels.api as sm

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

from jtb_2023_code.utils.figure_common import (
    pool_palette,
    cc_palette,
    plot_heatmap
)

from jtb_2023_code.figure_constants import (
    FIG_CC_LEGEND_VERTICAL_FILE_NAME,
    FIG_RAPA_LEGEND_VERTICAL_FILE_NAME,
    CC_LENGTH_DATA_FILE,
    FIGURE_2_SUPPLEMENTAL_FILE_NAME,
    SFIG2A_FILE_NAME,
    CC_COLS,
    N_PCS,
    MAIN_FIGURE_DPI
)

from inferelator_velocity.plotting.program_times import (
    program_time_summary
)
from inferelator_velocity.plotting.mcv_summary import (
    mcv_plot,
    cumulative_variance_plot
)

sys.setrecursionlimit(10000)


def figure_2_supplement_1_plot(data, save=True):
    # BUILD PLOT #
    fig_refs = {}

    layout = [
        ["pc12_1_cc", "pc12_1_t", ".", "pc12_2_cc", "pc12_2_t", "t_cbar"],
        ["pc13_1_cc", "pc13_1_t", ".", "pc13_2_cc", "pc13_2_t", "t_cbar"],
        ["pc14_1_cc", "pc14_1_t", ".", "pc14_2_cc", "pc14_2_t", "t_cbar"],
        ["pc23_1_cc", "pc23_1_t", ".", "pc23_2_cc", "pc23_2_t", "cc_cbar"],
        ["pc24_1_cc", "pc24_1_t", ".", "pc24_2_cc", "pc24_2_t", "cc_cbar"],
        ["pc34_1_cc", "pc34_1_t", ".", "pc34_2_cc", "pc34_2_t", "cc_cbar"],
    ]

    fig, axd = plt.subplot_mosaic(
        layout,
        gridspec_kw=dict(
            width_ratios=[1, 1, 0.1, 1, 1, 0.8],
            height_ratios=[1, 1, 1, 1, 1, 0.8],
            wspace=0,
            hspace=0.01,
        ),
        figsize=(6, 9),
        dpi=MAIN_FIGURE_DPI,
        constrained_layout=True,
    )

    for i in range(1, 3):
        for j, k in itertools.combinations(range(1, 5), 2):
            comp_str = str(j) + "," + str(k)
            for ak, c, palette in [
                ("_cc", "CC", cc_palette()),
                ("_t", "Pool", pool_palette()),
            ]:
                ax_key = "pc" + str(j) + str(k) + "_" + str(i) + ak
                fig_refs[ax_key] = sc.pl.pca(
                    data.expt_data[(i, "WT")],
                    ax=axd[ax_key],
                    components=comp_str,
                    color=c,
                    palette=palette,
                    title=None,
                    show=False,
                    alpha=0.25,
                    size=2,
                    legend_loc="none",
                    annotate_var_explained=True,
                )

                axd[ax_key].set_title("")
                if ak == "_t":
                    axd[ax_key].set_ylabel("")

    axd["pc12_1_cc"].set_title("Rep. 1")
    axd["pc12_2_cc"].set_title("Rep. 2")
    axd["pc12_1_t"].set_title("Rep. 1")
    axd["pc12_2_t"].set_title("Rep. 2")

    axd["cc_cbar"].imshow(
        plt.imread(FIG_CC_LEGEND_VERTICAL_FILE_NAME),
        aspect="equal"
    )
    axd["cc_cbar"].axis("off")
    axd["t_cbar"].imshow(
        plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME),
        aspect="equal"
    )
    axd["t_cbar"].axis("off")

    if save:
        fig.savefig(
            FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_1.png",
            facecolor="white",
            bbox_inches="tight",
        )

    return fig


def figure_2_supplement_14_plot(save=True):
    growth_data = pd.read_csv(CC_LENGTH_DATA_FILE, sep="\t").melt(
        id_vars="Time", var_name="Replicate", value_name="Count"
    )
    growth_data["Count"] *= 1e6
    growth_data["Count"] = np.log2(growth_data["Count"])

    model = sm.OLS(
        growth_data["Count"],
        np.hstack(
            (
                growth_data["Time"].values.reshape(-1, 1),
                np.ones((growth_data.shape[0], 1)),
            )
        ),
    )
    results = model.fit()
    results.params["x1"]

    _lsq = results.params["x1"], results.params["const"], results.bse["x1"]
    _se = 1 / (_lsq[0] - _lsq[2]) - 1 / (_lsq[0])

    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=MAIN_FIGURE_DPI)
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.25, right=0.95)

    ax.plot(
        np.arange(0, growth_data["Time"].max()),
        np.arange(0, growth_data["Time"].max()) * _lsq[0] + _lsq[1],
        linestyle=":",
        color="red",
    )

    ax.set_title("Doubling Time (YPD)", size=8)

    ax.annotate(
        f"$t_d$ = {1 / _lsq[0]:.2f} ± {_se:.2f}\nn = 6",
        (0.25, 0.12),
        xycoords="axes fraction",
        size=8,
    )
    ax.set_ylim(np.log2(2e6), np.log2(4e7))
    ax.set_yticks(
        [np.log2(2e6), np.log2(5e6), np.log2(1e7), np.log2(2e7)],
        ["2e6", "5e6", "1e7", "2e7"],
    )
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xlabel("Time [min]", size=8)
    ax.set_ylabel("Conc. [cells / mL]", size=8)

    for gd, marker in zip(
        growth_data.groupby("Replicate"), ["o", "^", "<", "s", "X", "D"]
    ):
        ax.scatter(
            gd[1]["Time"],
            gd[1]["Count"],
            color="black",
            marker=marker,
            s=2
        )

    if save:
        fig.savefig(
            FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_2.png",
            facecolor="white"
        )


def figure_2_supplement_2_plot(data, save=True):
    # BUILD PLOT #

    layout = [
        [".", ".", ".", "title_all", "title_all"],
        ["schema", "schema", ".", "mcv_all_0", "cum_var_all_0"],
        [".", ".", ".", ".", "."],
        ["title_0_1", "title_0_1", ".", "title_1_1", "title_1_1"],
        ["mcv_0_1", "cum_var_0_1", ".", "mcv_1_1", "cum_var_1_1"],
        [".", ".", ".", ".", "."],
        ["title_0_2", "title_0_2", ".", "title_1_2", "title_1_2"],
        ["mcv_0_2", "cum_var_0_2", ".", "mcv_1_2", "cum_var_1_2"],
        [".", ".", ".", ".", "."],
        ["title_0_3", "title_0_3", ".", "title_1_3", "title_1_3"],
        ["mcv_0_3", "cum_var_0_3", ".", "mcv_1_3", "cum_var_1_3"],
        [".", ".", ".", ".", "."],
        ["title_0_4", "title_0_4", ".", "title_1_4", "title_1_4"],
        ["mcv_0_4", "cum_var_0_4", ".", "mcv_1_4", "cum_var_1_4"],
        ["xaxis_lab_0", "xaxis_lab_0", ".", "xaxis_lab_1", "xaxis_lab_1"],
    ]

    panel_labels = {
        "schema": "A",
        "title_all": "B",
        "title_0_1": "C",
        "title_0_2": "D",
        "title_0_3": "E",
        "title_0_4": "F",
    }

    pad_height = 0.5

    fig, axd = plt.subplot_mosaic(
        layout,
        gridspec_kw=dict(
            width_ratios=[1, 1, 0.15, 1, 1],
            height_ratios=[
                0.005, 1, pad_height,
                0.005, 1, pad_height,
                0.005, 1, pad_height,
                0.005, 1, pad_height,
                0.005, 1, 0.01,
            ],
            wspace=1.5,
            hspace=0.25,
        ),
        figsize=(6, 9),
        dpi=MAIN_FIGURE_DPI,
    )

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc="left", weight="bold", x=-0.35)

    axd["schema"].imshow(plt.imread(SFIG2A_FILE_NAME), aspect="equal")
    axd["schema"].axis("off")

    axd["xaxis_lab_0"].set_title("# Comps", fontsize=10, y=-50)
    axd["xaxis_lab_0"].axis("off")

    axd["xaxis_lab_1"].set_title("# Comps", fontsize=10, y=-50)
    axd["xaxis_lab_1"].axis("off")

    for i, expt in enumerate(["all"] + data.expts):
        d = data.all_data if expt == "all" else data.expt_data[expt]

        if i == 0:
            mcv_plot(d, ax=axd["mcv_all_0"], add_labels=False)
            cumulative_variance_plot(
                d,
                ax=axd["cum_var_all_0"],
                add_labels=False
            )

            axd["title_all"].set_title("Combined Data")
            axd["title_all"].axis("off")
            axd["mcv_all_0"].set_ylabel("MSE")
            axd["cum_var_all_0"].set_ylabel("% Var.")

        else:
            for j, p in enumerate(["0", "1"]):
                mcv_plot(
                    d,
                    ax=axd[f"mcv_{j}_{i}"],
                    program=p,
                    add_labels=False
                )
                cumulative_variance_plot(
                    d,
                    ax=axd[f"cum_var_{j}_{i}"],
                    program=p,
                    add_labels=False
                )
                axd[f"mcv_{j}_{i}"].set_ylabel("MSE")
                axd[f"cum_var_{j}_{i}"].set_ylabel("% Var.")
                axd[f"title_{j}_{i}"].set_title(
                    f"{'Rapamycin' if p == '0' else 'Cell Cycle'} "
                    f"({expt[1]} [{expt[0]}]) MCV"
                )
                axd[f"title_{j}_{i}"].axis("off")

    if save:
        fig.savefig(
            FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_3.png",
            facecolor="white",
            bbox_inches="tight",
        )

    return fig


def figure_2_supplement_3_plot(adata, save=True):
    _ami_idx = np.array(
        dendrogram(
            linkage(
                squareform(adata.uns["programs"]["cosine_distance"])
            ),
            no_plot=True
        )["leaves"]
    )

    layout = [
        ["matrix_1", "matrix_1_cbar", ".", "matrix_2", "matrix_2_cbar"],
        ["matrix_3", "matrix_3_cbar", ".", "matrix_4", "matrix_4_cbar"],
    ]

    panel_labels = {
        "matrix_1": "A",
        "matrix_2": "B",
        "matrix_3": "C",
        "matrix_4": "D"
    }

    _pref = adata.uns["programs"]
    # Title, Metric, VMIN, VMAX, CBAR
    metrics = {
        "matrix_1": ("Cosine", "cosine", 0, 2, "magma_r"),
        "matrix_2": ("Information", "information", 0, 1, "magma_r"),
        "matrix_3": (
            "Euclidean",
            "euclidean",
            0,
            int(np.quantile(_pref["euclidean_distance"], 0.95)),
            "magma_r",
        ),
        "matrix_4": (
            "Manhattan",
            "manhattan",
            0,
            int(np.quantile(_pref["manhattan_distance"], 0.95)),
            "magma_r",
        ),
    }

    fig, axd = plt.subplot_mosaic(
        layout,
        gridspec_kw=dict(
            width_ratios=[1, 0.1, 0.2, 1, 0.1],
            height_ratios=[1, 1],
            wspace=0.05,
            hspace=0.25,
        ),
        figsize=(6, 6),
        dpi=MAIN_FIGURE_DPI,
    )

    for ax_ref, (title, metric, vmin, vmax, cbar_name) in metrics.items():
        plot_heatmap(
            fig,
            _pref[f"{metric}_distance"][_ami_idx, :][:, _ami_idx],
            cbar_name,
            axd[ax_ref],
            colorbar_ax=axd[ax_ref + "_cbar"],
            vmin=vmin,
            vmax=vmax,
        )

        axd[ax_ref].set_title(title)
        axd[ax_ref].set_xlabel("Genes")
        axd[ax_ref].set_ylabel("Genes")

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc="left", weight="bold")

    if save:
        fig.savefig(
            FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_4.png",
            facecolor="white"
        )

    return fig


def figure_2_supplement_5_12_plot(data, save=True):
    figs = []

    cc_program = data.all_data.uns["programs"]["cell_cycle_program"]
    rapa_program = data.all_data.uns["programs"]["rapa_program"]

    for i, j in zip(
        range(6, 14, 2),
        [(1, "WT"), (2, "WT"), (1, "fpr1"), (2, "fpr1")]
    ):
        _layout = [
            ["pca1", "M-G1 / G1", "G1 / S"],
            ["pca2", "S / G2", "G2 / M"],
            ["hist", "M / M-G1", "cbar"],
        ]

        fig, ax = plt.subplot_mosaic(
            _layout,
            gridspec_kw=dict(
                width_ratios=[1] * len(_layout[0]),
                height_ratios=[1, 1, 1],
                wspace=0.25,
                hspace=0.35,
            ),
            figsize=(6, 8),
            dpi=MAIN_FIGURE_DPI,
        )

        program_time_summary(
            data.expt_data[j],
            cc_program,
            cluster_order=CC_COLS,
            cluster_colors={k: v for k, v in zip(CC_COLS, cc_palette())},
            cbar_title="Phase",
            ax=ax,
            alpha=0.1 if j[1] == "WT" else 0.5,
        )

        fig.suptitle(f"Cell Cycle Program Replicate {j[0]} [{j[1]}]")

        if save:
            fig.savefig(
                FIGURE_2_SUPPLEMENTAL_FILE_NAME + f"_{i}.png",
                facecolor="white"
            )

        figs.append(fig)

        _layout = [
            ["pca1", "12 / 3", "3 / 4"],
            ["pca2", "4 / 5", "5 / 6"],
            ["hist", "6 / 7", "7 / 8"],
            ["cbar", "cbar", "cbar"],
        ]

        fig, ax = plt.subplot_mosaic(
            _layout,
            gridspec_kw=dict(
                width_ratios=[1] * len(_layout[0]),
                height_ratios=[1, 1, 1, 0.2],
                wspace=0.25,
                hspace=0.35,
            ),
            figsize=(6, 8),
            dpi=MAIN_FIGURE_DPI,
        )

        program_time_summary(
            data.expt_data[j],
            rapa_program,
            ax=ax,
            cluster_order=["12", "3", "4", "5", "6", "7", "8"],
            cluster_colors={
                k: v
                for k, v in zip(
                    ["12", "3", "4", "5", "6", "7", "8"], pool_palette()[1:]
                )
            },
            cbar_title="Time [Groups]",
            cbar_horizontal=True,
            time_limits=(-15, 70),
            alpha=0.1 if j[1] == "WT" else 0.5,
        )

        fig.suptitle(f"Rapamycin Response Program Replicate {j[0]} [{j[1]}]")

        if save:
            fig.savefig(
                FIGURE_2_SUPPLEMENTAL_FILE_NAME + f"_{i + 1}.png",
                facecolor="white"
            )

        figs.append(fig)

    return figs


def figure_2_supplement_13_plot(data, save=True):
    # Reprocess rho data for heatmaps
    # Select non-denoised
    ptr = data.pseudotime_rho.loc[:, (slice(None), False, slice(None))]

    # Transpose, select WT, and drop indices into columns
    ptr = ptr.T.loc[:, (slice(None), "WT")].reset_index().droplevel(1, axis=1)

    # Throw away PCA and pull string pcs_neighbors into integer columns
    ptr = ptr.loc[ptr["method"] != "pca", :]
    ptr[["pcs", "neighbors"]] = ptr["values"].str.split(
        "_", expand=True
    ).astype(int)

    neighbors = np.arange(15, 115, 10)

    def _overlay_rect(method, i, ax):
        _ideal_value = ptr.loc[
            ptr.loc[ptr["method"] == method, i].idxmax(), ["neighbors", "pcs"]
        ]

        y = np.where(_ideal_value["neighbors"] == neighbors)[0][0] - 0.5
        x = np.where(_ideal_value["pcs"] == N_PCS)[0][0] - 0.5

        return ax.add_patch(
            plt.Rectangle((x, y), 1, 1, fill=False, color="black", linewidth=1)
        )

    panel_labels = {
        "dpt_rho_1": "A",
        "cellrank_rho_1": "B",
        "monocle_rho_1": "C",
        "palantir_rho_1": "D",
    }

    panel_titles = {"dpt_rho_1": "Rep. 1", "dpt_rho_2": "Rep. 2"}

    layout = [
        ["dpt_rho_1", "dpt_rho_2", "."],
        ["cellrank_rho_1", "cellrank_rho_2", "cbar"],
        ["monocle_rho_1", "monocle_rho_2", "cbar"],
        ["palantir_rho_1", "palantir_rho_2", "."],
    ]

    fig_refs = {}

    fig, axd = plt.subplot_mosaic(
        layout,
        gridspec_kw=dict(
            width_ratios=[1, 1, 0.05],
            height_ratios=[1, 1, 1, 1],
            wspace=0.1,
            hspace=0.1,
        ),
        figsize=(6, 9),
        dpi=MAIN_FIGURE_DPI,
    )

    for pt, pt_key in [
        ("Diffusion PT", "dpt"),
        ("Cellrank PT", "cellrank"),
        ("Monocle PT", "monocle"),
        ("Palantir PT", "palantir"),
    ]:
        _bottom = pt_key == "palantir"
        for i in range(1, 3):
            _left = i == 1

            hm_data = (
                ptr.loc[ptr["method"] == pt_key]
                .pivot_table(index="neighbors", columns="pcs", values=i)
                .reindex(N_PCS, axis=1)
                .reindex(neighbors, axis=0)
            )

            ax_key = f"{pt_key}_rho_{i}"

            fig_refs[ax_key] = axd[ax_key].imshow(
                hm_data,
                vmin=0.75,
                vmax=1.0,
                cmap="plasma",
                aspect="auto",
                interpolation="nearest",
                alpha=0.75,
            )

            _overlay_rect(pt_key, i, axd[ax_key])

            if _left:
                axd[ax_key].set_yticks(
                    range(hm_data.shape[0]),
                    labels=hm_data.index
                )
                axd[ax_key].set_ylabel(pt + "\n # Neighbors")
            else:
                axd[ax_key].set_yticks([], labels=[])

            if _bottom:
                axd[ax_key].set_xticks(
                    range(hm_data.shape[1]),
                    labels=hm_data.columns,
                    rotation=90,
                    ha="center",
                )
                axd[ax_key].set_xlabel("PCs")
            else:
                axd[ax_key].set_xticks([], labels=[])

            for y in range(hm_data.shape[0]):
                for x in range(hm_data.shape[1]):
                    n = hm_data.iloc[y, x]
                    if np.isnan(n):
                        continue
                    axd[ax_key].text(
                        x,
                        y,
                        "%.2f" % n,
                        horizontalalignment="center",
                        verticalalignment="center",
                        size=4,
                    )

    for ax_key, title_str in panel_titles.items():
        axd[ax_key].set_title(title_str)

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc="left", weight="bold", x=-0.3, y=0.99)

    fig_refs["cbar"] = fig.colorbar(
        fig_refs["dpt_rho_1"],
        cax=axd["cbar"],
        orientation="vertical",
        aspect=60
    )
    fig_refs["cbar"].ax.set_title("ρ")

    if save:
        fig.savefig(
            FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_14.png",
            facecolor="white",
            bbox_inches="tight",
        )

    return fig


def figure_2_supplement_4_plot(data_obj, save=True):
    fig_refs = {}
    fig = plt.figure(figsize=(6, 6), dpi=MAIN_FIGURE_DPI)

    axd = {
        "rapa_heatmap": fig.add_axes([0.08, 0.45, 0.8, 0.5]),
        "rapa_cbar": fig.add_axes([0.9, 0.45, 0.02, 0.5]),
        "cc_heatmap": fig.add_axes([0.08, 0.08, 0.8, 0.26]),
        "cc_cbar": fig.add_axes([0.9, 0.08, 0.02, 0.26]),
    }

    _xticks = np.arange(-10, 70, 10)
    raw_data, tick_locations = _generate_heatmap_data(
        data_obj,
        data_obj.all_data.uns["programs"]["rapa_program"],
        count_threshold=0.1,
        obs_time_ticks=_xticks,
    )

    fig_refs["rapa_heatmap"] = axd["rapa_heatmap"].pcolormesh(
        raw_data, cmap="magma", vmin=0, vmax=np.floor(raw_data.max())
    )

    axd["rapa_heatmap"].set_xticks(tick_locations, labels=_xticks)
    axd["rapa_heatmap"].set_yticks([])
    axd["rapa_heatmap"].set_ylabel(
        "Rapamycin Response Genes",
        size=8
    )
    axd["rapa_heatmap"].set_xlabel(
        "Cells (Ordered by Rapamycin Response Time)",
        size=8
    )
    axd["rapa_heatmap"].spines["right"].set_visible(False)
    axd["rapa_heatmap"].spines["top"].set_visible(False)
    axd["rapa_heatmap"].spines["left"].set_visible(False)
    axd["rapa_heatmap"].spines["bottom"].set_visible(False)

    fig_refs["rapa_cbar"] = fig.colorbar(
        fig_refs["rapa_heatmap"],
        cax=axd["rapa_cbar"],
        orientation="vertical",
        ticks=[0, np.floor(raw_data.max())],
    )

    fig_refs["rapa_cbar"].set_label("$log_2$(Pseudocount)", labelpad=-1)

    del raw_data
    del tick_locations

    gc.collect()

    _xticks = np.arange(0, 80, 20).tolist() + [88]
    raw_data, tick_locations = _generate_heatmap_data(
        data_obj,
        data_obj.all_data.uns["programs"]["cell_cycle_program"],
        obs_time_ticks=_xticks,
        count_threshold=0.1,
    )

    fig_refs["cc_heatmap"] = axd["cc_heatmap"].pcolormesh(
        raw_data, cmap="magma", vmin=0, vmax=np.floor(raw_data.max())
    )

    axd["cc_heatmap"].set_xticks(tick_locations, labels=_xticks)
    axd["cc_heatmap"].set_yticks([])
    axd["cc_heatmap"].set_ylabel("Cell Cycle Response Genes", size=8)
    axd["cc_heatmap"].set_xlabel("Cells (Ordered by Cell Cycle Time)", size=8)
    axd["cc_heatmap"].spines["right"].set_visible(False)
    axd["cc_heatmap"].spines["top"].set_visible(False)
    axd["cc_heatmap"].spines["left"].set_visible(False)
    axd["cc_heatmap"].spines["bottom"].set_visible(False)

    fig_refs["cc_cbar"] = fig.colorbar(
        fig_refs["cc_heatmap"],
        cax=axd["cc_cbar"],
        orientation="vertical",
        ticks=[0, np.floor(raw_data.max())],
    )

    fig_refs["cc_cbar"].set_label("$log_2$(Pseudocount)", labelpad=-1)

    if save:
        fig.savefig(
            FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_5.png",
            facecolor="white"
        )

    return fig


def _generate_heatmap_data(
    data_obj, program, count_threshold=None, obs_time_ticks=None
):
    raw_data = ad.AnnData(
        data_obj.all_data.layers["counts"],
        dtype=float,
        var=data_obj.all_data.var
    )

    sc.pp.normalize_per_cell(raw_data)

    _program_idx = data_obj.all_data.var["programs"] == program

    if count_threshold is not None:
        _gene_means = data_obj.all_data.X.mean(axis=0)

        try:
            _gene_means = _gene_means.A1
        except AttributeError:
            pass

        _program_idx &= _gene_means > count_threshold

    sc.pp.log1p(raw_data, base=2)

    _gene_order_idx = dendrogram(
        linkage(
            squareform(
                data_obj.all_data.varp["cosine_distance"][_program_idx, :][
                    :, _program_idx
                ],
                checks=False,
            )
        ),
        no_plot=True,
    )["leaves"]

    _wt_idx = data_obj.all_data.obs["Gene"] == "WT"
    _rapa_time_key = f"program_{program}_time"

    _obs_order_idx = data_obj.all_data.obs.loc[
        _wt_idx,
        _rapa_time_key
    ].argsort()

    if obs_time_ticks is None:
        _obs_times = None
    else:
        _obs_times = data_obj.all_data.obs.loc[
            _wt_idx,
            _rapa_time_key
        ].sort_values()
        _obs_times = [
            np.abs(_obs_times - x).argmin()
            for x in obs_time_ticks
        ]

    raw_data = raw_data.X.astype(np.float32)[_wt_idx, :][:, _program_idx]
    raw_data = raw_data[_obs_order_idx, :][:, _gene_order_idx]
    raw_data = raw_data.T

    try:
        raw_data = raw_data.A
    except AttributeError:
        pass

    return raw_data, _obs_times
