import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import scanpy as sc

from jtb_2022_code.utils.figure_common import (
    pool_palette,
    to_pool_colors
)
from jtb_2022_code.figure_constants import (
    MAIN_FIGURE_DPI,
    SUPPLEMENTAL_FIGURE_DPI,
    FIG_RAPA_LEGEND_VERTICAL_FILE_NAME,
    FIGURE_4_SUPPLEMENTAL_FILE_NAME,
    SFIG3A_FILE_NAME
)

from matplotlib import collections
from sklearn.linear_model import LinearRegression

from jtb_2022_code.utils.model_result_loader import (
    load_model_results,
    plot_results,
    plot_losses,
    get_plot_idx,
)


def figure_4_supplement_1_plot(data, save=True):
    fig_refs = {}

    fig = plt.figure(figsize=(6, 4), dpi=MAIN_FIGURE_DPI)

    _halfwidth = 0.12
    _height = 0.22

    _x_left = 0.12
    _x_right = 0.55

    axd = {
        "exp_var_1": fig.add_axes(
            [_x_left + 0.05, 0.72, 0.2, 0.2]
        ),
        "exp_var_2": fig.add_axes(
            [0.6, 0.72, 0.2, 0.2]
        ),
        "pc12_1": fig.add_axes(
            [_x_left, 0.36, _halfwidth, _height]
        ),
        "denoised_pc12_1": fig.add_axes(
            [_x_left + _halfwidth + 0.05, 0.36, _halfwidth, _height]
        ),
        "pc12_2": fig.add_axes(
            [_x_right, 0.36, _halfwidth, _height]
        ),
        "denoised_pc12_2": fig.add_axes(
            [_x_right + _halfwidth + 0.05, 0.36, _halfwidth, _height]
        ),
        "pc13_1": fig.add_axes(
            [_x_left, 0.08, _halfwidth, _height]
        ),
        "denoised_pc13_1": fig.add_axes(
            [_x_left + _halfwidth + 0.05, 0.08, _halfwidth, _height]
        ),
        "pc13_2": fig.add_axes(
            [_x_right, 0.08, _halfwidth, _height]
        ),
        "denoised_pc13_2": fig.add_axes(
            [_x_right + _halfwidth + 0.05, 0.08, _halfwidth, _height]
        ),
        "legend": fig.add_axes(
            [0.9, 0.1, 0.1, 0.55]
        ),
    }

    for i in range(1, 3):
        expt_dd = data.denoised_data(i, "WT")
        expt_dd.obsm["X_pca"] = expt_dd.obsm["denoised_pca"]
        _n_pcs = 100

        rgen = np.random.default_rng(441)
        overplot_shuffle = np.arange(expt_dd.shape[0])
        rgen.shuffle(overplot_shuffle)

        _eref = data.expt_data[(i, "WT")]
        fig_refs[f"exp_var_{i}"] = axd[f"exp_var_{i}"].plot(
            np.arange(0, _n_pcs + 1),
            np.insert(
                np.cumsum(_eref.uns["pca"]["variance_ratio"][:_n_pcs]),
                0, 0
            ),
            c="black",
            alpha=1,
        )

        fig_refs[f"exp_var_{i}"] = axd[f"exp_var_{i}"].plot(
            np.arange(0, _n_pcs + 1),
            np.insert(
                np.cumsum(expt_dd.uns["pca"]["variance_ratio"][:_n_pcs]),
                0, 0
            ),
            c="black",
            linestyle="--",
            alpha=1,
        )

        axd[f"exp_var_{i}"].set_title(f"Replicate {i}", size=8)
        axd[f"exp_var_{i}"].set_ylim(0, 1)
        axd[f"exp_var_{i}"].set_xlim(0, 100)
        axd[f"exp_var_{i}"].set_xlabel("# PCs", size=8)
        axd[f"exp_var_{i}"].set_ylabel("Cum. Var. Expl.", size=8)
        axd[f"exp_var_{i}"].tick_params(labelsize=8)

        for k in range(2, 4):
            comp_str = str(1) + "," + str(k)
            sc.pl.pca(
                data.expt_data[(i, "WT")],
                ax=axd[f"pc1{k}_{i}"],
                components=comp_str,
                color="Pool",
                palette=pool_palette(),
                title=None,
                show=False,
                alpha=0.25,
                size=2,
                legend_loc="none",
                annotate_var_explained=True,
            )

            axd[f"pc1{k}_{i}"].set_title(None)
            axd[f"pc1{k}_{i}"].xaxis.label.set_size(8)
            axd[f"pc1{k}_{i}"].yaxis.label.set_size(8)

            sc.pl.pca(
                expt_dd,
                ax=axd[f"denoised_pc1{k}_{i}"],
                components=comp_str,
                color="Pool",
                palette=pool_palette(),
                title=None,
                show=False,
                alpha=0.25,
                size=2,
                legend_loc="none",
                annotate_var_explained=True,
            )

            axd[f"denoised_pc1{k}_{i}"].set_title(None)
            axd[f"denoised_pc1{k}_{i}"].xaxis.label.set_size(8)
            axd[f"denoised_pc1{k}_{i}"].yaxis.label.set_size(8)

            if k == 2:
                axd[f"pc1{k}_{i}"].set_title("Counts", size=8)
                axd[f"denoised_pc1{k}_{i}"].set_title("Denoised", size=8)

    axd["legend"].imshow(
        plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME),
        aspect="equal"
    )
    axd["legend"].axis("off")

    axd["exp_var_1"].set_title("A", loc="left", x=-0.6, y=0.9, weight="bold")
    axd["pc12_1"].set_title("B", loc="left", x=-0.4, y=0.9, weight="bold")

    if save:
        fig.savefig(
            FIGURE_4_SUPPLEMENTAL_FILE_NAME + "_1.png",
            facecolor="white"
        )

    return fig


def figure_4_supplement_2_plot(data, save=True):
    # GET DATA #
    expt2 = data.expt_data[(2, "WT")]
    expt2_denoised_X = data.denoised_data(2, "WT").X

    # MAKE FIGURE #
    fig = plt.figure(figsize=(5, 8), dpi=SUPPLEMENTAL_FIGURE_DPI)

    axd = {
        "image": fig.add_axes([0.075, 0.725, 0.85, 0.225]),
        "pca_1": fig.add_axes([0.075, 0.45, 0.35, 0.25]),
        "velo_1": fig.add_axes([0.60, 0.45, 0.35, 0.25]),
        "pca_2": fig.add_axes([0.075, 0.1, 0.35, 0.25]),
        "velo_2": fig.add_axes([0.60, 0.1, 0.35, 0.25]),
    }

    # PLOT ONE EXAMPLE #
    _rapa = data.all_data.uns['programs']['rapa_program']
    _plot_velocity_calc(
        expt2,
        expt2_denoised_X,
        expt2.obs.index.get_loc(
            expt2.obs.sample(1, random_state=100).index[0]
        ),
        pca_ax=axd["pca_1"],
        expr_ax=axd["velo_1"],
        gene="YOR063W",
        gene_name=data.gene_common_name("YOR063W"),
        time_obs_key=f"program_{_rapa}_time",
    )

    # PLOT ANOTHER EXAMPLE #
    _plot_velocity_calc(
        expt2,
        expt2_denoised_X,
        expt2.obs.index.get_loc(
            expt2.obs.sample(1, random_state=101).index[0]
        ),
        pca_ax=axd["pca_2"],
        expr_ax=axd["velo_2"],
        gene="YKR039W",
        gene_name=data.gene_common_name("YKR039W"),
        time_obs_key=f"program_{_rapa}_time",
    )

    axd["pca_1"].set_title(r"$\bf{B}$" + "  i", loc="left", x=-0.2)
    axd["pca_2"].set_title(r"$\bf{C}$" + "  i", loc="left", x=-0.2)
    axd["velo_1"].set_title("ii", loc="left", x=-0.2)
    axd["velo_2"].set_title("ii", loc="left", x=-0.2)

    # DRAW SCHEMATIC #
    axd["image"].set_title(r"$\bf{A}$", loc="left", x=-0.075)
    axd["image"].imshow(plt.imread(SFIG3A_FILE_NAME), aspect="equal")
    axd["image"].axis("off")

    if save:
        fig.savefig(
            FIGURE_4_SUPPLEMENTAL_FILE_NAME + "_2.png",
            facecolor="white"
        )

    return fig


def figure_4_supplement_3_plot(save=True):
    supirfactor_results, supirfactor_losses = load_model_results()

    fig, axd = plt.subplots(
        4,
        6,
        figsize=(8, 8),
        dpi=MAIN_FIGURE_DPI,
        gridspec_kw={"wspace": 0.5, "hspace": 0.6},
    )
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95)

    for i, (param, metric, other_param, other_param_val, result) in enumerate(
        [
            ("Learning_Rate", "AUPR", "Weight_Decay", 1e-7, True),
            ("Learning_Rate", "R2_validation", "Weight_Decay", 1e-7, True),
            (
                "Learning_Rate",
                list(map(str, range(1, 201))),
                "Weight_Decay",
                1e-7,
                False,
            ),
            ("Weight_Decay", "AUPR", "Learning_Rate", 5e-5, True),
            ("Weight_Decay", "R2_validation", "Learning_Rate", 5e-5, True),
            (
                "Weight_Decay",
                list(map(str, range(1, 201))),
                "Learning_Rate",
                5e-5,
                False,
            ),
        ]
    ):
        for j, model in enumerate(
            ["static_meta", "rnn", "rnn_predictive", "tuned"]
        ):

            if result:
                plot_results(
                    supirfactor_results.loc[
                        get_plot_idx(
                            supirfactor_results,
                            model,
                            other_param,
                            other_param_val,
                            time="rapa",
                            model_type="Velocity",
                        ),
                        :,
                    ].copy(),
                    metric,
                    param,
                    axd[j, i],
                )

            else:
                plot_losses(
                    supirfactor_losses.loc[
                        get_plot_idx(
                            supirfactor_losses,
                            model,
                            other_param,
                            other_param_val,
                            time="rapa",
                            model_type="Velocity",
                        ),
                        :,
                    ].copy(),
                    metric,
                    param,
                    axd[j, i],
                    ylim=(0.0, 2.0),
                    tuned=model == "tuned",
                )

            if i == 0:
                axd[j, i].set_title(
                    chr(65 + j), loc="left", weight="bold", size=10, x=-0.25
                )

    ax_lr = fig.add_axes([0.02, 0.05, 0.48, 0.93], zorder=-3)
    ax_wd = fig.add_axes([0.51, 0.05, 0.45, 0.93], zorder=-3)

    ax_lr.add_patch(patches.Rectangle((0, 0), 1, 1, color="lavender"))
    ax_lr.annotate(
        "Learning Rate ($\gamma$)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        size=10,
        weight="bold",
    )
    ax_lr.annotate(
        "($\lambda$ = 1e-7)",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        size=8,
        weight="bold",
    )
    ax_lr.annotate(
        "Static",
        xy=(0.001, 0.82),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.annotate(
        "Dynamical",
        xy=(0.001, 0.55),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.annotate(
        "Predictive",
        xy=(0.001, 0.31),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.annotate(
        "Tuned",
        xy=(0.001, 0.1),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.axis("off")

    ax_wd.add_patch(patches.Rectangle((0, 0), 1, 1, color="mistyrose"))
    ax_wd.annotate(
        "Weight Decay ($\lambda$)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        size=10,
        weight="bold",
    )
    ax_wd.annotate(
        "($\gamma$ = 5e-5)",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        size=8,
        weight="bold",
    )
    ax_wd.axis("off")

    if save:
        fig.savefig(
            FIGURE_4_SUPPLEMENTAL_FILE_NAME + "_3.png",
            facecolor="white"
        )

    return fig


def figure_4_supplement_4_plot(save=True):
    supirfactor_results, supirfactor_losses = load_model_results()

    fig, axd = plt.subplots(
        4,
        6,
        figsize=(8, 8),
        dpi=MAIN_FIGURE_DPI,
        gridspec_kw={"wspace": 0.5, "hspace": 0.6},
    )
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95)

    model_labels = {
        "static_meta": "Static",
        "rnn": "Dynamical",
        "rnn_predictive": "Predictive",
        "tuned": "Tuned",
    }

    for i, (param, metric, other_param, other_param_val, result) in enumerate(
        [
            ("Learning_Rate", "AUPR", "Weight_Decay", 1e-7, True),
            ("Learning_Rate", "R2_validation", "Weight_Decay", 1e-7, True),
            (
                "Learning_Rate",
                list(map(str, range(1, 201))),
                "Weight_Decay",
                1e-7,
                False,
            ),
            ("Weight_Decay", "AUPR", "Learning_Rate", 5e-5, True),
            ("Weight_Decay", "R2_validation", "Learning_Rate", 5e-5, True),
            (
                "Weight_Decay",
                list(map(str, range(1, 201))),
                "Learning_Rate",
                5e-5,
                False,
            ),
        ]
    ):
        for j, model in enumerate(["rnn", "rnn_predictive", "tuned"]):

            if result:
                plot_results(
                    supirfactor_results.loc[
                        get_plot_idx(
                            supirfactor_results,
                            model,
                            other_param,
                            other_param_val,
                            time="cc",
                            model_type="Velocity",
                        ),
                        :,
                    ].copy(),
                    metric,
                    param,
                    axd[j, i],
                )

            else:
                plot_losses(
                    supirfactor_losses.loc[
                        get_plot_idx(
                            supirfactor_losses,
                            model,
                            other_param,
                            other_param_val,
                            time="cc",
                            model_type="Velocity",
                        ),
                        :,
                    ].copy(),
                    metric,
                    param,
                    axd[j, i],
                    ylim=(0.0, 2.0),
                    tuned=model == "tuned",
                )

            if i == 0:
                axd[j, i].set_title(
                    chr(65 + j), loc="left", weight="bold", size=10, x=-0.25
                )

    axd[3, 0].set_title("D", loc="left", weight="bold", size=10, x=-0.25)
    for i, (param, metric, other_param, other_param_val, result) in enumerate(
        [
            ("Learning_Rate", "AUPR", "Weight_Decay", 1e-7, True),
            ("Weight_Decay", "AUPR", "Learning_Rate", 5e-5, True),
        ]
    ):
        for j, model in enumerate(["rnn", "rnn_predictive", "tuned"]):

            plot_results(
                supirfactor_results.loc[
                    get_plot_idx(
                        supirfactor_results,
                        model,
                        other_param,
                        other_param_val,
                        time="combined",
                        model_type="Velocity",
                    ),
                    :,
                ].copy(),
                metric,
                param,
                axd[3, j + i * 3],
            )

            axd[3, j + i * 3].set_title(
                model_labels[model], weight="bold", size=8
            )

    ax_lr = fig.add_axes([0.02, 0.05, 0.48, 0.93], zorder=-3)
    ax_wd = fig.add_axes([0.51, 0.05, 0.45, 0.93], zorder=-3)
    ax_comb = fig.add_axes([0.01, 0.04, 0.96, 0.23], zorder=-2)

    ax_comb.add_patch(
        patches.Rectangle((0, 0), 1, 1, color="slategray", alpha=0.4)
    )
    ax_comb.axis("off")
    ax_comb.annotate(
        "Combined AUPR",
        xy=(0.011, 0.22),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )

    ax_lr.add_patch(patches.Rectangle((0, 0), 1, 1, color="lavender"))
    ax_lr.annotate(
        "Learning Rate ($\gamma$)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        size=10,
        weight="bold",
    )
    ax_lr.annotate(
        "($\lambda$ = 1e-7)",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        size=8,
        weight="bold",
    )
    ax_lr.annotate(
        "Dynamical",
        xy=(0.001, 0.79),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.annotate(
        "Predictive",
        xy=(0.001, 0.56),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.annotate(
        "Tuned",
        xy=(0.001, 0.34),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.axis("off")

    ax_wd.add_patch(patches.Rectangle((0, 0), 1, 1, color="mistyrose"))
    ax_wd.annotate(
        "Weight Decay ($\lambda$)",
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        size=10,
        weight="bold",
    )
    ax_wd.annotate(
        "($\gamma$ = 5e-5)",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        size=8,
        weight="bold",
    )
    ax_wd.axis("off")

    if save:
        fig.savefig(
            FIGURE_4_SUPPLEMENTAL_FILE_NAME + "_4.png",
            facecolor="white"
        )

    return fig


def _plot_velocity_calc(
    adata,
    expr_layer,
    center,
    pca_ax=None,
    expr_ax=None,
    time_obs_key="program_0_time",
    gene="YKR039W",
    gene_name=None,
):
    if pca_ax is None or expr_ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(5, 3), dpi=MAIN_FIGURE_DPI)
        pca_ax = axs[0]
        expr_ax = axs[1]

    else:
        fig = None

    connecteds = adata.obsp[
        "noise2self_distance_graph"
    ][center, :].nonzero()[1]

    pca_ax.scatter(
        adata.obsm["X_pca"][:, 0],
        adata.obsm["X_pca"][:, 1],
        c=to_pool_colors(adata.obs["Pool"]),
        s=0.25,
        alpha=0.1,
        zorder=-1,
    )

    pca_ax.add_collection(
        collections.LineCollection(
            [
                (adata.obsm["X_pca"][center, 0:2], adata.obsm["X_pca"][n, 0:2])
                for n in connecteds
            ],
            colors="black",
            linewidths=0.5,
            alpha=0.25,
        )
    )

    pca_ax.scatter(
        adata.obsm["X_pca"][connecteds, 0],
        adata.obsm["X_pca"][connecteds, 1],
        s=0.25,
        alpha=0.5,
        c="black",
    )

    pca_ax.scatter(
        adata.obsm["X_pca"][center, 0],
        adata.obsm["X_pca"][center, 1],
        facecolors="none",
        edgecolors="r",
        linewidths=0.5,
        s=4,
        zorder=5,
    )

    pca_ax.set_xlabel(
        f"PC1 ({adata.uns['pca']['variance_ratio'][0] * 100:.2f}%)", size=8
    )
    pca_ax.set_xticks([], [])
    pca_ax.set_ylabel(
        f"PC2 ({adata.uns['pca']['variance_ratio'][1] * 100:.2f}%)", size=8
    )
    pca_ax.set_yticks([], [])

    pca_ax.annotate(
        f"t = {adata.obs[time_obs_key].iloc[center]:0.2f} min",
        (0.40, 0.05),
        xycoords="axes fraction",
        fontsize="medium",
        color="darkred",
    )

    gene_loc = adata.var_names.get_loc(gene)
    gene_data = expr_layer[:, gene_loc]

    try:
        gene_data = gene_data.A
    except AttributeError:
        pass

    delta_x = (
        adata.obs[time_obs_key].iloc[connecteds] -
        adata.obs[time_obs_key].iloc[center]
    )
    delta_y = gene_data[connecteds] - gene_data[center]

    lr = LinearRegression(fit_intercept=False).fit(
        delta_x.values.reshape(-1, 1), delta_y.reshape(-1, 1)
    )
    slope_dxdt = lr.coef_[0][0]

    expr_ax.scatter(
        delta_x,
        delta_y,
        c=to_pool_colors(adata.obs["Pool"].iloc[connecteds]),
        s=1,
        alpha=0.75,
    )

    expr_ax.scatter(
        0,
        0,
        c=to_pool_colors(adata.obs["Pool"].iloc[[center]]),
        s=20,
        edgecolors="r",
        linewidths=0.5,
    )

    if gene_name is None:
        gene_name = gene

    expr_ax.set_title(r"$\bf{" + gene_name + "}$: " + f"Cell {center}", size=8)
    expr_ax.set_xlabel("dt [min]", size=8)
    expr_ax.set_ylabel("dx [counts]", size=8)
    expr_ax.tick_params(labelsize=8)

    xlim = np.abs(delta_x).max()
    ylim = np.abs(delta_y).max()

    expr_ax.set_xlim(-1 * xlim, xlim)
    expr_ax.set_ylim(-1 * ylim, ylim)

    expr_ax.axline(
        (0, 0),
        slope=slope_dxdt,
        linestyle="--",
        linewidth=1.0,
        c="black"
    )
    expr_ax.annotate(
        f"{slope_dxdt:0.3f} [counts/min]",
        (0.20, 0.8),
        xycoords="axes fraction",
        fontsize="medium",
        color="black",
    )

    expr_ax.annotate(
        f"n = {len(connecteds)} cells",
        (0.40, 0.05),
        xycoords="axes fraction",
        fontsize="medium",
        color="darkred",
    )

    return fig
