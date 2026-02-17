import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import anndata as ad
import scipy.sparse as sps

from jtb_2023_code.utils.figure_common import (
    to_pool_colors,
    velocity_axes
)
from jtb_2023_code.utils.process_published_data import process_all_decay_links

from jtb_2023_code.figure_constants import (
    SUPPLEMENTAL_FIGURE_DPI,
    FIGURE_4_GENES,
    SFIG3B_FILE_NAME,
    FIGURE_5_SUPPLEMENTAL_FILE_NAME,
    SUPIRFACTOR_DECAY_MODEL,
    RAPA_VELO_LAYER
)
from jtb_2023_code.utils.figure_data import common_name

from inferelator_velocity.utils.aggregation import (
    aggregate_sliding_window_times
)
from inferelator_velocity.decay import calc_decay
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from jtb_2023_code.utils.model_result_loader import (
    load_model_results,
    plot_results,
    plot_losses,
    get_plot_idx,
)

from jtb_2023_code.utils.model_prediction import predict_from_model

from supirfactor_dynamical import read


def figure_5_supplement_1_plot(data, f3_data=None, save=True):
    if f3_data is None:
        f3_data = _get_fig5_data(data)

    time_key = f"program_{data.all_data.uns['programs']['rapa_program']}_time"

    def ticks_off(ax):
        ax.set_yticks([], [])
        ax.set_xticks([], [])

    fig_refs = {}

    fig = plt.figure(figsize=(4, 4), dpi=SUPPLEMENTAL_FIGURE_DPI)

    _small_w = 0.1
    _small_h = 0.125

    axd = {
        "image": fig.add_axes([0.15, 0.67, 0.77, 0.33]),
        "decay_1": fig.add_axes([0.2, 0.39, 0.32, 0.23]),
        "decay_2": fig.add_axes([0.60, 0.39, 0.32, 0.23]),
        "decay_1-1": fig.add_axes([0.2, 0.16, _small_w, _small_h]),
        "decay_1-2": fig.add_axes([0.31, 0.16, _small_w, _small_h]),
        "decay_1-3": fig.add_axes([0.42, 0.16, _small_w, _small_h]),
        "decay_1-4": fig.add_axes([0.2, 0.025, _small_w, _small_h]),
        "decay_1-5": fig.add_axes([0.31, 0.025, _small_w, _small_h]),
        "decay_1-6": fig.add_axes([0.42, 0.025, _small_w, _small_h]),
        "decay_2-1": fig.add_axes([0.6, 0.16, _small_w, _small_h]),
        "decay_2-2": fig.add_axes([0.71, 0.16, _small_w, _small_h]),
        "decay_2-3": fig.add_axes([0.82, 0.16, _small_w, _small_h]),
        "decay_2-4": fig.add_axes([0.6, 0.025, _small_w, _small_h]),
        "decay_2-5": fig.add_axes([0.71, 0.025, _small_w, _small_h]),
        "decay_2-6": fig.add_axes([0.82, 0.025, _small_w, _small_h]),
    }

    color_vec = to_pool_colors(f3_data.obs["Pool"])

    rgen = np.random.default_rng(8)
    overplot_shuffle = np.arange(f3_data.shape[0])
    rgen.shuffle(overplot_shuffle)

    for i, g in enumerate(FIGURE_4_GENES):
        fig_refs[f"decay_{i}"] = axd[f"decay_{i+1}"].scatter(
            x=f3_data.layers["denoised"][overplot_shuffle, i],
            y=f3_data.layers["velocity"][overplot_shuffle, i],
            c=color_vec[overplot_shuffle],
            alpha=0.2,
            s=1,
        )

        xlim = np.quantile(f3_data.layers["denoised"][:, i], 0.995)
        ylim = np.abs(
            np.quantile(f3_data.layers["velocity"][:, i], [0.001, 0.999])
        ).max()

        axd[f"decay_{i+1}"].set_xlim(0, xlim)
        axd[f"decay_{i+1}"].set_ylim(-1 * ylim, ylim)
        velocity_axes(axd[f"decay_{i+1}"])
        axd[f"decay_{i+1}"].tick_params(labelsize=8)

        if i == 0:
            axd[f"decay_{i+1}"].set_ylabel(
                "RNA Velocity\n(Counts/minute)", size=8
            )

        axd[f"decay_{i+1}"].set_xlabel(
            "Expression (Counts)", size=8, labelpad=20
        )
        axd[f"decay_{i+1}"].set_title(
            data.gene_common_name(g),
            size=8,
            fontdict={"fontweight": "bold", "fontstyle": "italic"},
        )

        dc = calc_decay(
            f3_data.layers["denoised"][:, i].reshape(-1, 1),
            f3_data.layers["velocity"][:, i].reshape(-1, 1),
            include_alpha=False,
            lstatus=False,
        )[0][0]

        axd[f"decay_{i+1}"].axline(
            (0, 0), slope=-1 * dc, color="darkred", linewidth=1, linestyle="--"
        )

        axd[f"decay_{i+1}"].annotate(
            r"$\lambda$" + f" = {dc:0.3f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            size=6,
        )

        for j in range(1, 7):
            if j == 1:
                _start = -10
            else:
                _start = 10 * (j - 1)

            _window_idx = f3_data.obs[time_key] > _start
            _window_idx &= f3_data.obs[time_key] <= (_start + 10)

            _window_adata = f3_data[_window_idx, :]
            _w_overplot = np.arange(_window_adata.shape[0])
            rgen.shuffle(_w_overplot)

            fig_refs[f"decay_{i}_{j}"] = axd[f"decay_{i+1}-{j}"].scatter(
                x=_window_adata.layers["denoised"][_w_overplot, i],
                y=_window_adata.layers["velocity"][_w_overplot, i],
                c=color_vec[_window_idx][_w_overplot],
                alpha=0.2,
                s=0.5,
            )

            dc = calc_decay(
                _window_adata.layers["denoised"][:, i].reshape(-1, 1),
                _window_adata.layers["velocity"][:, i].reshape(-1, 1),
                include_alpha=False,
                lstatus=False,
            )[0][0]

            axd[f"decay_{i+1}-{j}"].axline(
                (0, 0),
                slope=-1 * dc,
                color="darkred",
                linewidth=1,
                linestyle="--"
            )

            axd[f"decay_{i+1}-{j}"].annotate(
                r"$\lambda$" + f" = {dc:0.3f}",
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                size=6,
            )

            velocity_axes(axd[f"decay_{i+1}-{j}"])
            ticks_off(axd[f"decay_{i+1}-{j}"])

            axd[f"decay_{i+1}-{j}"].set_xlim(0, xlim)
            axd[f"decay_{i+1}-{j}"].set_ylim(-1 * ylim, ylim)

    axd["image"].imshow(plt.imread(SFIG3B_FILE_NAME), aspect="equal")
    axd["image"].axis("off")

    axd["image"].set_title("A", loc="left", weight="bold", x=-0.28, y=0.8)
    axd["decay_1"].set_title("B", loc="left", weight="bold", x=-0.4, y=0.9)
    axd["decay_1-1"].set_title("C", loc="left", weight="bold", x=-1.2, y=0.5)

    if save:
        fig.savefig(
            FIGURE_5_SUPPLEMENTAL_FILE_NAME + "_1.png",
            facecolor="white"
        )

    return fig


def figure_5_supplement_2_plot(data, save=True):

    pubbed = process_all_decay_links(data.all_data.var_names)

    _calc_decays = data.all_data.varm["rapamycin_window_decay"][
        :, 1:9
    ].mean(1)

    pubbed["This Work"] = np.log(2) / _calc_decays
    pubbed = pubbed.dropna()

    order = [
        "This Work",
        "Neymotin2014",
        "Chan2018",
        "Miller2011",
        "Munchel2011",
        "Geisberg2014",
    ]

    fig, axd = plt.subplots(6, 6, figsize=(6, 6), dpi=SUPPLEMENTAL_FIGURE_DPI)

    ax_hm = fig.add_axes([0.65, 0.65, 0.2, 0.2])
    ax_hm_cbar = fig.add_axes([0.875, 0.7, 0.02, 0.1])

    corr_mat = np.full((len(order), len(order)), np.nan, dtype=float)

    for i, dataset in enumerate(order):
        for j, dataset2 in enumerate(order[i:]):
            corr_mat[i + j, i] = spearmanr(
                pubbed[dataset2],
                pubbed[dataset]
            ).statistic

    for i, dataset in enumerate(order):
        axd[i, 0].set_ylabel(dataset, size=6)
        axd[-1, i].set_xlabel(dataset, size=6)

        for j, dataset2 in enumerate(order[i:]):
            axd[j + i, i].scatter(
                pubbed[dataset2],
                pubbed[dataset],
                s=1,
                alpha=0.1,
                color="black"
            )
            axd[j + i, i].set_xticks([], [])
            axd[j + i, i].set_yticks([], [])
            axd[j + i, i].set_xlim(0, np.quantile(pubbed[dataset2], 0.99))
            axd[j + i, i].set_ylim(0, np.quantile(pubbed[dataset], 0.99))
            model = LinearRegression(fit_intercept=False).fit(
                pubbed[dataset2].values.reshape(-1, 1),
                pubbed[dataset].values.reshape(-1, 1),
            )
            r2 = corr_mat[i + j, i]
            axd[j + i, i].axline(
                (0, 0),
                slope=model.coef_.ravel()[0],
                color="red",
                linestyle="--",
                linewidth=1
            )
            axd[j + i, i].annotate(
                f"ρ={r2:.2f}",
                (0.1, 0.75),
                xycoords="axes fraction",
                color="black",
                size=6,
                bbox=dict(
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="white",
                    boxstyle="round"
                ),
            )

        for k in range(i):
            axd[k, i].axis("off")

    for i, dataset in enumerate(order):
        for j, dataset2 in enumerate(order[i:]):
            if (i + j) == i:
                corr_mat[i + j, i] = np.nan

    hm_ref = ax_hm.pcolormesh(corr_mat[::-1, :], cmap="Reds", vmin=0, vmax=1)

    for i in range(len(order)):
        for j in range(len(order) - i):
            _val = corr_mat[::-1][i, j]
            if not np.isnan(_val):
                ax_hm.text(
                    j + 0.5,
                    i + 0.5,
                    f"{corr_mat[::-1][i, j]:.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=4,
                )

    ax_hm.set_xticks(np.arange(len(order)) + 0.5, order, rotation=90, size=6)

    ax_hm.set_yticks(np.arange(len(order)) + 0.5, order[::-1], size=6)

    axd[0, 0].set_title("A", loc="left", weight="bold", size=8, y=0.85, x=-0.4)
    ax_hm.set_title("B", loc="left", weight="bold", size=8)

    plt.colorbar(hm_ref, ax_hm_cbar)
    ax_hm_cbar.set_yticks([0, 1], [0, 1], size=8)
    ax_hm_cbar.tick_params(axis="y", length=2, pad=1)
    ax_hm_cbar.set_title("ρ", size=8)

    if save:
        fig.savefig(
            FIGURE_5_SUPPLEMENTAL_FILE_NAME + "_2.png",
            facecolor="white"
        )

    return fig


def figure_5_supplement_3_plot(
    data,
    model_data,
    model_scaler,
    save=True
):
    decay_model = read(SUPIRFACTOR_DECAY_MODEL).eval()

    decay_predicts = predict_from_model(
        decay_model,
        model_data,
        return_decay_constants=True,
        untreated_only=False
    )[1]

    decay_predicts = ad.AnnData(
        decay_predicts.reshape(-1, decay_predicts.shape[2]) * -1
    )
    decay_predicts.var_names = model_data.var_names.copy()
    decay_predicts.obs["program_rapa_time"] = np.tile(
        np.arange(-10, 60) + 0.5, int(decay_predicts.shape[0] / 70)
    )

    decay_velo_predicts = predict_from_model(
        decay_model,
        model_data,
        untreated_only=False,
        return_data_stacked=True
    )

    fig, axd = plt.subplots(
        3,
        4,
        figsize=(5, 5),
        dpi=SUPPLEMENTAL_FIGURE_DPI,
        gridspec_kw={"wspace": 0.5, "hspace": 0.7},
    )
    plt.subplots_adjust(top=0.9, bottom=0.12, left=0.15, right=0.95)

    def shift_axis(ax, x=None, y=None):
        _p = ax.get_position()

        if x is not None:
            _p.x0 += x
            _p.x1 += x
        if y is not None:
            _p.y0 += y
            _p.y1 += y

        ax.set_position(_p)

    for i in range(4):
        shift_axis(axd[1, i], y=-0.025)

    ax_lr = fig.add_axes([0.02, 0.64, 0.50, 0.35], zorder=-3)
    ax_wd = fig.add_axes([0.525, 0.64, 0.45, 0.35], zorder=-3)

    ax_lr.add_patch(patches.Rectangle((0, 0), 1, 1, color="lavender"))
    ax_lr.annotate(
        r"Learning Rate ($\gamma$)",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        size=10,
        weight="bold",
    )
    ax_lr.annotate(
        r"($\lambda$ = 1e-7)",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
        size=8,
        weight="bold",
    )
    ax_lr.axis("off")

    ax_wd.add_patch(patches.Rectangle((0, 0), 1, 1, color="mistyrose"))
    ax_wd.annotate(
        r"Weight Decay ($\lambda$)",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        size=10,
        weight="bold",
    )
    ax_wd.annotate(
        r"($\gamma$ = 5e-5)",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
        size=8,
        weight="bold",
    )
    ax_wd.axis("off")

    supirfactor_results, supirfactor_losses = load_model_results()

    for i, (param, metric, other_param, other_param_val, result) in enumerate(
        [
            ("Learning_Rate", "R2_validation", "Weight_Decay", 1e-7, True),
            (
                "Learning_Rate",
                list(map(str, range(1, 201))),
                "Weight_Decay",
                1e-7,
                False,
            ),
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

        if result:
            _result = supirfactor_results.loc[
                get_plot_idx(
                    supirfactor_results,
                    "decay",
                    other_param,
                    other_param_val,
                    time="rapa",
                    model_type="Decay",
                ),
                :,
            ].copy()
            _result = _result[_result["Decay_Model_Width"] == 50]

            plot_results(
                _result,
                metric,
                param,
                axd[0, i],
                ylim=(0, 0.3),
                yticks=(0, 0.1, 0.2, 0.3),
            )

        else:
            _loss = supirfactor_losses.loc[
                get_plot_idx(
                    supirfactor_losses,
                    "decay",
                    other_param,
                    other_param_val,
                    time="rapa",
                    model_type="Decay",
                ),
                :,
            ].copy()
            _loss = _loss[_loss["Decay_Model_Width"] == 50]

            plot_losses(_loss, metric, param, axd[0, i], ylim=(0.0, 10))

    axd[0, 0].set_title("A", loc="left", weight="bold", size=10, x=-0.6)
    axd[1, 0].set_title("B", loc="left", weight="bold", size=10, x=-0.6)
    axd[2, 0].set_title("C", loc="left", weight="bold", size=10, x=-0.6)

    for i, g in enumerate(["YKR039W", "YOR063W", "YDR224C", "YPR035W"]):
        _dc_idx = decay_predicts.var_names == g
        _dc = decay_predicts.X[:, _dc_idx].ravel()
        _dc /= model_scaler.scale_[_dc_idx]

        axd[1, i].plot(
            np.arange(-10, 60) + 0.5,
            np.log(2)
            / aggregate_sliding_window_times(
                _dc.reshape(-1, 1),
                decay_predicts.obs["program_rapa_time"],
                width=1,
                centers=np.arange(-10, 60) + 0.5,
            )[0],
            alpha=0.75,
            color="red",
        )

        axd[1, i].plot(
            np.arange(-10, 60) + 0.5,
            np.log(2)
            / aggregate_sliding_window_times(
                data.all_data.layers["decay_constants"][
                    (data.all_data.obs["Gene"] == "WT").values,
                    :
                ][
                    :,
                    data.all_data.var_names == g
                ],
                model_data.obs["program_rapa_time"],
                width=1,
                centers=np.arange(-10, 60) + 0.5,
            )[0],
            alpha=0.5,
            color="black",
        )

        axd[1, i].axvline(0, 0, 1, linestyle="--", linewidth=1.0, c="black")
        axd[1, i].set_ylim(0, 80)
        axd[1, i].set_xticks([0, 50], [0, 50])
        axd[1, i].set_title(common_name(g), style="italic", size=8)
        axd[1, i].tick_params(axis="both", which="major", labelsize=8)

        _transcription_rate = np.subtract(
            decay_velo_predicts[:, :, _dc_idx, 1],
            decay_velo_predicts[:, :, _dc_idx, 2],
        )

        _tr_data = aggregate_sliding_window_times(
            _transcription_rate.reshape(-1, 1),
            np.tile(np.arange(-10, 60), decay_velo_predicts.shape[0]),
            centers=np.arange(-10, 60) + 0.5,
            width=1.0,
        )[0]

        axd[2, i].scatter(
            np.tile(np.arange(-10, 60), decay_velo_predicts.shape[0]),
            _transcription_rate,
            color="gray",
            s=1,
            alpha=0.05,
        )

        axd[2, i].plot(
            np.arange(-10, 60) + 0.5,
            _tr_data,
            color="black",
            linestyle=":"
        )

        axd[2, i].tick_params(axis="both", which="major", labelsize=8)
        axd[2, i].set_xticks([0, 30, 60], [0, 30, 60])

        _ylim = np.quantile(_transcription_rate, 0.95)
        axd[2, i].set_ylim(-1 * _ylim, _ylim)
        axd[2, i].set_yticks([-1, 0, 1])

        velocity_axes(axd[2, i])

    axd[1, 0].set_ylabel("Half-life [min]", size=8)
    axd[2, 0].set_ylabel("Transcription Rate\n[counts/min]", size=8)

    if save:
        fig.savefig(
            FIGURE_5_SUPPLEMENTAL_FILE_NAME + "_3.png",
            facecolor="white"
        )

    return fig


def _get_fig5_data(data_obj, genes=None):
    if genes is None:
        genes = FIGURE_4_GENES

    _var_idx = data_obj.all_data.var_names.get_indexer(genes)

    fig3_data = data_obj.all_data.X[:, _var_idx]

    try:
        fig3_data = fig3_data.A
    except AttributeError:
        pass

    fig3_data = ad.AnnData(
        fig3_data,
        obs=data_obj.all_data.obs[
            ["Pool", "Experiment", "Gene", "program_0_time", "program_1_time"]
        ],
        dtype=np.float32,
    )

    fig3_data.var_names = FIGURE_4_GENES

    fig3_data.uns["window_times"] = data_obj.all_data.uns[
        "rapamycin_window_decay"
    ]["times"]

    fig3_data.layers["velocity"] = np.full(
        fig3_data.X.shape,
        np.nan,
        dtype=np.float32
    )
    fig3_data.layers["denoised"] = np.full(
        fig3_data.X.shape,
        np.nan,
        dtype=np.float32
    )
    fig3_data.varm["decay"] = data_obj.all_data.varm["rapamycin_window_decay"][
        _var_idx, :
    ]

    for k in data_obj.expts:
        _idx = data_obj._all_data_expt_index(*k)

        _vdata = data_obj.decay_data(*k)

        fig3_data.layers["velocity"][_idx, :] = _vdata.layers[RAPA_VELO_LAYER][
            :, _var_idx
        ]

        if k[1] == "WT":
            fig3_data.varm[f"decay_{k[0]}"] = _vdata.varm[
                "rapamycin_window_decay"
            ][_var_idx, :]

        del _vdata
        _dd = data_obj.denoised_data(*k).X[
            :, _var_idx
        ]
        fig3_data.layers["denoised"][_idx, :] = _dd.toarray() if sps.issparse(_dd) else _dd

    fig3_data = fig3_data[fig3_data.obs["Gene"] == "WT", :].copy()

    return fig3_data
