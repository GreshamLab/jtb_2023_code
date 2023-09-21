import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from inferelator_velocity.utils.aggregation import (
    aggregate_sliding_window_times
)

from jtb_2023_code.utils.model_prediction import (
    plot_gene,
)

from jtb_2023_code.figure_constants import (
    MAIN_FIGURE_DPI,
    FIG_DYNAMICAL_FILE_NAME,
    FIGURE_5_FILE_NAME
)

from jtb_2023_code.utils.figure_common import (
    align_ylim,
    velocity_axes,
    plot_correlations,
    symmetric_ylim,
    cluster_on_rows
)

from jtb_2023_code.utils.model_prediction import _to_dataloader


def _heatmap_data(data, var_names, times=None, lfc=True):
    
    _d = pd.DataFrame(data, columns=var_names)
    
    if times is not None:
        _d = _d.groupby(times).agg('mean')
    
    if lfc:
        _untreated = _d.iloc[0:10, :].mean(0)
        _d = _d.divide(_untreated, axis=1)
        _d = np.log2(_d)
        _d_good = np.all(np.isfinite(_d), 0)
        _d = _d.loc[:, _d_good]
        
    return _d


def _heatmap_axes(ax, has_untreated=True, x_ticks=True):
    ax.set_yticks([], [])
    if has_untreated and x_ticks:
        ax.set_xticks([10, 40, 70], [0, 30, 60], size=8)
    elif x_ticks:
        ax.set_xticks([0, 30, 60], [0, 30, 60], size=8)
    else:
        ax.set_xticks([], [])


def plot_figure_5(model_data, velo_data, predicts, save=True):

    fig = plt.figure(figsize=(7, 4), dpi=MAIN_FIGURE_DPI)

    axd = {
        "schematic": fig.add_axes(
            [0, 0.5, 0.3, 0.475]
        ),
        "gene_1_expr": fig.add_axes(
            [0.35, 0.75, 0.10, 0.18]
        ),
        "gene_2_expr": fig.add_axes(
            [0.35, 0.55, 0.10, 0.18]
        ),
        "gene_1_velo": fig.add_axes(
            [0.52, 0.75, 0.10, 0.18]
        ),
        "gene_2_velo": fig.add_axes(
            [0.52, 0.55, 0.10, 0.18]
        ),
        "gene_1_comp": fig.add_axes(
            [0.64, 0.75, 0.10, 0.18]
        ),
        "gene_2_comp": fig.add_axes(
            [0.64, 0.55, 0.10, 0.18]
        ),
        "rp_decay": fig.add_axes(
            [0.8, 0.56, 0.18, 0.34]
        ),
        "expr_actual": fig.add_axes(
            [0.05, 0.11, 0.2, 0.27]
        ),
        "expr_actual_cat": fig.add_axes(
            [0.05, 0.08, 0.2, 0.03]
        ),
        "expr_predict": fig.add_axes(
            [0.27, 0.11, 0.2, 0.27]
        ),
        "expr_predict_cat": fig.add_axes(
            [0.27, 0.08, 0.2, 0.03]
        ),
        "transcription_predict": fig.add_axes(
            [0.49, 0.11, 0.2, 0.27]
        ),
        "transcription_predict_cat": fig.add_axes(
            [0.49, 0.08, 0.2, 0.03]
        ),
        "decay_predict": fig.add_axes(
            [0.71, 0.11, 0.2, 0.27]
        ),
        "decay_predict_cat": fig.add_axes(
            [0.71, 0.08, 0.2, 0.03]
        ),
        "cbar": fig.add_axes(
            [0.95, 0.14, 0.01, 0.21]
        )
    }

    axd["schematic"].set_title("A", loc="left", weight="bold", y=0.95)
    axd["gene_1_expr"].set_title("B", loc="left", weight="bold", x=-0.28)
    axd["gene_1_velo"].set_title("C", loc="left", weight="bold", x=-0.28)
    axd["gene_1_comp"].set_title("D", loc="left", weight="bold", x=-0.18)
    axd["rp_decay"].set_title("E", loc="left", weight="bold", x=-0.28, y=1.085)
    axd["expr_actual"].set_title("F", loc="left", weight="bold", x=-0.1)
    axd["expr_predict"].set_title("G", loc="left", weight="bold", x=-0.05)
    axd["transcription_predict"].set_title("H", loc="left", weight="bold", x=-0.05)
    axd["decay_predict"].set_title("I", loc="left", weight="bold", x=-0.05)

    axd["gene_2_expr"].set_ylabel("Transcript Counts (X)", y=1.05, labelpad=-0.75, size=8)
    axd["gene_2_velo"].set_ylabel("Transcript Velocity (dX/dt)", y=1.05, labelpad=-0.75, size=8)
    axd["gene_2_expr"].set_xlabel("Time (min)", size=8, labelpad=-0.5)
    axd["gene_2_velo"].set_xlabel("Time (min)", size=8, labelpad=34)
    axd["gene_2_comp"].set_xlabel("Time (min)", size=8, labelpad=34)

    axd["schematic"].imshow(
        plt.imread(FIG_DYNAMICAL_FILE_NAME),
        aspect="equal"
    )
    axd["schematic"].axis("off")

    rgen = np.random.default_rng(441)
    for i, g in enumerate(["YKR039W", "YOR063W"]):
        plot_gene(
            model_data,
            g,
            axd[f"gene_{i + 1}_expr"],
            rgen,
            velocity=False,
            test_only=True,
            annotation_loc=(0.6, 0.8),
        )
        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_expr"],
            rgen,
            layer="biophysical_predict_counts",
            predicts=True,
            time_positive_only=True,
            annotation_loc=None,
        )

        plot_gene(
            velo_data,
            g,
            axd[f"gene_{i + 1}_velo"],
            rgen,
            velocity=True,
            annotation_loc=(0.6, 0.8),
        )
        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_velo"],
            rgen,
            layer="biophysical_predict_velocity",
            predicts=True,
            time_positive_only=True,
            annotation_loc=None,
        )
        symmetric_ylim(axd[f"gene_{i + 1}_velo"])

        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_comp"],
            rgen,
            layer="biophysical_predict_transcription",
            velocity=True,
            predicts=True,
            time_positive_only=False,
            annotation_loc=(0.6, 0.8),
            predicts_color='cornflowerblue'
        )
        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_comp"],
            rgen,
            layer="biophysical_predict_decay",
            predicts=True,
            time_positive_only=False,
            annotation_loc=None,
            predicts_color="violet"
        )

        axd[f"gene_{i + 1}_comp"].annotate(
            "α",
            (0.22, 0.875),
            xycoords="axes fraction",
            color="cornflowerblue",
            size=7
        )

        axd[f"gene_{i + 1}_comp"].annotate(
            "-λX",
            (0.7, 0.1),
            xycoords="axes fraction",
            color="violet",
            size=7
        )

        align_ylim(axd[f"gene_{i + 1}_velo"], axd[f"gene_{i + 1}_comp"])
        velocity_axes(axd[f"gene_{i + 1}_comp"])
        axd[f"gene_{i + 1}_comp"].axvline(
            0, 0, 1, linestyle="--", linewidth=1.0, c="black"
        )
        axd[f"gene_{i + 1}_comp"].tick_params(labelleft=False) 

        if i == 1:
            axd[f"gene_{i + 1}_expr"].set_xticks(
                [0, 30, 60],
                [0, 30, 60],
                size=8
            )

    axd['gene_1_expr'].set_ylim(0, 8)
    axd['gene_2_expr'].set_ylim(0, 22)
    axd['gene_2_velo'].set_ylim(-1, 1)
    axd['gene_2_comp'].set_ylim(-1, 1)
    axd["rp_decay"].plot(
        np.arange(-10, 60) + 0.5,
        np.log(2)
        / aggregate_sliding_window_times(
            predicts.layers["biophysical_predict_decay_constant"][
                :, model_data.var["RP"]
            ],
            predicts.obs["program_rapa_time"],
            width=1,
            centers=np.arange(-10, 60) + 0.5,
            agg_kwargs={"axis": 0},
        )[0]
        * -1,
        color="black",
        alpha=0.05,
    )

    axd["rp_decay"].plot(
        np.arange(-10, 60) + 0.5,
        np.log(2)
        / aggregate_sliding_window_times(
            predicts.layers["biophysical_predict_decay_constant"][
                :, model_data.var_names == "YOR063W"
            ],
            predicts.obs["program_rapa_time"],
            width=1,
            centers=np.arange(-10, 60) + 0.5,
            agg_kwargs={"axis": 0},
        )[0]
        * -1,
        color="crimson"
    )
    axd["rp_decay"].annotate(
        "RPL3",
        (0.75, 0.33),
        xycoords="axes fraction",
        color="crimson",
        size=8
    )
    axd["rp_decay"].set_ylim(0, 60)
    axd["rp_decay"].set_ylabel("Half-life (min)", labelpad=-1, size=8)
    axd["rp_decay"].set_title("Ribosomal Protein (RP)\nTranscript Stability", size=8)
    axd["rp_decay"].set_xticks([0, 30, 60], [0, 30, 60], size=8)
    axd["rp_decay"].set_xlabel("Time (min)", size=8)
    axd["rp_decay"].axvline(0, 0, 1, linestyle="--", linewidth=1.0, c="black")
    
    
    obs = _heatmap_data(
        _to_dataloader(model_data, layer="X", untreated_only=False).mean(0),
        predicts.var_names
    )
    preds =  _heatmap_data(
        predicts.layers['biophysical_predict_counts'],
        predicts.var_names,
        times=predicts.obs['program_rapa_time'].values,
    )
    preds_up = _heatmap_data(
        predicts.layers['biophysical_predict_transcription'],
        predicts.var_names,
        times=predicts.obs['program_rapa_time'].values,
        lfc=True
    )
    preds_down = _heatmap_data(
        np.log(2) / predicts.layers['biophysical_predict_decay_constant'],
        predicts.var_names,
        times=predicts.obs['program_rapa_time'].values
    )
    order = cluster_on_rows(obs.T)
    order = order.intersection(preds.columns)

    _cpred = np.array([224, 172, 252]).astype(np.uint8)
    _cobs = np.array([155, 223, 250]).astype(np.uint8)

    axd['expr_actual'].pcolormesh(
        obs.loc[:, order].T,
        cmap='bwr',
        vmin=-5, vmax=5
    )
    axd['expr_actual'].axvline(
        10, 0, 1, linestyle="--", linewidth=1.0, c="black"
    )

    axd['expr_actual_cat'].pcolormesh(
        np.tile(_cobs, 70).T.reshape(1, 70, 3)
    )

    axd['expr_actual_cat'].annotate(
        "Observed",
        (0.5, 0.45),
        xycoords="axes fraction",
        size=8,
        ha='center',
        va='center'
    )

    _cbar_ref = axd['expr_predict'].pcolormesh(
        preds.loc[:, order].T,
        cmap='bwr',
        vmin=-5, vmax=5
    )
    axd['expr_predict'].axvline(
        10, 0, 1, linestyle="--", linewidth=1.0, c="black"
    )

    axd['expr_predict_cat'].pcolormesh(
        np.concatenate((
            np.tile(_cobs, 10),
            np.tile(_cpred, 60)
        )).T.reshape(1, 70, 3)
    )

    axd['expr_predict_cat'].annotate(
        "In.",
        (0.071, 0.45),
        xycoords="axes fraction",
        size=8,
        ha='center',
        va='center'
    )
    axd['expr_predict_cat'].annotate(
        "Prediction",
        (0.571, 0.45),
        xycoords="axes fraction",
        size=8,
        ha='center',
        va='center'
    )

    axd['transcription_predict'].pcolormesh(
        preds_up.loc[0:, order].T,
        cmap='bwr',
        vmin=-5, vmax=5
    )

    axd['transcription_predict_cat'].pcolormesh(
        np.tile(_cpred, 60).T.reshape(1, 60, 3)
    )
    axd['transcription_predict_cat'].annotate(
        "Prediction",
        (0.5, 0.45),
        xycoords="axes fraction",
        size=8,
        ha='center',
        va='center'
    )

    axd['decay_predict'].pcolormesh(
        preds_down.loc[0:, order].T,
        cmap='bwr',
        vmin=-5, vmax=5
    )
    axd['decay_predict_cat'].pcolormesh(
        np.tile(_cpred, 60).T.reshape(1, 60, 3)
    )
    axd['decay_predict_cat'].annotate(
        "Prediction",
        (0.5, 0.45),
        xycoords="axes fraction",
        size=8,
        ha='center',
        va='center'
    )
    plt.colorbar(_cbar_ref, axd['cbar'])
    axd['cbar'].tick_params(axis="both", labelsize=8)
    axd['cbar'].set_title("Log$_2$FC", size=8)

    axd['expr_actual'].set_title("Observed\nExpression", size=8)
    axd['expr_actual'].set_ylabel("Transcripts", size=8)
    axd['expr_predict'].set_title("Predicted\nExpression", size=8)
    axd['transcription_predict'].set_title("Predicted\nTranscription Rate", size=8)
    axd['decay_predict'].set_title("Predicted\nRNA Half-Life ", size=8)
    axd["expr_actual_cat"].set_xlabel("Time (min)", size=8, labelpad=-1)
    axd["expr_predict_cat"].set_xlabel("Time (min)", size=8, labelpad=-1)
    axd["transcription_predict_cat"].set_xlabel("Time (min)", size=8, labelpad=-1)
    axd["decay_predict_cat"].set_xlabel("Time (min)", size=8, labelpad=-1)

    _heatmap_axes(axd['expr_actual'], x_ticks=False)
    _heatmap_axes(axd['expr_actual_cat'])

    _heatmap_axes(axd['expr_predict'], x_ticks=False)
    _heatmap_axes(axd['expr_predict_cat'])

    _heatmap_axes(axd['transcription_predict'], has_untreated=False, x_ticks=False)
    _heatmap_axes(axd['transcription_predict_cat'], has_untreated=False)

    _heatmap_axes(axd['decay_predict'], has_untreated=False, x_ticks=False)
    _heatmap_axes(axd['decay_predict_cat'], has_untreated=False)

    if save:
        fig.savefig(FIGURE_5_FILE_NAME + ".png", facecolor="white")

    return fig
