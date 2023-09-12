import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from inferelator_velocity.utils.aggregation import (
    aggregate_sliding_window_times
)

from jtb_2022_code.utils.model_prediction import (
    plot_gene,
)

from jtb_2022_code.figure_constants import (
    MAIN_FIGURE_DPI,
    FIG_DYNAMICAL_FILE_NAME,
    FIGURE_5_FILE_NAME
)
from jtb_2022_code.utils.figure_common import (
    align_ylim,
    velocity_axes,
    plot_correlations,
    symmetric_ylim
)


def plot_figure_5(model_data, velo_data, predicts, save=True):

    fig = plt.figure(figsize=(8, 5), dpi=MAIN_FIGURE_DPI)

    y_top = 0.55
    y_top_2 = 0.19
    x_top = 0.4
    x_top_delta = 0.15
    h_top = 0.16
    w_top = 0.10

    y_bottom = 0.1
    h_bottom = 0.35

    axd = {
        "schematic": fig.add_axes(
            [0.05, y_top, 0.3, 0.35]
        ),
        "gene_1_expr": fig.add_axes(
            [x_top, y_top + y_top_2, w_top, h_top]
        ),
        "gene_2_expr": fig.add_axes(
            [x_top, y_top, w_top, h_top]
        ),
        "gene_1_velo": fig.add_axes(
            [x_top + x_top_delta, y_top + y_top_2, w_top, h_top]
        ),
        "gene_2_velo": fig.add_axes(
            [x_top + x_top_delta, y_top, w_top, h_top]
        ),
        "gene_1_comp": fig.add_axes(
            [x_top + 2 * x_top_delta, y_top + y_top_2, w_top, h_top]
        ),
        "gene_2_comp": fig.add_axes(
            [x_top + 2 * x_top_delta, y_top, w_top, h_top]
        ),
        "gene_1_decay": fig.add_axes(
            [x_top + 3 * x_top_delta, y_top + y_top_2, w_top, h_top]
        ),
        "gene_2_decay": fig.add_axes(
            [x_top + 3 * x_top_delta, y_top, w_top, h_top]
        ),
        "rp_decay": fig.add_axes(
            [0.525, y_bottom, 0.18, h_bottom]
        ),
        "decay_cat": fig.add_axes(
            [0.06, y_bottom, 0.015, h_bottom]
        ),
        "decay_corr": fig.add_axes(
            [0.08, y_bottom, 0.18, h_bottom]
        ),
        "count_corr": fig.add_axes(
            [0.265, y_bottom, 0.18, h_bottom]
        ),
        "rp_corr_count": fig.add_axes(
            [0.73, 0.3, 0.09, 0.15]
        ),
        "rp_corr_velo": fig.add_axes(
            [0.83, 0.3, 0.09, 0.15]
        ),
        "rp_corr_trans": fig.add_axes(
            [0.73, y_bottom, 0.09, 0.15]
        ),
        "rp_corr_decay": fig.add_axes(
            [0.83, y_bottom, 0.09, 0.15]
        ),
        "corr_cbar": fig.add_axes(
            [0.95, y_bottom + 0.03, 0.01, h_bottom - 0.06]
        ),
    }

    axd["schematic"].set_title("A", loc="left", weight="bold", y=1.03)
    axd["gene_1_expr"].set_title("B", loc="left", weight="bold", x=-0.28)
    axd["gene_1_velo"].set_title("C", loc="left", weight="bold", x=-0.28)
    axd["gene_1_comp"].set_title("D", loc="left", weight="bold", x=-0.28)
    axd["gene_1_decay"].set_title("E", loc="left", weight="bold", x=-0.28)
    axd["gene_1_expr"].set_title("Expression\n(Counts)", size=8)
    axd["gene_1_velo"].set_title("Velocity\n(Counts/Min)", size=8)
    axd["gene_1_comp"].set_title("Components\n(Counts/Min)", size=8)
    axd["gene_1_decay"].set_title("Half-Life\n(Min)", size=8)

    axd["rp_decay"].set_title("G", loc="left", weight="bold", x=-0.28)
    axd["decay_corr"].set_title("F", loc="left", weight="bold", x=-0.18)
    axd["rp_corr_count"].set_title("H", loc="left", weight="bold", x=-0.28)

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
        )

        align_ylim(axd[f"gene_{i + 1}_velo"], axd[f"gene_{i + 1}_comp"])
        velocity_axes(axd[f"gene_{i + 1}_comp"])
        axd[f"gene_{i + 1}_comp"].axvline(
            0, 0, 1, linestyle="--", linewidth=1.0, c="black"
        )

        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_decay"],
            rgen,
            layer="biophysical_predict_decay_constant",
            predicts=True,
            time_positive_only=False,
            annotation_loc=(0.6, 0.8) if i == 1 else (0.6, 0.2),
            gene_data_hook=lambda x: np.log(2) / x * -1,
        )
        axd[f"gene_{i + 1}_decay"].set_ylim(0, 60)
        axd[f"gene_{i + 1}_decay"].axvline(
            0, 0, 1, linestyle="--", linewidth=1.0, c="black"
        )

        if i == 1:
            axd[f"gene_{i + 1}_expr"].set_xticks(
                [0, 30, 60],
                [0, 30, 60],
                size=8
            )
            axd[f"gene_{i + 1}_decay"].set_xticks(
                [0, 30, 60],
                [0, 30, 60],
                size=8
            )

    plot_correlations(
        predicts.layers["biophysical_predict_decay_constant"],
        axd["decay_corr"],
        cmap="bwr",
        cat_ax=axd["decay_cat"],
        cat_cmap=colors.ListedColormap(["lightgray", "red"]),
        cat_var=model_data.var["RP"].astype(int).values,
    )

    _corr_ref, _all_idx = plot_correlations(
        predicts.layers["biophysical_predict_decay_constant"],
        axd["decay_corr"],
        cmap="bwr",
        cat_ax=axd["decay_cat"],
        cat_cmap=colors.ListedColormap(["lightgray", "red"]),
        cat_var=model_data.var["RP"].astype(int).values,
    )

    plot_correlations(
        predicts.layers["biophysical_predict_counts"],
        axd["count_corr"],
        cmap="bwr",
        plot_index=_all_idx,
    )

    _corr_ref, _rp_idx = plot_correlations(
        predicts.layers["biophysical_predict_decay_constant"][
            :, model_data.var["RP"]
        ],
        axd["rp_corr_decay"],
    )

    plot_correlations(
        predicts.layers["biophysical_predict_counts"][
            :, model_data.var["RP"]
        ],
        axd["rp_corr_count"],
        plot_index=_rp_idx,
    )

    plot_correlations(
        predicts.layers["biophysical_predict_transcription"][
            :, model_data.var["RP"]
        ],
        axd["rp_corr_trans"],
        plot_index=_rp_idx,
    )

    plot_correlations(
        predicts.layers["biophysical_predict_velocity"][
            :, model_data.var["RP"]
        ],
        axd["rp_corr_velo"],
        plot_index=_rp_idx,
    )

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
        alpha=0.1,
    )

    axd["decay_cat"].set_xticks([0.5], ["RP"], size=8, rotation=90)
    axd["decay_cat"].tick_params(axis="x", pad=2)

    axd["rp_decay"].set_ylim(0, 60)
    axd["rp_decay"].set_ylabel("Half-life (min)", size=8)
    axd["rp_decay"].set_title("Ribosomal Protein (RP)\nHalf-Life", size=8)
    axd["rp_decay"].set_xticks([0, 30, 60], [0, 30, 60], size=8)
    axd["rp_decay"].set_xlabel("Treatment Time (min)", size=8)
    axd["rp_decay"].axvline(0, 0, 1, linestyle="--", linewidth=1.0, c="black")

    axd["decay_corr"].set_title("Correlation\n(Decay)", size=8)
    axd["count_corr"].set_title("Correlation\n(Expression)", size=8)

    axd["rp_corr_decay"].set_title("Decay", size=8, y=0.9)
    axd["rp_corr_count"].set_title("Expression", size=8, y=0.9)
    axd["rp_corr_velo"].set_title("Velocity", size=8, y=0.9)
    axd["rp_corr_trans"].set_title("Transcription", size=8, y=0.9)

    plt.colorbar(_corr_ref, axd["corr_cbar"])
    axd["corr_cbar"].set_yticks([-1, 0, 1], [-1, 0, 1], size=8)
    axd["corr_cbar"].tick_params(axis="y", length=2, pad=1)
    axd["corr_cbar"].set_title("ρ", size=8)

    if save:
        fig.savefig(FIGURE_5_FILE_NAME + ".png", facecolor="white")

    return fig
