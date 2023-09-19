import numpy as np

import matplotlib.pyplot as plt

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
    symmetric_ylim
)


def plot_figure_5(model_data, velo_data, predicts, save=True):

    fig = plt.figure(figsize=(8, 2.5), dpi=MAIN_FIGURE_DPI)

    axd = {
        "schematic": fig.add_axes(
            [0, 0, 0.3, 0.95]
        ),
        "gene_1_expr": fig.add_axes(
            [0.35, 0.55, 0.10, 0.35]
        ),
        "gene_2_expr": fig.add_axes(
            [0.35, 0.12, 0.10, 0.35]
        ),
        "gene_1_velo": fig.add_axes(
            [0.52, 0.55, 0.10, 0.35]
        ),
        "gene_2_velo": fig.add_axes(
            [0.52, 0.12, 0.10, 0.35]
        ),
        "gene_1_comp": fig.add_axes(
            [0.64, 0.55, 0.10, 0.35]
        ),
        "gene_2_comp": fig.add_axes(
            [0.64, 0.12, 0.10, 0.35]
        ),
        "rp_decay": fig.add_axes(
            [0.8, 0.15, 0.18, 0.72]
        )
    }

    axd["schematic"].set_title("A", loc="left", weight="bold")
    axd["gene_1_expr"].set_title("B", loc="left", weight="bold", x=-0.28)
    axd["gene_1_velo"].set_title("C", loc="left", weight="bold", x=-0.28)
    axd["gene_1_comp"].set_title("D", loc="left", weight="bold", x=-0.18)
    axd["rp_decay"].set_title("E", loc="left", weight="bold", x=-0.28, y=1.045)

    axd["gene_2_expr"].set_ylabel("Transcript Counts", y=1.05, size=8)
    axd["gene_2_velo"].set_ylabel("Transcript Velocity", y=1.05, size=8)
    axd["gene_2_expr"].set_xlabel("Time (min)", size=8, labelpad=-0.5)
    axd["gene_2_velo"].set_xlabel("Time (min)", size=8, labelpad=39)

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
    axd["rp_decay"].set_ylabel("Half-life (min)", size=8)
    axd["rp_decay"].set_title("Ribosomal Protein (RP)\nStability", size=8)
    axd["rp_decay"].set_xticks([0, 30, 60], [0, 30, 60], size=8)
    axd["rp_decay"].set_xlabel("Time (min)", size=8)
    axd["rp_decay"].axvline(0, 0, 1, linestyle="--", linewidth=1.0, c="black")

    if save:
        fig.savefig(FIGURE_5_FILE_NAME + ".png", facecolor="white")

    return fig
