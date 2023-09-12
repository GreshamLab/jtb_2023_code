import matplotlib.pyplot as plt

from jtb_2023_code.figure_constants import (
    MAIN_FIGURE_DPI,
    FIG_DEEP_LEARNING_FILE_NAME,
    FIG_RAPA_LEGEND_VERTICAL_FILE_NAME,
    FIGURE_3_FILE_NAME
)
from jtb_2023_code.utils.model_result_loader import (
    load_model_results,
    summarize_model_results,
)

from jtb_2023_code.utils.model_prediction import plot_gene

import numpy as np

REP_COLORS = ["darkslateblue", "darkgoldenrod", "black"]


def figure_3_plot(model_data, predicts, save=True):
    summary_results, model_stats = summarize_model_results(
        load_model_results(trim_nans=False)[0]
    )

    fig = plt.figure(figsize=(4.5, 2), dpi=MAIN_FIGURE_DPI)

    rng = np.random.default_rng(100)

    axd = {
        "schematic": fig.add_axes([0.02, 0.02, 0.24, 0.96]),
        "results": fig.add_axes([0.36, 0.35, 0.23, 0.55]),
        "up_predicts": fig.add_axes([0.7, 0.6, 0.2, 0.35]),
        "down_predicts": fig.add_axes([0.7, 0.2, 0.2, 0.35]),
        "legend": fig.add_axes([0.91, 0.1, 0.08, 0.9]),
    }

    axd["schematic"].imshow(
        plt.imread(FIG_DEEP_LEARNING_FILE_NAME),
        aspect="equal"
    )
    axd["schematic"].axis("off")
    axd["schematic"].set_title(
        "A", loc="left", weight="bold", size=8, x=-0.1, y=0.92
    )

    _jitter = rng.uniform(-0.2, 0.2, summary_results.shape[0])
    axd["results"].scatter(
        summary_results["x_loc"] + _jitter,
        summary_results["AUPR"],
        color=summary_results["x_color"],
        s=5,
        alpha=0.5,
    )

    axd["results"].scatter(
        model_stats["x_loc"] + 0.5,
        model_stats["mean"],
        color=model_stats["x_color"],
        s=15,
        edgecolor="black",
        linewidth=0.25,
        alpha=1,
    )

    axd["results"].errorbar(
        model_stats["x_loc"] + 0.5,
        model_stats["mean"],
        yerr=model_stats["std"],
        fmt="none",
        color="black",
        alpha=1,
        linewidth=0.5,
        zorder=-1,
    )

    axd["results"].set_ylim(0, 0.3)
    axd["results"].set_xlim(0, 18.5)
    axd["results"].set_yticks([0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3], size=8)
    axd["results"].set_xticks(
        [1, 5, 9, 13, 17],
        ["Linear", "Static", "Dynamical", "Predictive", "Tuned\nPredictive"],
        size=7,
        rotation=90,
    )
    axd["results"].set_title("B", loc="left", weight="bold", size=8, x=-0.1)
    axd["results"].set_ylabel("AUPR", size=8)

    rgen = np.random.default_rng(441)

    plot_gene(
        model_data,
        "YKR039W",
        axd["up_predicts"],
        rgen,
        test_only=True,
        annotation_loc=(0.65, 0.8),
    )
    plot_gene(
        predicts,
        "YKR039W",
        axd["up_predicts"],
        rgen,
        predicts=True,
        annotation_loc=None,
        time_positive_only=True,
    )

    plot_gene(
        model_data,
        "YOR063W",
        axd["down_predicts"],
        rgen,
        test_only=True,
        annotation_loc=(0.65, 0.8),
    )
    plot_gene(
        predicts,
        "YOR063W",
        axd["down_predicts"],
        rgen,
        predicts=True,
        annotation_loc=None,
        time_positive_only=True,
    )

    axd["up_predicts"].set_title(
        "C", loc="left", weight="bold", size=8, x=-0.45, y=0.84
    )
    axd["up_predicts"].set_ylim(0, 8)
    axd["up_predicts"].set_xticks([], [])
    axd["down_predicts"].set_ylabel(
        "Transcript Counts", size=8, y=1.05, x=-0.15
    )
    axd["down_predicts"].set_xlabel("Time (min)", size=8)

    axd["down_predicts"].set_ylim(0, 22)

    axd["legend"].imshow(
        plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME),
        aspect="equal"
    )
    axd["legend"].axis("off")

    if save:
        fig.savefig(FIGURE_3_FILE_NAME + ".png", facecolor="white")

    return fig
