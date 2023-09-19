from jtb_2023_code.figure_constants import (
    MAIN_FIGURE_DPI,
    FIGURE_4_GENES,
    FIG_RAPA_LEGEND_VERTICAL_FILE_NAME,
    FIGURE_4_FILE_NAME
)
from jtb_2023_code.utils.figure_common import (
    to_pool_colors
)

from jtb_2023_code.utils.model_result_loader import (
    load_model_results,
    summarize_model_results,
)
from jtb_2023_code.utils.model_prediction import plot_gene

import numpy as np

import matplotlib.pyplot as plt


def plot_figure_4(model_data, velo_data, predicts, save=True):
    results, _ = load_model_results("Velocity")
    summary_results, summary_stats = summarize_model_results(
        results,
        "Velocity"
    )

    fig = plt.figure(figsize=(6, 2.75), dpi=MAIN_FIGURE_DPI)

    axd = {
        "results": fig.add_axes([0.08, 0.27, 0.15, 0.63]),
        "gene_1_velo": fig.add_axes([0.335, 0.55, 0.15, 0.35]),
        "gene_2_velo": fig.add_axes([0.335, 0.12, 0.15, 0.35]),
        "gene_1_expr": fig.add_axes([0.56, 0.55, 0.15, 0.35]),
        "gene_2_expr": fig.add_axes([0.56, 0.12, 0.15, 0.35]),
        "pca_predicts": fig.add_axes([0.76, 0.12, 0.15, 0.35]),
        "pca_counts": fig.add_axes([0.76, 0.55, 0.15, 0.35]),
        "legend": fig.add_axes([0.91, 0.2, 0.1, 0.6]),
    }

    rgen = np.random.default_rng(441)

    for i, g in enumerate(FIGURE_4_GENES):
        plot_gene(
            model_data,
            g,
            axd[f"gene_{i + 1}_expr"],
            rgen,
            velocity=False,
            annotation_loc=(0.65, 0.8),
            test_only=True,
        )
        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_expr"],
            rgen,
            predicts=True,
            layer="velocity_predict_counts",
            alpha=0.01,
            annotation_loc=None,
            time_positive_only=True,
        )

        plot_gene(
            velo_data,
            g,
            axd[f"gene_{i + 1}_velo"],
            rgen,
            velocity=True,
            annotation_loc=(0.65, 0.8),
            test_only=True,
        )
        plot_gene(
            predicts,
            g,
            axd[f"gene_{i + 1}_velo"],
            rgen,
            layer="velocity_predict_velocity",
            predicts=True,
            alpha=0.01,
            annotation_loc=None,
            time_positive_only=True,
        )

    axd["gene_1_expr"].set_title(
        "C", loc="left", weight="bold", size=8, x=-0.3
    )
    axd["gene_1_velo"].set_title(
        "B", loc="left", weight="bold", size=8, x=-0.3
    )
    axd["gene_1_expr"].set_ylim(0, 8)
    axd["gene_2_expr"].set_ylim(0, 22)
    axd["pca_counts"].set_title("D", loc="left", weight="bold", size=8, x=-0.2)
    axd["results"].set_title("A", loc="left", weight="bold", size=8, x=-0.5)

    axd["gene_2_expr"].set_xlabel("Time (min)", size=8, labelpad=-0.5)
    axd["gene_2_expr"].set_xticks([0, 30, 60], [0, 30, 60], size=8)
    axd["gene_2_velo"].set_xlabel("Time (min)", size=8, labelpad=42)
    axd["gene_2_velo"].set_ylabel(
        "Transcript Velocity", size=8, y=1.05, x=-0.15
    )
    axd["gene_2_expr"].set_ylabel(
        "Transcript Counts", size=8, y=1.05, x=-0.15
    )

    _jitter = rgen.uniform(-0.2, 0.2, summary_results.shape[0])
    axd["results"].scatter(
        summary_results["x_loc"] + _jitter,
        summary_results["AUPR"],
        color=summary_results["x_color"],
        s=5,
        alpha=0.7,
    )

    _is_test = model_data.obs["Test"].values
    _count_overplot = np.arange(_is_test.sum())
    rgen.shuffle(_count_overplot)

    axd["pca_counts"].scatter(
        model_data.obsm["X_pca"][_is_test, 0][_count_overplot],
        model_data.obsm["X_pca"][_is_test, 1][_count_overplot],
        s=1,
        color=to_pool_colors(
            model_data.obs["Pool"]
        ).astype(str)[_is_test][_count_overplot],
        alpha=0.1,
    )
    axd["pca_counts"].set_xticks([], [])
    axd["pca_counts"].set_yticks([], [])
    axd["pca_counts"].set_xlabel(
        f"PC1 ({model_data.uns['pca']['variance_ratio'][0] * 100:.0f}%)",
        size=8
    )
    axd["pca_counts"].set_ylabel(
        f"PC2 ({model_data.uns['pca']['variance_ratio'][1] * 100:.0f}%)",
        size=8
    )
    axd["pca_counts"].annotate(
        "Observed",
        (0.35, 0.85),
        xycoords="axes fraction",
        color="black",
        size=7,
        bbox=dict(
            facecolor="white",
            alpha=0.5,
            edgecolor="white",
            boxstyle="round"
        ),
    )

    _predict_overplot = np.arange(predicts.shape[0])
    rgen.shuffle(_predict_overplot)

    axd["pca_predicts"].scatter(
        predicts.obsm["X_pca"][_predict_overplot, 0] * -1,
        predicts.obsm["X_pca"][_predict_overplot, 1],
        color=predicts.obs["color"].values[_predict_overplot],
        s=1,
        alpha=0.1,
    )

    axd["pca_predicts"].set_xticks([], [])
    axd["pca_predicts"].set_yticks([], [])
    axd["pca_predicts"].set_xlabel(
        f"PC1 ({predicts.uns['pca']['variance_ratio'][0] * 100:.0f}%)",
        size=8
    )
    axd["pca_predicts"].set_ylabel(
        f"PC2 ({predicts.uns['pca']['variance_ratio'][1] * 100:.0f}%)",
        size=8
    )
    axd["pca_predicts"].annotate(
        "Predicted",
        (0.35, 0.85),
        xycoords="axes fraction",
        color="black",
        size=7,
        bbox=dict(
            facecolor="white",
            alpha=0.5,
            edgecolor="white",
            boxstyle="round"
        ),
    )

    axd["results"].scatter(
        summary_stats["x_loc"] + 0.5,
        summary_stats["mean"],
        color=summary_stats["x_color"],
        s=15,
        edgecolor="black",
        linewidth=0.25,
        alpha=1,
    )

    axd["results"].errorbar(
        summary_stats["x_loc"] + 0.5,
        summary_stats["mean"],
        yerr=summary_stats["std"],
        fmt="none",
        color="black",
        alpha=1,
        linewidth=0.5,
        zorder=-1,
    )

    axd["results"].set_ylim(0, 0.3)
    axd["results"].set_ylabel("AUPR", size=8)
    axd["results"].set_xlim(3.5, 18.5)
    axd["results"].set_yticks([0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3], size=8)
    axd["results"].set_xticks(
        [5, 9, 13, 17],
        ["Static", "Dynamical", "Predictive", "Tuned\nPredictive"],
        size=7,
        rotation=90,
    )

    axd["results"].tick_params(axis="both", which="major", labelsize=8)

    axd["legend"].imshow(
        plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME),
        aspect="equal"
    )
    axd["legend"].axis("off")

    if save:
        fig.savefig(FIGURE_4_FILE_NAME + ".png", facecolor="white")

    return fig
