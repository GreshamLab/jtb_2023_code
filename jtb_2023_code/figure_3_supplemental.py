import numpy as np
import pandas as pd
import anndata as ad

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from supirfactor_dynamical import TruncRobustScaler

from jtb_2023_code.figure_constants import (
    INFERELATOR_DATA_FILE,
    INFERELATOR_PRIORS_FILE,
    SUPPLEMENTAL_FIGURE_DPI,
    FIGURE_3_SUPPLEMENTAL_FILE_NAME,
    FIG_DEEP_LEARNING_TRAINING_FILE_NAME
)
from jtb_2023_code.utils.model_result_loader import (
    load_model_results,
    plot_results,
    plot_losses,
    get_plot_idx,
)

model_labels = {
    "static_meta": "Static",
    "rnn": "Dynamical",
    "rnn_predictive": "Predictive",
    "tuned": "Tuned",
}


def figure_3_supplement_1_plot(save=True):
    data = ad.read(INFERELATOR_DATA_FILE)
    prior = pd.read_csv(INFERELATOR_PRIORS_FILE, sep="\t", index_col=0)

    data_scaler = RobustScaler(with_centering=False)
    data.layers["robust_scaled"] = data_scaler.fit_transform(data.X)
    data_scaler_v = RobustScaler(with_centering=False)
    data.layers["robust_scaled_rapa_velocity"] = data_scaler_v.fit_transform(
        data.layers["rapamycin_velocity"]
    )

    trunc_scaler = TruncRobustScaler(with_centering=False)
    data.layers["special_scaled"] = trunc_scaler.fit_transform(data.X)
    data.layers["special_scaled_rapa_velocity"] = trunc_scaler.fit_transform(
        data.layers["rapamycin_velocity"]
    )

    class RobustMinScaler(RobustScaler):
        """Applies the RobustScaler and then adjust minimum value to 0."""

        def transform(self, X):
            X = super().transform(X)

            _data_min = X.min(axis=0)

            try:
                _data_min = _data_min.A.flatten()
            except AttributeError:
                pass

            _min_g_zero = _data_min > 0

            if np.any(_min_g_zero):
                _mins = _data_min[_min_g_zero][None, :]
                X[:, _min_g_zero] = X[:, _min_g_zero] - _mins

            return X

    rmin_scaler = RobustMinScaler(with_centering=False, quantile_range=(1, 99))
    data.layers["robustminscaler_scaled"] = rmin_scaler.fit_transform(data.X)
    data.layers["robustminscaler_scaled_velocity"] = rmin_scaler.fit_transform(
        data.layers["rapamycin_velocity"]
    )

    fig, ax = plt.subplots(
        2, 4,
        dpi=SUPPLEMENTAL_FIGURE_DPI,
        figsize=(8, 6),
        gridspec_kw={"wspace": 0.3, "hspace": 0.75, 'bottom': 0.4}
    )

    axd = {
        "prior_bar": fig.add_axes([0.125, 0.075, 0.345, 0.2]),
        "prior_genes": fig.add_axes([0.555, 0.075, 0.345, 0.2]),
    }

    axd["prior_bar"].bar(
        np.arange(prior.shape[1]),
        (prior != 0).sum().sort_values(ascending=False),
        width=1.0,
        color="black",
    )

    axd["prior_bar"].set_title("Prior Knowledge Network Connectivity", size=8)
    axd["prior_bar"].set_title(
        "C", loc="left", weight="bold", size=10, x=-0.25
    )
    axd["prior_bar"].set_xlabel("Regulatory TFs", size=8)
    axd["prior_bar"].set_ylabel("# Target Genes", size=8)
    axd["prior_bar"].set_xticks([0, prior.shape[1]], [0, prior.shape[1]])
    axd["prior_bar"].tick_params(axis="both", which="major", labelsize=8)
    axd["prior_bar"].set_xlim(0, prior.shape[1])

    axd["prior_bar"].annotate(
        f"n = {(prior != 0).sum().sum()} edges",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        size=8,
    )

    axd["prior_genes"].bar(
        np.arange(prior.shape[0]),
        (prior != 0).sum(axis=1).sort_values(ascending=False),
        width=1.0,
        color="black",
    )

    axd["prior_genes"].set_xlabel("Target Genes", size=8)
    axd["prior_genes"].set_ylabel("# Regulatory TFs", size=8)
    axd["prior_genes"].set_xticks([0, prior.shape[0]], [0, prior.shape[0]])
    axd["prior_genes"].tick_params(axis="both", which="major", labelsize=8)
    axd["prior_genes"].set_xlim(0, prior.shape[0])

    axd["prior_genes"].annotate(
        f"n = {(prior != 0).sum().sum()} edges",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        size=8,
    )

    def _get_var(x):
        scalar = StandardScaler(with_mean=False)
        scalar.fit(x)
        return scalar.var_

    def _plot_var_hist(layer, ax):
        if f"{layer}_var" not in data.var.columns:
            data.var[f"{layer}_var"] = _get_var(
                data.layers[layer] if layer != "X" else data.X
            )

        _vars = data.var[f"{layer}_var"].values
        ax.hist(np.log(_vars), bins=100, color="gray")
        ax.set_xlim(-10, 10)
        ax.set_xlabel("Log Variance", size=8)
        ax.axvline(
            0, 0, 1,
            color="black",
            zorder=1,
            linewidth=0.5,
            linestyle="--"
        )
        ax.annotate(
            f"Max Var:\n{_vars.max():.1f}",
            xy=(0.55, 0.75),
            xycoords="axes fraction",
            size=6,
        )
        ax.tick_params(axis="both", which="major", labelsize=8)

    _plot_var_hist("X", ax[0, 0])
    ax[0, 0].set_title("Counts", size=8)
    ax[0, 0].set_title("A", loc="left", weight="bold", size=10, x=-0.25)

    _plot_var_hist("robust_scaled", ax[0, 1])
    ax[0, 1].set_title("RobustScaled\nCounts", size=8)

    _plot_var_hist("special_scaled", ax[0, 2])
    ax[0, 2].set_title("TruncRobustScaled\nCounts", size=8)

    _plot_var_hist("robustminscaler_scaled", ax[0, 3])
    ax[0, 3].set_title("RobustMinScaled\nCounts", size=8)

    _plot_var_hist("rapamycin_velocity", ax[1, 0])
    ax[1, 0].set_title("Velocity", size=8)
    ax[1, 0].set_title("B", loc="left", weight="bold", size=10, x=-0.25)

    _plot_var_hist("robust_scaled_rapa_velocity", ax[1, 1])
    ax[1, 1].set_title("RobustScaled\nVelocity", size=8)

    _plot_var_hist("special_scaled_rapa_velocity", ax[1, 2])
    ax[1, 2].set_title("TruncRobustScaled\nVelocity", size=8)

    _plot_var_hist("robustminscaler_scaled_velocity", ax[1, 3])
    ax[1, 3].set_title("RobustMinScaled\nVelocity", size=8)

    if save:
        fig.savefig(
            FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_1.png",
            facecolor="white"
        )

    return fig


def figure_3_supplement_2_plot(save=True):

    supirfactor_results, supirfactor_losses = load_model_results()

    fig, axd = plt.subplots(
        4,
        6,
        figsize=(8, 8),
        dpi=SUPPLEMENTAL_FIGURE_DPI,
        gridspec_kw={"wspace": 0.5, "hspace": 0.6},
    )
    plt.subplots_adjust(top=0.8, bottom=0.1, left=0.08, right=0.95)

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
                        ),
                        :,
                    ].copy(),
                    metric,
                    param,
                    axd[j, i],
                    tuned=model == "tuned",
                )

            if i == 0:
                axd[j, i].set_title(
                    chr(66 + j), loc="left", weight="bold", size=10, x=-0.25
                )

    ax_lr = fig.add_axes([0.02, 0.02, 0.48, 0.88], zorder=-3)
    ax_wd = fig.add_axes([0.51, 0.02, 0.45, 0.88], zorder=-3)
    ax_schema = fig.add_axes([0.08, 0.92, 0.35, 0.08])
    ax_schema.imshow(
        plt.imread(FIG_DEEP_LEARNING_TRAINING_FILE_NAME),
        aspect="equal"
    )
    ax_schema.set_title(
        "A", loc="left", weight="bold", size=10, x=-0.07, y=0.7
    )
    ax_schema.axis("off")

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
        xy=(0.001, 0.79),
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
        xy=(0.001, 0.32),
        xycoords="axes fraction",
        size=10,
        weight="bold",
        rotation=90,
    )
    ax_lr.annotate(
        "Tuned",
        xy=(0.001, 0.13),
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
            FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_2.png",
            facecolor="white"
        )

    return fig


def figure_3_supplement_3_plot(save=True):

    supirfactor_results, supirfactor_losses = load_model_results()

    fig, axd = plt.subplots(
        4,
        6,
        figsize=(8, 8),
        dpi=SUPPLEMENTAL_FIGURE_DPI,
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
                        ),
                        :,
                    ].copy(),
                    metric,
                    param,
                    axd[j, i],
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
            FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_3.png",
            facecolor="white"
        )

    return fig
