import matplotlib.pyplot as plt

from jtb_2023_code.figure_6 import (
    _plot_elife_predicts,
    _plot_tfa_difference_and_marginal_loss,
)

from jtb_2023_code.figure_constants import (
    MAIN_FIGURE_DPI,
    FIGURE_6_SUPPLEMENTAL_FILE_NAME
)


def figure_6_supplement_1_plot(predictions, rapa, save=True):
    fig, axd = plt.subplots(
        4,
        3,
        figsize=(4, 5),
        dpi=MAIN_FIGURE_DPI,
        gridspec_kw={"wspace": 0.4, "hspace": 0.5, "left": 0.2},
    )

    _gene = ["YKR039W", "YPR035W", "YNL142W", "YOR063W"]

    for i, _g in enumerate(_gene):
        for j, (_tf, _tf_common, _tf_name) in enumerate(
            [
                (None, "WT(ho)", "WT"),
                ("YER040W", "gln3", "Δgln3"),
                ("YEL009C", "gcn4", "Δgcn4"),
            ]
        ):
            _plot_elife_predicts(
                predictions,
                rapa,
                _g,
                _tf,
                _tf_common,
                axd[i, j]
            )

            axd[i, j].set_ylim(0, [10, 20, 10, 30][i])
            axd[i, j].set_xticks([0, 30, 60], [0, 30, 60])
            axd[i, j].tick_params(axis="both", which="major", labelsize=8)

            if i == 0:
                axd[i, j].set_title(f"{_tf_name}", size=8)
            if j == 0:
                axd[i, j].set_title(
                    chr(65 + i), loc="left", size=8, weight="bold", x=-0.28
                )
                axd[i, j].set_ylabel("Expression\n(Counts)", size=8)

    if save:
        fig.savefig(
            FIGURE_6_SUPPLEMENTAL_FILE_NAME + "_1.png",
            facecolor="white"
        )

    return fig


def figure_6_supplement_2_plot(
    tfa_predictions,
    prediction_gradients,
    model,
    save=True
):
    fig = plt.figure(figsize=(7, 4), dpi=MAIN_FIGURE_DPI)

    axd = {
        "tfa_all": fig.add_axes([0.05, 0.575, 0.9, 0.325]),
        "loss_all": fig.add_axes([0.05, 0.10, 0.9, 0.325]),
    }

    _wt_tfa = tfa_predictions[None].mean(0)
    _has_tfa = _wt_tfa.mean(0) > 0

    _plot_tfa_difference_and_marginal_loss(
        model.prior_network_labels[1][_has_tfa],
        [
            "YER040W",
            "YEL009C",
            "YJL110C",
            "YOL067C",
            "YBL103C",
            "YDR463W",
            "YHR006W",
            "YIR023W",
        ],
        prediction_gradients,
        axd["tfa_all"],
        axd["loss_all"],
        _has_tfa,
        marginal_labelpad=40,
    )

    axd["tfa_all"].set_ylabel("TFA (t=30 minutes)", size=9)
    axd["tfa_all"].tick_params("y", size=8)
    axd["tfa_all"].spines["right"].set_color("none")
    axd["tfa_all"].spines["top"].set_color("none")

    axd["loss_all"].set_ylim(-0.0015, 0.0015)
    axd["loss_all"].tick_params("y", size=8, which="major", length=0)
    axd["loss_all"].set_ylabel("TFA Error", size=8)

    axd["tfa_all"].set_title("A", size=8, loc="left", weight="bold", x=-0.025)
    axd["loss_all"].set_title("B", size=8, loc="left", weight="bold", x=-0.025)

    if save:
        fig.savefig(
            FIGURE_6_SUPPLEMENTAL_FILE_NAME + "_2.png",
            facecolor="white"
        )

    return fig
