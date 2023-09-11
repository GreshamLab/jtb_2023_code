from jtb_2022_code.utils.figure_data import common_name
from jtb_2022_code.figure_constants import *
from jtb_2022_code.utils.figure_common import *

from jtb_2022_code.utils.model_prediction import (
    predict_all
)

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches

from jtb_2022_code.utils.model_prediction import (
    predict_all
)

from supirfactor_dynamical import (
    TimeDataset,
    TruncRobustScaler
)

from torch.utils.data import DataLoader


def _gradient_to_marginal(grad_tuple):
    return torch.multiply(
        grad_tuple[1],
        grad_tuple[2]
    ).numpy()


def _plot_elife_predicts(
    predictions,
    rapa,
    gene,
    genotype,
    genotype_common,
    ax,
    seed=44
):
    
    _rng = np.random.default_rng(seed)
    
    _gene_idx = rapa.var_names == gene

    _ypd = predictions[genotype][:, 0:10, _gene_idx].ravel()
    
    _rapa_tf_idx = rapa.obs["Genotype_Group"] == genotype_common
    _rapa = rapa.X[_rapa_tf_idx, _gene_idx].A.ravel()

    ax.scatter(
        np.tile(np.arange(0, 60), predictions[genotype].shape[0]),
        predictions[genotype][:, 10:, _gene_idx].ravel(),
        color='red',
        alpha=0.025,
        s=1
    )

    ax.scatter(
        _rng.uniform(-10, 0, size=_ypd.shape[0]),
        _ypd,
        color='black',
        alpha=0.025,
        s=1
    )

    ax.scatter(
        _rng.uniform(27.5, 32.5, size=_rapa.shape[0]),
        _rapa,
        color='black',
        alpha=0.025,
        s=1
    )

    ax.plot(
        [25, 35],
        [_rapa.mean()] * 2,
        color='black'
    )
    ax.plot(
        [25, 35],
        [predictions[genotype][:, 38:43, _gene_idx].mean()] * 2,
        color='darkred'
    )
    
    ax.annotate(
        common_name(gene),
        (0.1, 0.75),
        xycoords='axes fraction',
        color='black',
        size=7,
        fontstyle='italic',
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='white', boxstyle='round')
    )

    ax.axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black')
    ax.set_xticks([0, 30, 60], [])
    ax.tick_params('both', labelsize=8)


def _plot_tfa_difference_and_marginal_loss(
    plot_tfs,
    tfs,
    prediction_gradients,
    tfa_difference_ax=None,
    marginal_loss_ax=None,
    include_tf_index=None,
    marginal_labelpad=70
):
    
    x_all = np.arange(0, len(tfs) * 4 + 1, 2)
    
    if include_tf_index is None:
        include_tf_index = slice()
    
    for _i, tf in enumerate(tfs):
        x_axis = x_all[2 * _i: 2 * _i + 2]

        _tfa = np.hstack((
            prediction_gradients[None][2][:, -1, include_tf_index].mean(axis=0).reshape(-1, 1),
            prediction_gradients[tf][2][:, -1, include_tf_index].mean(axis=0).reshape(-1, 1)
        )).T
        
        _ups = plot_tfs[
            np.argsort(prediction_gradients[tf][2][:, -1, include_tf_index].mean(axis=0))[-3:]
        ]
                
        if tfa_difference_ax is not None:

            tfa_difference_ax.plot(
                x_axis,
                _tfa,
                color='black',
                alpha=0.1
            )

            tfa_difference_ax.scatter(
                np.repeat(x_axis, _tfa.shape[1]),
                _tfa.ravel(),
                color='black',
                s=4,
                alpha=1
            )

            tfa_difference_ax.plot(
                x_axis,
                _tfa[:, plot_tfs.isin(_ups)],
                color='red',
                alpha=1
            )

            tfa_difference_ax.scatter(
                np.repeat(x_axis, len(_ups)),
                _tfa[:, plot_tfs.isin(_ups)].ravel(),
                color='red',
                s=4,
                alpha=1
            )

            if tf in plot_tfs:
                _self_tfa = _tfa[:, plot_tfs == tf]
            else:
                _self_tfa = np.zeros_like(x_axis)
                
            tfa_difference_ax.plot(
                x_axis,
                _self_tfa,
                color='blue',
                alpha=1
            )

            tfa_difference_ax.scatter(
                x_axis,
                _self_tfa,
                color='blue',
                s=4,
                alpha=1
            )

            for _t in _ups:
                tfa_difference_ax.annotate(
                    common_name(_t),
                    (x_axis[1] + 0.05, _tfa[-1, plot_tfs == _t]),
                    color='red',
                    size=7
                )

            tfa_difference_ax.annotate(
                common_name(tf),
                (x_axis[0] - 1, _tfa[0, plot_tfs == tf]),
                color='blue',
                size=7
            )

        if marginal_loss_ax is not None:
            
            _marginal_loss = np.hstack((
                _gradient_to_marginal(
                    prediction_gradients[None]
                ).sum(axis=1).mean(axis=0).reshape(-1, 1),
                _gradient_to_marginal(
                    prediction_gradients[tf]
                ).sum(axis=1).mean(axis=0).reshape(-1, 1)
            ))[include_tf_index, :].T
            
            marginal_loss_ax.plot(
                x_axis,
                _marginal_loss,
                color='black',
                alpha=0.1
            )

            marginal_loss_ax.plot(
                x_axis,
                _marginal_loss[:, plot_tfs.isin(_ups)],
                color='red',
                alpha=1
            )

            for _i, _t in enumerate(_ups):
                if tf == "YER040W":
                    if _i == 0:
                        _offset = 0.0002
                    elif _i == 1:
                        _offset = -0.0001
                    else:
                        _offset = 0
                else:
                    if _i == 0:
                        _offset = 0.0001
                    elif _i == 1:
                        _offset = 0.00005
                    else:
                        _offset = -0.00005

                marginal_loss_ax.annotate(
                    common_name(_t),
                    (x_axis[1] + 0.05, _marginal_loss[-1, plot_tfs == _t] - _offset),
                    color='red',
                    size=7
                )

    if tfa_difference_ax is not None:
        tfa_difference_ax.set_xlim(-1, x_all.max() + 1)

        _labs = []
        for t in tfs:
            _labs.append(f"WT")
            _labs.append(f"Δ{common_name(t).lower()}")

        tfa_difference_ax.tick_params('x', length=0, which='major')
        tfa_difference_ax.set_xticks(
            np.arange(0, x_all.max(), 2),
            _labs,
            rotation=90, size=8
        )
        tfa_difference_ax.set_yticks(
            [], []
        )

    if marginal_loss_ax is not None:
        velocity_axes(marginal_loss_ax)
        marginal_loss_ax.set_xlim(-1, x_all.max() + 2)

        _labs = []
        for t in tfs:
            _labs.append(f"WT\n({prediction_gradients[None][0]:.3f})")
            _labs.append(f"Δ{common_name(t).lower()}\n({prediction_gradients[t][0]:.3f})")

        marginal_loss_ax.tick_params('x', length=0, which='major', pad=marginal_labelpad)
        marginal_loss_ax.set_xticks(
            np.arange(0, x_all.max(), 2),
            _labs,
            rotation=90,
            size=8
        )

        marginal_loss_ax.set_yticks([], [])



def plot_figure_6(
    predictions,
    rapa,
    tfa_predictions,
    prediction_gradients,
    model,
    save=True
):
    
    fig_refs = {}
    fig = plt.figure(figsize=(7, 3), dpi=MAIN_FIGURE_DPI)

    axd = {
        'tfa': fig.add_axes([0.05, 0.15, 0.22, 0.725]),
        'predict_1': fig.add_axes([0.31, 0.15, 0.13, 0.3]),
        'predict_2': fig.add_axes([0.31, 0.555, 0.13, 0.3]),
        'gln3_tfa': fig.add_axes([0.48, 0.15, 0.2, 0.725]),
        'marginal_loss': fig.add_axes([0.73, 0.15, 0.23, 0.725])
    }

    axd['tfa'].set_title("A", size=8, loc='left', weight='bold', y=1.05, x=-0.1)
    axd['predict_2'].set_title("B", size=8, loc='left', weight='bold', y=1.15, x=-0.1)
    axd['gln3_tfa'].set_title("C", size=8, loc='left', weight='bold', y=1.05, x=-0.1)
    axd['marginal_loss'].set_title("D", size=8, loc='left', weight='bold', y=1.05, x=-0.1)

    _wt_tfa = tfa_predictions[None].mean(0)
    _has_tfa = _wt_tfa.mean(0) > 0

    axd['tfa'].plot(
        np.arange(-10, 60),
        _wt_tfa[:, _has_tfa],
        color='black',
        alpha=0.1
    )

    _tfs = model.prior_network_labels[1][_has_tfa]
    _pre_tfs = _tfs[np.argsort(_wt_tfa[0, _has_tfa])[-4:]]
    _post_tfs = _tfs[np.argsort(_wt_tfa[-1, _has_tfa])[-2:]]

    axd['tfa'].plot(
        np.arange(-10, 60),
        _wt_tfa[:, _has_tfa][:, _tfs.isin(_pre_tfs)],
        color='blue',
        alpha=1
    )

    axd['tfa'].plot(
        np.arange(-10, 60),
        _wt_tfa[:, _has_tfa][:, _tfs.isin(_post_tfs)],
        color='red',
        alpha=1
    )

    for _i, _t in enumerate(_pre_tfs):
        if _i == 0:
            _offset = 0.2
        elif _i == 1:
            _offset = -0.1
        else:
            _offset = 0.1
        axd['tfa'].annotate(
            common_name(_t),
            (-28, _wt_tfa[0, _has_tfa][_tfs == _t] - _offset),
            color='blue',
            size=7
        )

    for _t in _post_tfs:
        axd['tfa'].annotate(
            common_name(_t),
            (60, _wt_tfa[-1, _has_tfa][_tfs == _t]),
            color='red',
            size=7
        )

    axd['tfa'].set_xlim(-30, 80)
    axd['tfa'].set_xlabel("Rapamycin Response Time (min)", size=8)
    axd['tfa'].set_ylabel("TF Activity (TFA)", size=8)
    axd['tfa'].set_xticks([0, 30, 60], [0, 30, 60], size=8)
    axd['tfa'].set_yticks([], [])
    axd['tfa'].axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black')
    axd['tfa'].axvline(30, 0, 1, linestyle=':', linewidth=1.0, c='magenta')
    axd['tfa'].set_title("Transcription Factor\nActivity (TFA)", size=8)
    axd['tfa'].spines['right'].set_color('none')
    axd['tfa'].spines['top'].set_color('none')

    _plot_elife_predicts(predictions, rapa, "YKR039W", None, "WT(ho)", axd['predict_2'])
    _plot_elife_predicts(predictions, rapa, "YOR063W", None, "WT(ho)", axd['predict_1'])
    axd['predict_2'].set_title("Timepoint\nPrediction (WT)", size=8)
    axd['predict_2'].set_ylim(0, 10)
    axd['predict_1'].set_ylim(0, 22)
    axd['predict_1'].set_xticks([0, 30, 60], [0, 30, 60], size=8)

    _plot_tfa_difference_and_marginal_loss(
        model.prior_network_labels[1][_has_tfa],
        ['YER040W', 'YEL009C'],
        prediction_gradients,
        axd['gln3_tfa'],
        axd['marginal_loss'],
        _has_tfa
    )

    axd['gln3_tfa'].set_ylabel("TFA (t=30 minutes)", size=9)
    axd['gln3_tfa'].set_title("TFA ($\it{In \; silico}$\nTF Deletion)", size=8)
    axd['gln3_tfa'].spines['right'].set_color('none')
    axd['gln3_tfa'].spines['top'].set_color('none')

    axd['marginal_loss'].set_ylim(-0.0015, 0.0015)
    axd['marginal_loss'].set_xlim(-2, 7)

    axd['marginal_loss'].set_title("Marginal\nTFA Error", size=8)
    axd['marginal_loss'].set_ylabel("TFA Error", size=8)

    axd['marginal_loss'].annotate(
        "TFA Over-\nestimated",
        (0.05, 0.875),
        xycoords='axes fraction',
        size=8
    )
    axd['marginal_loss'].annotate(
        "TFA Under-\nestimated",
        (0.05, 0.2),
        xycoords='axes fraction',
        size=8
    )

    if save:
        fig.savefig(FIGURE_6_FILE_NAME + ".png", facecolor='white')
        
    return fig
