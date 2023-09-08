import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *
from jtb_2022_code.utils.decay_common import _halflife
from jtb_2022_code.utils.model_result_loader import (
    load_model_results,
    summarize_model_results
)

from jtb_2022_code.utils.model_prediction import (
    plot_gene
)

from inferelator_velocity.utils.aggregation import aggregate_sliding_window_times

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

REP_COLORS = ['darkslateblue', 'darkgoldenrod', 'black']


def figure_3_plot(model_data, predicts, save=True):
    
    summary_results, model_stats = summarize_model_results(
        load_model_results(trim_nans=False)[0]
    )
    
    fig_refs = {}
    fig = plt.figure(figsize=(4.5, 2), dpi=MAIN_FIGURE_DPI)

    rng = np.random.default_rng(100)

    _height = 0.37
    _width = 0.25

    axd = {
        'schematic': fig.add_axes([0.02, 0.02, 0.24, 0.96]),
        'results': fig.add_axes([0.36, 0.35, 0.23, 0.55]),
        'up_predicts': fig.add_axes([0.7, 0.6, 0.2, 0.35]),
        'down_predicts': fig.add_axes([0.7, 0.2, 0.2, 0.35]),
        'legend': fig.add_axes([0.91, 0.1, 0.08, 0.9])
    }

    axd['schematic'].imshow(plt.imread(FIG_DEEP_LEARNING_FILE_NAME), aspect='equal')
    axd['schematic'].axis('off')
    axd['schematic'].set_title("A", loc='left', weight='bold', size=8, x=-0.1, y=0.92)

    axd['results'].scatter(
        summary_results['x_loc'] + rng.uniform(-0.2, 0.2, summary_results.shape[0]),
        summary_results['AUPR'],
        color=summary_results['x_color'],
        s=5,
        alpha=0.5
    )

    axd['results'].scatter(
        model_stats['x_loc'] + 0.5,
        model_stats['mean'],
        color=model_stats['x_color'],
        s=15,
        edgecolor='black',
        linewidth=0.25,
        alpha=1
    )

    axd['results'].errorbar(
        model_stats['x_loc'] + 0.5,
        model_stats['mean'],
        yerr=model_stats['std'],
        fmt='none',
        color='black',
        alpha=1,
        linewidth=0.5,
        zorder=-1
    )

    axd['results'].set_ylim(0, 0.3)
    axd['results'].set_xlim(0, 18.5)
    axd['results'].set_yticks([0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3], size=8)
    axd['results'].set_xticks([1, 5, 9, 13, 17], ['Linear', 'Static', 'Dynamical', 'Predictive', 'Tuned\nPredictive'], size=7, rotation=90)
    axd['results'].set_title("B", loc='left', weight='bold', size=8, x=-.1)
    axd['results'].set_ylabel("AUPR", size=8)

    rgen = np.random.default_rng(441)

    plot_gene(model_data, "YKR039W", axd['up_predicts'], rgen, test_only=True, annotation_loc=(0.65, 0.8))
    plot_gene(predicts, "YKR039W", axd['up_predicts'], rgen, predicts=True, annotation_loc=None)

    plot_gene(model_data, "YOR063W", axd['down_predicts'], rgen, test_only=True, annotation_loc=(0.65, 0.8))
    plot_gene(predicts, "YOR063W", axd['down_predicts'], rgen, predicts=True, annotation_loc=None)

    axd['up_predicts'].set_title("C", loc='left', weight='bold', size=8, x=-0.45, y=0.84)
    axd['up_predicts'].set_ylim(0, 8)
    axd['up_predicts'].set_xticks([], [])
    axd['down_predicts'].set_ylabel("Transcript Counts", size=8, y=1.05, x=-0.15)
    axd['down_predicts'].set_xlabel("Time (min)", size=8)

    axd['down_predicts'].set_ylim(0, 22)

    axd['legend'].imshow(plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME), aspect='equal')
    axd['legend'].axis('off')

    if save:
        fig.savefig(FIGURE_3_FILE_NAME + ".png", facecolor="white")

    return fig


def _fig3_plot(
    fig_data,
    color_data,
    genes,
    gene_labels=None,
    time_key='program_0_time'
):
    
    fig_refs = {}
    fig = plt.figure(figsize=(4.5, 5), dpi=MAIN_FIGURE_DPI)

    _left_x = 0.05
    _height = 0.27
    _width = 0.3
    _x_right = 0.6

    axd = {
        'counts_1': fig.add_axes([0.175, 0.68, _width, _height]),
        'velocity_1': fig.add_axes([0.175, 0.38, _width, _height]),
        'decay_1': fig.add_axes([0.175, 0.08, _width, _height]),
        'counts_2': fig.add_axes([_x_right, 0.68, _width, _height]),
        'velocity_2': fig.add_axes([_x_right, 0.38, _width, _height]),
        'decay_2': fig.add_axes([_x_right, 0.08, _width, _height]),
        'legend': fig.add_axes([0.91, 0.38, 0.1, 0.57]),
        'elegend': fig.add_axes([0.91, 0.08, 0.1, _height])
    }

    if gene_labels is None:
        gene_labels = genes
    
    for i, g in enumerate(genes):

        rgen = np.random.default_rng(441)
        overplot_shuffle = np.arange(fig_data.shape[0])
        rgen.shuffle(overplot_shuffle)

        fig_refs[f'counts{i+1}'] = axd[f'counts_{i+1}'].scatter(
            x=fig_data.obs[time_key][overplot_shuffle], 
            y=fig_data.layers['denoised'][overplot_shuffle, i],
            c=color_data[overplot_shuffle],
            alpha=0.2, 
            s=1
        )

        median_counts, _window_centers = aggregate_sliding_window_times(
            fig_data.layers['denoised'][:, i].reshape(-1, 1),
            fig_data.obs[time_key],
            centers=np.linspace(-10 + 0.5, 65 - 0.5, 75),
            width=1.
        )

        axd[f'counts_{i+1}'].plot(
            _window_centers, 
            median_counts,
            c='black',
            alpha=0.75,
            linewidth=1.0
        )

        axd[f'counts_{i+1}'].set_xlim(-10, 65)
        axd[f'counts_{i+1}'].set_xticks([0, 30, 60], [])
        axd[f'counts_{i+1}'].set_ylim(0, np.quantile(fig_data.layers['denoised'][:, i], 0.995))
        axd[f'counts_{i+1}'].axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black')
        axd[f'counts_{i+1}'].tick_params(labelsize=8)
        
        axd[f'counts_{i+1}'].set_title(
            gene_labels[i], 
            size=8,
            fontdict={'fontweight': 'bold', 'fontstyle': 'italic'}
        )

        fig_refs[f'velocity_{i+1}'] = axd[f'velocity_{i+1}'].scatter(
            x=fig_data.obs[time_key][overplot_shuffle], 
            y=fig_data.layers['velocity'][overplot_shuffle, i],
            c=color_data[overplot_shuffle],
            alpha=0.2, 
            s=1
        )

        median_counts, _window_centers = aggregate_sliding_window_times(
            fig_data.layers['velocity'][:, i].reshape(-1, 1),
            fig_data.obs[time_key],
            centers=np.linspace(-10 + 0.5, 60 - 0.5, 70),
            width=1.
        )

        axd[f'velocity_{i+1}'].plot(
            _window_centers, 
            median_counts,
            c='black',
            alpha=0.75,
            linewidth=1.0
        )

        velocity_axes(axd[f'velocity_{i+1}'])
        axd[f'velocity_{i+1}'].set_xlim(-10, 65)
        axd[f'velocity_{i+1}'].set_xticks([0, 30, 60], [])
        axd[f'velocity_{i+1}'].axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black')
        _ylim = np.abs(np.quantile(fig_data.layers['velocity'][:, i], [0.001, 0.999])).max()
        axd[f'velocity_{i+1}'].set_ylim(-1 * _ylim, _ylim)
        axd[f'velocity_{i+1}'].tick_params(labelsize=8)

        for ic, i_decays in zip(
            REP_COLORS,
            [
                fig_data.varm[f'decay_1'][i, :],
                fig_data.varm[f'decay_2'][i, :],
                fig_data.varm[f'decay'][i, :]
            ]
        ):
            axd[f'decay_{i+1}'].plot(
                fig_data.uns['window_times'], 
                _halflife(i_decays),
                marker=".", 
                linestyle='-', 
                linewidth=1.0, 
                markersize=2 if ic == 'black' else 1, 
                c=ic,
                alpha=1 if ic == 'black' else 0.66
            )

        axd[f'decay_{i+1}'].set_xlim(-10, 65)
        axd[f'decay_{i+1}'].set_xticks([0, 30, 60], [0, 30, 60], size=8)
        axd[f'decay_{i+1}'].set_xlabel("Time (minutes)", size=8)
        axd[f'decay_{i+1}'].set_xlim(-10, 65)
        axd[f'decay_{i+1}'].set_ylim(0, 50)
        axd[f'decay_{i+1}'].axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black')
        axd[f'decay_{i+1}'].tick_params(labelsize=8)

    axd[f'counts_1'].set_ylabel(
        "RNA Expression\n(Counts)", size=8
    )
    axd[f'velocity_1'].set_ylabel(
        "RNA Velocity\n(Counts/minute)", size=8
    )
    axd[f'decay_1'].set_ylabel(
        "RNA Half-life\n(minutes)", size=8
    )
    axd[f'counts_1'].set_title("A", loc='left', x=-0.4, y=0.9, weight='bold')
    axd[f'velocity_1'].set_title("B", loc='left', x=-0.4, y=0.9, weight='bold')
    axd[f'decay_1'].set_title("C", loc='left', x=-0.4, y=0.9, weight='bold')

    axd['legend'].imshow(plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME), aspect='equal')
    axd['legend'].axis('off')

    axd['elegend'].imshow(plt.imread(FIG_EXPT_LEGEND_VERTICAL_FILE_NAME), aspect='equal')
    axd['elegend'].axis('off')
        
    return fig


def _get_fig4_data(data_obj, genes=None):
    
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
        obs = data_obj.all_data.obs[['Pool', 'Experiment', 'Gene', 'program_0_time', 'program_1_time']],
        dtype=np.float32
    )

    fig3_data.var_names=FIGURE_4_GENES
    
    fig3_data.uns['window_times'] = data_obj.all_data.uns['rapamycin_window_decay']['times']

    fig3_data.layers['velocity'] = np.full(fig3_data.X.shape, np.nan, dtype=np.float32)
    fig3_data.layers['denoised'] = np.full(fig3_data.X.shape, np.nan, dtype=np.float32)
    fig3_data.varm['decay'] = data_obj.all_data.varm['rapamycin_window_decay'][_var_idx, :]

    for k in data_obj.expts:

        _idx = data_obj._all_data_expt_index(*k)

        _vdata = data_obj.decay_data(*k)
        
        fig3_data.layers['velocity'][_idx, :] = _vdata.layers[RAPA_VELO_LAYER][:, _var_idx]
        
        if k[1] == "WT":
            fig3_data.varm[f'decay_{k[0]}'] =  _vdata.varm['rapamycin_window_decay'][_var_idx, :]
       
        del _vdata

        fig3_data.layers['denoised'][_idx, :] = data_obj.denoised_data(*k).X[:, _var_idx]

    fig3_data = fig3_data[fig3_data.obs['Gene'] == "WT", :].copy()
    
    return fig3_data
