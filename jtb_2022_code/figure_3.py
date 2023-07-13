import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *
from jtb_2022_code.utils.decay_common import _halflife

from inferelator_velocity.utils.aggregation import aggregate_sliding_window_times

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

REP_COLORS = ['darkslateblue', 'darkgoldenrod', 'black']


def figure_3_plot(data, save=True):
    
    fig_3_data = _get_fig3_data(data)
    color_vec = to_pool_colors(fig_3_data.obs['Pool'])
    
    fig = _fig3_plot(
        fig_3_data,
        color_vec,
        FIGURE_4_GENES,
        gene_labels=[data.gene_common_name(g) for g in FIGURE_4_GENES],
        time_key=f"program_{data.all_data.uns['programs']['rapa_program']}_time"
    )
    
    if save:
        fig.savefig(FIGURE_3_FILE_NAME + ".png", facecolor="white")

    return fig, fig_3_data


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
            centers=np.linspace(-10 + 0.5, 65 - 0.5, 75),
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


def _get_fig3_data(data_obj, genes=None):
    
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
