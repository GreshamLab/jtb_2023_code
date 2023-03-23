from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import anndata as ad
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from inferelator_velocity.utils.keys import (
    PROGRAM_KEY
)

def plot_figure_2(data, save=True):
    
    fig = plt.figure(figsize=(5, 3.5), dpi=MAIN_FIGURE_DPI)

    _bottom_h = 0.27
    _bottom_y = 0.15
    _top_h = 0.325
    _top_y = 0.59
    _pt_hist_left = 0.3
    _pt_hist_width = 0.6

    axd = {
        'rows': fig.add_axes([0.01, _top_y, 0.025, _top_h]),
        'matrix': fig.add_axes([0.035, _top_y, 0.22, _top_h]),
        'info_cbar': fig.add_axes([0.265, _top_y, 0.015,_top_h]),
        'rep_t': fig.add_axes([0.415, _top_y, 0.23, _top_h]),
        'rep_cc': fig.add_axes([0.67, _top_y, 0.23, _top_h]),
        'method_hm': fig.add_axes([0.02, _bottom_y, 0.12, _bottom_h]),
        'pseudotime_pca': fig.add_axes([_pt_hist_left, _bottom_y + _bottom_h / 3 * 2, _pt_hist_width, _bottom_h / 3]),
        'pseudotime_dpt': fig.add_axes([_pt_hist_left, _bottom_y + _bottom_h / 3, _pt_hist_width, _bottom_h / 3]),
        'pseudotime_palantir': fig.add_axes([_pt_hist_left, _bottom_y, _pt_hist_width, _bottom_h / 3]),
        'legend_rapa': fig.add_axes([0.89, 0.1 , 0.1, 0.4]),
        'legend_cc': fig.add_axes([0.91, 0.55, 0.1, 0.4])
    }

    PROGRAM_PALETTE = 'tab10'

    CATEGORY_COLORS = ["gray", "skyblue", "lightgreen"]
    PROGRAM_COLORS = [
        colors.rgb2hex(plt.get_cmap(PROGRAM_PALETTE)(k))
        for k in range(len(data.all_data.var[PROGRAM_KEY].cat.categories))
    ]

    joint_colormap = colors.ListedColormap(CATEGORY_COLORS + PROGRAM_COLORS)

    _ami_linkage = linkage(squareform(data.all_data.uns[PROGRAM_KEY]['information_distance'], checks=False))
    _ami_dendrogram = dendrogram(_ami_linkage, no_plot=True)
    _ami_idx = np.array(_ami_dendrogram['leaves'])
    _ami_distances = data.all_data.uns[PROGRAM_KEY]['information_distance'][_ami_idx, :][:, _ami_idx]
    _ami_information = data.all_data.uns[PROGRAM_KEY]['mutual_information'][_ami_idx, :][:, _ami_idx]

    fig_refs = {}

    panel_labels = {
        'rows': ("A", 0.0),
        'rep_t': ("B", -0.4),
        'method_hm': ("C", 0.0),
        'pseudotime_pca': ("D", 0.0)
    }

    hm_labels = {
        'time': 'Time [min]', 
        'pca': 'PCA', 
        'dpt': 'DPT',
        'palantir': 'Palantir',
        'cellrank': 'CellRank', 
        'monocle': 'Monocle3'
     }

    _gene_order = data.all_data.uns[PROGRAM_KEY]['metric_genes'][_ami_idx]

    _cat_series = data.all_data.var.loc[_gene_order, 'category'].cat.codes
    _prog_series = data.all_data.var.loc[_gene_order, PROGRAM_KEY].astype(int) + _cat_series.max() + 1

    _rcolors = pd.DataFrame([_cat_series, _prog_series]).T.loc[_gene_order, :]

    fig_refs.update(plot_heatmap(
        fig,
        _ami_distances,
        'magma_r',
        axd['matrix'],
        _ami_information,
        _ami_linkage,
        None,
        row_data=_rcolors,
        row_cmap=joint_colormap,
        row_ax=axd['rows'],
        row_xlabels=["Cat.", "Prog."],
        colorbar_label=None,
        vmin=0, vmax=1,
        colorbar_ax=axd['info_cbar']
    ))

    axd['matrix'].set_xlabel("Genes", size=8)
    axd['matrix'].set_title("Information\nDistance", size=8)

    _time_x = np.concatenate([
        data.expt_data[(i, "WT")].obs['program_rapa_time'].values
        for i in range(1, 3)
    ])

    _time_y = np.concatenate([
        data.expt_data[(i, "WT")].obs['program_cc_time'].values
        for i in range(1, 3)
    ])

    _color_pool = to_pool_colors(pd.concat([
        data.expt_data[(i, "WT")].obs['Pool']
        for i in range(1, 3)
    ]))

    _color_cc = to_cc_colors(pd.concat([
        data.expt_data[(i, "WT")].obs['CC']
        for i in range(1, 3)
    ]))

    rgen = np.random.default_rng(123)
    overplot_shuffle = np.arange(len(_time_x))
    rgen.shuffle(overplot_shuffle)

    axd[f'rep_t'].scatter(
        _time_x[overplot_shuffle],
        _time_y[overplot_shuffle], 
        c=_color_pool[overplot_shuffle], 
        s=1,
        alpha=0.2
    )

    axd[f'rep_cc'].scatter(
        _time_x[overplot_shuffle],
        _time_y[overplot_shuffle], 
        c=_color_cc[overplot_shuffle], 
        s=1,
        alpha=0.2
    )

    axd[f'rep_t'].set_ylabel(f"Cell Cycle [min]", size=8)
    axd[f'rep_t'].set_title("Collection Time", size=8)
    axd[f'rep_t'].set_xlabel('Rapamycin Response [min]', labelpad=-1, size=8, x=1)
    axd[f'rep_t'].set_xlim(-10, 60)
    axd[f'rep_t'].set_ylim(0, CC_LENGTH)
    axd[f'rep_t'].set_yticks([0, int(CC_LENGTH / 2), CC_LENGTH])
    axd[f'rep_t'].tick_params(labelsize=8)
    axd[f'rep_t'].axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black', alpha=0.5)

    axd[f'rep_cc'].set_title("Cell Cycle Phase", size=8)
    axd[f'rep_cc'].set_xlim(-10, 60)
    axd[f'rep_cc'].set_ylim(0, CC_LENGTH)
    axd[f'rep_cc'].set_yticks([0, int(CC_LENGTH / 2), CC_LENGTH])
    axd[f'rep_cc'].set_yticklabels([])
    axd[f'rep_cc'].tick_params(labelsize=8)
    axd[f'rep_cc'].annotate(f"n = {np.sum(data.all_data.obs['Gene'] == 'WT')}", xy=(5, 0.2),  
                            xycoords='data', xytext=(0.35, 0.05), 
                            textcoords='axes fraction', fontsize='small')
    axd[f'rep_cc'].axvline(0, 0, 1, linestyle='--', linewidth=1.0, c='black', alpha=0.5)

    fig_refs['time_hist'] = plt_time_histogram(
        data,
        [axd['pseudotime_pca'], axd['pseudotime_dpt'], axd['pseudotime_palantir']],
        ['pca', 'dpt', 'palantir'],
        labels = ["PCA", "DPT", "Palantir"]
    )
    
    axd['pseudotime_palantir'].tick_params(labelsize=8)
    axd['pseudotime_palantir'].set_xlabel("Pseudotime", size=8)

    hm_data = _make_method_heatmap_data(data).reindex(hm_labels.keys())

    fig_refs['method_hm'] = axd['method_hm'].imshow(
        hm_data, 
        vmin=0.75, vmax=1.0,
        cmap='plasma', aspect='auto', 
        interpolation='nearest', alpha=0.75
    )

    axd['method_hm'].set_yticks(range(hm_data.shape[0]), labels=hm_data.index.map(lambda x: hm_labels[x]), size=8)
    axd['method_hm'].set_xticks(range(hm_data.shape[1]), labels=[1, 2], size=8)
    axd['method_hm'].set_xlabel("Expt.\nReplicate", size=8)

    axd['method_hm'].yaxis.tick_right()

    # https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib
    for y in range(hm_data.shape[0]):
        for x in range(hm_data.shape[1]):
            n = hm_data.iloc[y, x]
            if np.isnan(n):
                continue
            axd['method_hm'].text(
                x, y, '%.2f' % n, 
                horizontalalignment='center',
                verticalalignment='center', 
                size=7
            )

    axd['legend_rapa'].imshow(plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME), aspect='equal')
    axd['legend_rapa'].axis('off')

    axd['legend_cc'].imshow(plt.imread(FIG_CC_LEGEND_VERTICAL_FILE_NAME), aspect='equal')
    axd['legend_cc'].axis('off')

    for ax_id, (label, offset) in panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold', x=offset)

    if save:
        fig.savefig(FIGURE_2_FILE_NAME + ".png", facecolor="white")
    
    return fig

def plt_time_histogram(data_obj, axs, keys, bins=80, labels=None):
    
    if labels is None:
        labels = keys
    
    hist_limit, fref = [], []
    hist_labels = np.arange(bins)
    
    data = data_obj.all_data.obs[["Pool"]].copy()
    ideal_rhos = data_obj.optimal_pseudotime_rho
    
    pt = data_obj.pseudotime
    ptr = data_obj.pseudotime_rho.loc[:, (slice(None), False, slice(None))]

    # Transpose, select WT, and drop indices into columns
    ptr = ptr.T.loc[:, (slice(None), "WT")].reset_index().droplevel(1, axis=1)
    ptr['mean_rho'] = ptr[[1, 2]].mean(axis=1)
    
    for k in keys:
        rhos = ptr.loc[ptr['method'] == k, :]
        data[k] = pt.loc[:, (k, False, ideal_rhos.loc[k, 'values'])].values
        
    data = data.loc[data_obj.all_data.obs["Gene"] == "WT", :].copy()

    for i, (k, ax) in enumerate(zip(keys, axs)):
       
        bottom_line = None
        for j, hist_data in enumerate(_get_pt_hist(data, bins=bins, key=k)):
            
            bottom_line = np.zeros_like(hist_data) if bottom_line is None else bottom_line
                                
            fref.append(
                ax.bar(
                    hist_labels, 
                    hist_data, 
                    bottom=bottom_line, 
                    width=0.5, 
                    label=i,
                    color=pool_palette()[j]
                )
            )
                        
            bottom_line = bottom_line + hist_data
            
            hist_limit.append(np.max(np.abs(bottom_line)))

    hist_limit = max(hist_limit)
    
    for i, (k, ax) in enumerate(zip(keys, axs)):
        
        if i + 1 == len(axs):
            ax.set_xticks(np.arange(0, 11, 2) * int(bins / 10))
            ax.set_xticklabels(np.arange(0, 11, 2) / 10, rotation=90)
        else:
            ax.set_xticks([], [])

        ax.set_yticks([], [])
        ax.set_ylim(0, hist_limit)
        ax.set_xlabel("Pseudotime")
        
        if labels is not None:
            ax.annotate(
                f"{labels[i]}",
                xy=(80, 500),
                xycoords='data',
                xytext=(0.65, 0.55),
                textcoords='axes fraction',
                size=8
            )

        ax.axhline(0, 0, 1, linestyle='-', linewidth=1.0, c='black')
        ax.set_xlim(0, bins)
                        
    return fref

def _get_pt_hist(data_df, bins=80, key='time_pca_pt'):
    cuts = np.arange(bins + 1) / bins
    return [np.bincount(pd.cut(data_df.loc[data_df['Pool'] == x, key], 
                               cuts, labels=np.arange(bins)).dropna(),
                        minlength=bins) for x in range(1, 9)]

def _make_method_heatmap_data(pdata):
    return pdata.max_pseudotime_rho.loc[
        (slice(None), "WT"),
        (slice(None), False)
    ].melt(
        ignore_index=False
    ).reset_index(
    ).drop(
        ["Gene", "denoised"],
        axis=1
    ).pivot_table(
        index="method",
        columns="Experiment",
        values='value'
    )
