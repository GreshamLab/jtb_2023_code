from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import anndata as ad
import numpy as np
import pandas as pd
import math 

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def plot_figure_2(data, save=True):

    CATEGORY_COLORS = ["gray", "skyblue", "lightgreen"]
    PROGRAM_COLORS = [colors.rgb2hex(plt.get_cmap(PROGRAM_PALETTE)(k)) for k in range(len(data.all_data.var['program'].cat.categories))]

    joint_colormap = colors.ListedColormap(CATEGORY_COLORS + PROGRAM_COLORS)
    
    _ami_linkage = linkage(squareform(data.all_data.uns['programs']['information_distance'], checks=False))
    _ami_dendrogram = dendrogram(_ami_linkage, no_plot=True)
    _ami_idx = np.array(_ami_dendrogram['leaves'])
    _ami_distances = data.all_data.uns['programs']['information_distance'][_ami_idx, :][:, _ami_idx]
    _ami_information = data.all_data.uns['programs']['mutual_information'][_ami_idx, :][:, _ami_idx]

    layout = [['schematic', 'schematic', 'schematic', 'schematic', 'schematic'],
              ['rows', 'matrix', 'matrix', 'matrix', 'dendro'],
              ['.', '.', '.', '.', '.'],
              ['.', 'rep_t', '.', 'rep_cc', 'cc_cbar'],
              ['.', '.', '.', '.', '.'],
              ['t_cbar', 't_cbar', 't_cbar', 't_cbar', '.']]

    fig_refs = {}

    panel_labels = {
        'schematic': "A",
        'rows': "B",
        'rep_t': "C"
    }
    
    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[0.15, 1, 0.1, 1, 0.2], 
                                                   height_ratios=[1, 2, 0.25, 1, 0.1, 0.2],
                                                   wspace=0.01, hspace=0.2), 
                                  figsize=(4, 7), dpi=600)
    
    #axd['schematic'].set_position(Bbox([[0.116, 0.70], [0.855, 0.88]]))
    axd['schematic'].imshow(plt.imread(FIG2A_FILE_NAME), aspect='equal')
    axd['schematic'].axis('off')

    _gene_order = data.all_data.uns['programs']['metric_genes'][_ami_idx]

    _cat_series = data.all_data.var.loc[_gene_order, 'category'].cat.codes
    _prog_series = data.all_data.var.loc[_gene_order, 'program'].astype(int) + _cat_series.max() + 1

    _rcolors = pd.DataFrame([_cat_series, _prog_series]).T.loc[_gene_order, :]

    fig_refs.update(plot_heatmap(
        fig,
        _ami_distances,
        'magma_r',
        axd['matrix'],
        _ami_information,
        _ami_linkage,
        axd['dendro'],
        row_data=_rcolors,
        row_cmap=joint_colormap,
        row_ax=axd['rows'],
        row_xlabels=["Cat.", "Prog."],
        colorbar_label="Info Dist.",
        colorbar_loc=[0.9, 0.57, 0.02, 0.15],
        vmin=0, vmax=1
    ))

    axd['matrix'].set_xlabel("Genes")

    for i in range(1, 3):
        expt_ref = data.expt_data[(i, "WT")]

        rgen = np.random.default_rng(123)
        overplot_shuffle = np.arange(expt_ref.X.shape[0])
        rgen.shuffle(overplot_shuffle)

        _time_x = expt_ref.obs['program_rapa_time'].values[overplot_shuffle]
        _time_y = expt_ref.obs['program_cc_time'].values[overplot_shuffle]

        axd[f'rep_t'].scatter(_time_x, _time_y, 
                              c=to_pool_colors(expt_ref.obs['Pool'])[overplot_shuffle], 
                              s=1, alpha=0.1)

        axd[f'rep_cc'].scatter(_time_x, _time_y,
                               c=to_cc_colors(expt_ref.obs['CC'])[overplot_shuffle], 
                               s=1, alpha=0.1)

        
    #axd[f'rep_t'].set_xlabel(f"Rapamycin Reponse Timaxd[f'rep_cc'].e [min]")
    axd[f'rep_t'].set_ylabel(f"Cell Cycle Time\n[min]")
    fig.text(0.5, 0.16, 'Rapamycin Response Time [min]', ha='center', va='center')
    axd[f'rep_t'].set_xlim(-10, 60)
    axd[f'rep_t'].set_ylim(0, 88)

    axd[f'rep_cc'].set_xlim(-10, 60)
    axd[f'rep_cc'].set_ylim(0, 88)
    axd[f'rep_cc'].set_yticklabels([])
    axd[f'rep_cc'].annotate(f"n = {np.sum(data.all_data.obs['Gene'] == 'WT')}", xy=(5, 0.2),  
                            xycoords='data', xytext=(0.1, 0.05), 
                            textcoords='axes fraction')

    add_legend(axd['t_cbar'], pool_palette(), list(range(1, 9)), fontsize='x-small', horizontal=True)
    add_legend(axd['cc_cbar'], cc_palette(), CC_COLS, title="Phase", fontsize='x-small')

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold')

    if save:
        fig.savefig(FIGURE_2_FILE_NAME + ".png", facecolor="white", bbox_inches='tight')
    
    return fig
