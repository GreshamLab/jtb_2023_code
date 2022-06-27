import itertools
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

from scipy.cluster.hierarchy import linkage, dendrogram
from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *
import numpy as np

from inferelator_velocity.plotting.program_times import program_time_summary


def figure_2_supplement_1_plot(data, save=True):

    ### BUILD PLOT ###
    fig_refs = {}

    layout = [['pc12_1_cc', 'pc12_1_t', '.', 'pc12_2_cc', 'pc12_2_t', '.'],
              ['pc13_1_cc', 'pc13_1_t', '.', 'pc13_2_cc', 'pc13_2_t', 't_cbar'],
              ['pc14_1_cc', 'pc14_1_t', '.', 'pc14_2_cc', 'pc14_2_t', 't_cbar'],
              ['pc23_1_cc', 'pc23_1_t', '.', 'pc23_2_cc', 'pc23_2_t', 'cc_cbar'],
              ['pc24_1_cc', 'pc24_1_t', '.', 'pc24_2_cc', 'pc24_2_t', 'cc_cbar'],
              ['pc34_1_cc', 'pc34_1_t', '.', 'pc34_2_cc', 'pc34_2_t', '.']]

    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[1, 1, 0.1, 1, 1, 0.1], 
                                                   height_ratios=[1, 1, 1, 1, 1, 1],
                                                   wspace=0, hspace=0.01), 
                                  figsize=(6, 9), dpi=300,
                                  constrained_layout=True)

    for i in range(1,3):
        for j, k in itertools.combinations(range(1,5), 2):
            comp_str = str(j) + ',' + str(k)
            for ak, c, palette in [("_cc", "CC", cc_palette()), ("_t", "Pool", pool_palette())]:
                ax_key = 'pc' + str(j) + str(k) + "_" + str(i) + ak
                fig_refs[ax_key] = sc.pl.pca(data.expt_data[(i, "WT")], ax=axd[ax_key], components=comp_str,
                                             color=c, palette=palette, title=None,
                                             show=False, alpha=0.25, size=2, legend_loc='none',
                                             annotate_var_explained=True)
                axd[ax_key].set_title("")
                if ak == "_t":
                    axd[ax_key].set_ylabel("")

    axd['pc12_1_cc'].set_title("Rep. 1")
    axd['pc12_2_cc'].set_title("Rep. 2")
    axd['pc12_1_t'].set_title("Rep. 1")
    axd['pc12_2_t'].set_title("Rep. 2")

    axd['cc_cbar'].axis('off')
    fig_refs['cc_cbar'] = add_legend(axd['cc_cbar'], 
                                     cc_palette(), 
                                     CC_COLS,
                                     title="Cell Cycle")

    axd['t_cbar'].axis('off')
    fig_refs['t_cbar'] = add_legend(axd['t_cbar'], 
                                    pool_palette(), 
                                    data.all_data.obs['Pool'].dtype.categories.values,
                                    title="Time")

    if save:
        fig.savefig(FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_1.png", facecolor="white", bbox_inches='tight')
    
    return fig


def figure_2_supplement_2_plot(adata, save=True):
    
    _ami_idx = np.array(
        dendrogram(
            linkage(
                adata.uns['programs']['mutual_information'], 
                metric='correlation'), 
            no_plot=True
        )['leaves']
    )

    layout = [['matrix_1', 'matrix_1_cbar', '.', 'matrix_2', 'matrix_2_cbar'],
              ['matrix_3', 'matrix_3_cbar', '.', 'matrix_4', 'matrix_4_cbar']]

    fig_refs = {}

    panel_labels = {'matrix_1': "A",
                    'matrix_2': "B",
                    'matrix_3': "C",
                    'matrix_4': "D"}

    # Title, Metric, VMIN, VMAX, CBAR
    metrics = {
        'matrix_1': ("Information", 'information', 0, 1, 'magma_r'),
        'matrix_2': ("Cosine", 'cosine', 0, 2, 'magma_r'),
        'matrix_3': ("Euclidean", 'euclidean', 0, int(np.quantile(adata.uns['programs']['euclidean_distance'], 0.95)), 'magma_r'),
        'matrix_4': ("Manhattan", 'manhattan', 0, int(np.quantile(adata.uns['programs']['manhattan_distance'], 0.95)), 'magma_r'),
    }

    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[1, 0.1, 0.2, 1, 0.1], 
                                                   height_ratios=[1, 1],
                                                   wspace=0.05, hspace=0.25), 
                                  figsize=(6, 6), dpi=600)


    for ax_ref, (title, metric, vmin, vmax, cbar_name) in metrics.items():

        plot_heatmap(
            fig,
            adata.uns['programs'][f'{metric}_distance'][_ami_idx, :][:, _ami_idx],
            cbar_name,
            axd[ax_ref],
            colorbar_ax=axd[ax_ref + "_cbar"],
            vmin=vmin, vmax=vmax
        )

        axd[ax_ref].set_title(title)
        axd[ax_ref].set_xlabel('Genes')
        axd[ax_ref].set_ylabel('Genes')

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold')
        
    if save:
        fig.savefig(FIGURE_2_SUPPLEMENTAL_FILE_NAME + "_2.png", facecolor="white")
    
    return fig

def figure_2_supplement_3_10_plot(data, cc_program='1', rapa_program='0', save=True):

    figs = []
    
    for i, j in zip(range(3, 11, 2), [(1, "WT"), (2, "WT"), (1, "fpr1"), (2, "fpr1")]):

        _layout = [['pca1', 'M-G1 / G1', 'G1 / S'], ['pca2', 'S / G2', 'G2 / M'], ['hist', 'M / M-G1', 'cbar']]

        fig, ax = plt.subplot_mosaic(_layout,
                                     gridspec_kw=dict(width_ratios=[1] * len(_layout[0]),
                                                      height_ratios=[1, 1, 1],
                                                      wspace=0.25, hspace=0.35),
                                     figsize=(6, 8), dpi=300)

        program_time_summary(
            data.expt_data[j],
            cc_program,
            cluster_order = CC_COLS,
            cluster_colors = {k: v for k, v in zip(CC_COLS, cc_palette())},
            cbar_title='Phase',
            ax=ax,
            wrap_time=88,
            alpha=0.1 if j[1] == "WT" else 0.5
        )

        fig.suptitle(f"Cell Cycle Program Replicate {j[0]} [{j[1]}]")
        
        if save:
            fig.savefig(FIGURE_2_SUPPLEMENTAL_FILE_NAME + f"_{i}.png", facecolor="white")
            
        figs.append(fig)

        _layout = [['pca1', '12 / 3', '3 / 4'], ['pca2', '4 / 5', '5 / 6'], ['hist', '6 / 7' , '7 / 8'], ['cbar', 'cbar', 'cbar']]

        fig, ax = plt.subplot_mosaic(_layout,
                                     gridspec_kw=dict(width_ratios=[1] * len(_layout[0]),
                                                      height_ratios=[1, 1, 1, 0.2],
                                                      wspace=0.25, hspace=0.35),
                                     figsize=(6, 8), dpi=300)

        program_time_summary(
            data.expt_data[j],
            rapa_program,
            ax=ax,
            cluster_order = ['12', '3', '4', '5', '6', '7', '8'],
            cluster_colors = {k: v for k, v in zip(['12', '3', '4', '5', '6', '7', '8'], pool_palette()[1:])},
            cbar_title='Time [Groups]',
            cbar_horizontal=True,
            time_limits=(-25, 70),
            alpha=0.1 if j[1] == "WT" else 0.5
        )

        fig.suptitle(f"Rapamycin Response Program Replicate {j[0]} [{j[1]}]")
        
        if save:
            fig.savefig(FIGURE_2_SUPPLEMENTAL_FILE_NAME + f"_{i + 1}.png", facecolor="white")
        
        figs.append(fig)
        
    return figs
        