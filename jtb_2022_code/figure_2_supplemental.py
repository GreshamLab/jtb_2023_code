import itertools
import matplotlib
import matplotlib.pyplot as plt

from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *

import scanpy as sc
import numpy as np

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
                                  figsize=(6, 10), dpi=300,
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

    axd['pc12_1_cc'].set_title("Experiment 1")
    axd['pc12_2_cc'].set_title("Experiment 2")
    axd['pc12_1_t'].set_title("Experiment 1")
    axd['pc12_2_t'].set_title("Experiment 2")

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
        fig.savefig(FIGURE_2_1_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white')
    
    return fig


def figure_2_supplement_2_plot(data, save=True):
    
    def _pt_rho_heatmap(data, key, expt=1):
        dpt = data.all_data.uns[key].T.unstack().T
        dpt = dpt.reindex(N_PCS.astype(str), axis=1)
        dpt = dpt.reindex(N_NEIGHBORS.astype(str)[::-1], axis=0, level=2) 
        return dpt.loc[(expt, "WT"), :].applymap(np.abs)

    panel_titles = {'dpt_rho_1': "Experiment 1",
                    'dpt_rho_2': "Experiment 2"}

    layout = [['dpt_rho_1', 'dpt_rho_2', '.'],
              ['cellrank_rho_1', 'cellrank_rho_2', 'cbar'],
              ['monocle_rho_1', 'monocle_rho_2', 'cbar'],
              ['palantir_rho_1', 'palantir_rho_2', '.']]

    fig_refs = {}

    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[1, 1, 0.05], 
                                                   height_ratios=[1, 1, 1, 1],
                                                   wspace=0.1, hspace=0.1),
                                  figsize=(6, 10), dpi=300)

    for pt, pt_key in [('Diffusion PT (ρ)', 'dpt_rho'), 
                       ('Cellrank PT (ρ)', 'cellrank_rho'),
                       ('Monocle PT (ρ)', 'monocle_rho'),
                       ('Palantir PT (ρ)', 'palantir_rho')]:

        _bottom = pt_key == "palantir_rho"
        for i in range(1, 3):
            _left = i == 1
            hm_data = _pt_rho_heatmap(data, pt_key, expt=i)
            ax_key = pt_key + "_" + str(i)
            fig_refs[ax_key] = axd[ax_key].imshow(hm_data, vmin=0.75, vmax=1.0,
                                                  cmap='plasma', aspect='auto', interpolation='nearest')

            if _left:
                axd[ax_key].set_yticks(range(hm_data.shape[0]), labels=hm_data.index)
                axd[ax_key].set_ylabel(pt + "\nNeighbors")
            else:
                axd[ax_key].set_yticks([], labels=[])

            if _bottom:
                axd[ax_key].set_xticks(range(hm_data.shape[1]), labels=hm_data.columns, rotation=90, ha="center")
                axd[ax_key].set_xlabel("PCs") 
            else:
                axd[ax_key].set_xticks([], labels=[])

            # https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib
            for y in range(hm_data.shape[0]):
                for x in range(hm_data.shape[1]):
                    axd[ax_key].text(x, y, '%.2f' % hm_data.iloc[y, x], 
                                     horizontalalignment='center', 
                                     verticalalignment='center',
                                     size=4
                             )

    for ax_key, title_str in panel_titles.items():
        axd[ax_key].set_title(title_str)


    fig.colorbar(fig_refs['dpt_rho_1'], cax=axd['cbar'], orientation="vertical",aspect=60)
    
    if save:
        fig.savefig(FIGURE_2_2_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white')
    
    return fig