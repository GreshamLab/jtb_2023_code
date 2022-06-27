import itertools
import matplotlib
import matplotlib.pyplot as plt

from jtb_2022_code.utils.pseudotime_common import do_pca_pt, spearman_rho_pools
from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *

import scanpy as sc
import numpy as np


def figure_3_supplement_1_plot(data, save=True):
    
    def _pt_rho_heatmap(data, key, expt=1):
        dpt = data.all_data.uns[key].T.unstack().T
        dpt = dpt.reindex(N_PCS.astype(str), axis=1)
        dpt = dpt.reindex(N_NEIGHBORS.astype(str)[::-1], axis=0, level=2) 
        return dpt.loc[(expt, "WT"), :].applymap(np.abs)

    panel_labels = {'dpt_rho_1': "A",
                    'cellrank_rho_1': "B",
                    'monocle_rho_1': "C",
                    'palantir_rho_1': "D"}
    
    panel_titles = {'dpt_rho_1': "Rep. 1",
                    'dpt_rho_2': "Rep. 2"}

    layout = [['dpt_rho_1', 'dpt_rho_2', '.'],
              ['cellrank_rho_1', 'cellrank_rho_2', 'cbar'],
              ['monocle_rho_1', 'monocle_rho_2', 'cbar'],
              ['palantir_rho_1', 'palantir_rho_2', '.']]

    fig_refs = {}

    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[1, 1, 0.05], 
                                                   height_ratios=[1, 1, 1, 1],
                                                   wspace=0.1, hspace=0.1),
                                  figsize=(6, 9), dpi=300)

    for pt, pt_key in [('Diffusion PT', 'dpt_rho'), 
                       ('Cellrank PT', 'cellrank_rho'),
                       ('Monocle PT', 'monocle_rho'),
                       ('Palantir PT', 'palantir_rho')]:

        _bottom = pt_key == "palantir_rho"
        for i in range(1, 3):
            _left = i == 1
            hm_data = _pt_rho_heatmap(data, pt_key, expt=i)
            ax_key = pt_key + "_" + str(i)
            fig_refs[ax_key] = axd[ax_key].imshow(hm_data, vmin=0.75, vmax=1.0,
                                                  cmap='plasma', aspect='auto', interpolation='nearest')

            if _left:
                axd[ax_key].set_yticks(range(hm_data.shape[0]), labels=hm_data.index)
                axd[ax_key].set_ylabel(pt + "\n # Neighbors")
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
                    n = hm_data.iloc[y, x]
                    if np.isnan(n):
                        continue
                    axd[ax_key].text(x, y, '%.2f' % n, 
                                     horizontalalignment='center', 
                                     verticalalignment='center',
                                     size=4
                             )

    for ax_key, title_str in panel_titles.items():
        axd[ax_key].set_title(title_str)

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold', x=-0.3, y=0.99)

    fig_refs['cbar'] = fig.colorbar(fig_refs['dpt_rho_1'], cax=axd['cbar'], orientation="vertical",aspect=60)
    fig_refs['cbar'].ax.set_title('ρ')
    
    if save:
        fig.savefig(FIGURE_3_1_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white', bbox_inches='tight')
    
    return fig