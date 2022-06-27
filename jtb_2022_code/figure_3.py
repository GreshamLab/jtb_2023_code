import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox

from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.utils.adata_common import *
from jtb_2022_code.utils.pseudotime_common import do_pca_pt, spearman_rho_pools

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

def figure_3_plot(data, save=True):

    panel_labels = {'pca_1': "A",
                    'pca_pt_1': "B",
                    'pca_d1': "C",
                    'pca_pt_d1': "D",
                    'method_hm': "E",
                    'time_hist': "F"}

    panel_titles = {'pca_1': "Rep. 1",
                    'pca_2': "Rep. 2",
                    'pca_pt_1': "Rep. 1",
                    'pca_pt_2': "Rep. 2",
                    'pca_d1': "",
                    'pca_d2': "",
                    'pca_pt_d1': "",
                    'pca_pt_d2': "",}

    layout = [['pca_1', 'pca_2', '.', 'pca_pt_1', 'pca_pt_2', 'pca_lgd'],
              ['.', '.', '.', '.', '.', 'pca_lgd'],
              ['pca_d1', 'pca_d2', '.', 'pca_pt_d1', 'pca_pt_d2', 'pca_lgd'],
              ['.', '.', '.', '.', '.', 'pca_lgd'],
              ['method_hm', 'time_hist', 'time_hist', 'time_hist', 'time_hist', 'pca_lgd']]

    hm_labels = {'time': 'Time [min]', 
                 'pca': 'PCA', 
                 'dpt': 'DPT',
                 'palantir': 'Palantir',
                 'cellrank': 'CellRank', 
                 'monocle': 'Monocle3'
                 }

    fig_refs = {}

    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[1, 1, 0.02, 0.85, 0.85, 0.05], 
                                                   height_ratios=[1, 0.05, 1, 0.2, 1],
                                                   wspace=0.4, hspace=0.4), 
                                  figsize=(7, 6), dpi=300)

    axd["pca_lgd"].axis('off')




    for i in range(1,3):
        pca_key = "pca_" + str(i)
        pca_d_key = "pca_d" + str(i)

        pt_key = "pca_pt_" + str(i)
        pt_d_key = "pca_pt_d" + str(i)

        pca_data = data.expt_data[i, "WT"]
        pca_d_data = get_clean_anndata(pca_data)
        pca_d_data.obsm['X_pca'] = pca_data.obsm['DEWAKSS_pca'].copy()
        pca_d_data.uns['pca'] = pca_data.uns['DEWAKSS_pca'].copy()


        ### PANEL A/C ###
        fig_refs[pca_key] = sc.pl.pca(pca_data, ax=axd[pca_key], 
                                      color="Pool", palette=pool_palette(), 
                                      show=False, alpha=0.25, size=2, legend_loc='none',
                                      annotate_var_explained=True)
        fig_refs[pca_d_key] = sc.pl.pca(pca_d_data, ax=axd[pca_d_key], 
                                        color="Pool", palette=pool_palette(),
                                        show=False, alpha=0.25, size=2, legend_loc='none',
                                        annotate_var_explained=True)

        ### PANEL B/D ###
        fig_refs[pt_key] = _plt_pt_violins(pca_data, axd[pt_key], 'pca_pt', include_labels= i == 1)
        fig_refs[pt_d_key] = _plt_pt_violins(pca_d_data, axd[pt_d_key], 'denoised_pca_pt', include_labels= i == 1)


    fig_refs['hist_divider'] = make_axes_locatable(axd['time_hist'])
    fig_refs['hist_leftpad'] = fig_refs['hist_divider'].append_axes('left', size='20%', pad=0.1)
    fig_refs['hist_leftpad'].axis('off')
    fig_refs['time_hist'] = plt_time_histogram(data, axd['time_hist'], ['dpt_pt', 'palantir_pt'], labels = ["DPT", "Palantir"])

    hm_data = _make_method_heatmap_data(data).reindex(hm_labels.keys())
    fig_refs['method_hm'] = axd['method_hm'].imshow(
        hm_data, 
        vmin=0.75, vmax=1.0,
        cmap='plasma', aspect='auto', 
        interpolation='nearest'
    )

    axd['method_hm'].set_yticks(range(hm_data.shape[0]), labels=hm_data.index.map(lambda x: hm_labels[x]))
    axd['method_hm'].set_xticks(range(hm_data.shape[1]), labels=[1, 2, 1, 2])
    axd['method_hm'].xaxis.tick_top()
    axd['method_hm'].yaxis.tick_right()
    axd['method_hm'].axvline(1.5, 0, 1, linestyle='-', linewidth=1.0, c='black')
    axd['method_hm'].annotate(f"Denoised", xy=(4, 0), xycoords='data', xytext=(0.45, 1.32), textcoords='axes fraction', annotation_clip=False)

    divider = make_axes_locatable(axd['method_hm'])
    axd['method_hm_cbar'] = divider.append_axes('bottom', size='10%', pad=0.02)
    fig_refs['method_hm_cbar'] = fig.colorbar(fig_refs['method_hm'], cax=axd['method_hm_cbar'], orientation="horizontal", aspect=80)
    fig_refs['method_hm_cbar'].set_label("Spearman ρ")

    # https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib
    for y in range(hm_data.shape[0]):
        for x in range(hm_data.shape[1]):
            n = hm_data.iloc[y, x]
            if np.isnan(n):
                continue
            axd['method_hm'].text(x, y, '%.2f' % n, 
                                  horizontalalignment='center', 
                                  verticalalignment='center', 
                                  size=4)

    fig_refs['pca_legend'] = add_legend(axd['pca_lgd'], 
                                        pool_palette(), 
                                        data.all_data.obs['Pool'].dtype.categories.values,
                                        title = "Time",
                                        fontsize = 'small')

    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold')

    for ax_id, label in panel_titles.items():
        axd[ax_id].set_title(label)

    if save:
        fig.savefig(FIGURE_3_FILE_NAME + ".png", facecolor="white", bbox_inches='tight') 

    return fig

def _minmax_time(adata, key='time_pca_pt'):
    return adata.obs[key].min(), adata.obs[key].max()


def _get_pt_hist(data_df, bins=80, key='time_pca_pt'):
    cuts = np.arange(bins + 1) / bins
    return [np.bincount(pd.cut(data_df.loc[data_df['Pool'] == x, key], 
                               cuts, labels=np.arange(bins)).dropna(),
                        minlength=bins) for x in range(1, 9)]


def _make_method_heatmap_data(pdata):
    raw = pdata.all_data.uns['rho'].loc[(slice(None), "WT"), :].T
    raw.columns = pd.MultiIndex.from_tuples([(1, "raw"), (2, "raw")])
    
    denoised = pdata.all_data.uns['denoised_rho'].loc[(slice(None), "WT"), :].T.reindex(raw.index)
    denoised.columns = pd.MultiIndex.from_tuples([(1, "denoised"), (2, "denoised")])
    
    return pd.concat((raw, denoised), axis=1)


def _plt_pt_violins(pdata, ax, pt_key, include_labels=True):
    pca_pt_data = {k: v[pt_key] for k, v in pdata.obs[[pt_key, 'Pool']].groupby("Pool")}
    ref = ax.violinplot([pca_pt_data[j] for j in range(1,9)], showmeans=False, showmedians=True, showextrema=False)
    ax.set_xticks(np.arange(8) + 1, labels=np.arange(8) + 1)
    ax.set_yticks([0.0, 0.5, 1.0], labels=[0.0, 0.5, 1.0] if include_labels else ['', '', ''])
    
    if include_labels:
        ax.set_ylabel("PCA1 PT")
    
    rho = spearman_rho_pools(pdata.obs['Pool'], pdata.obs[pt_key])
    ax.annotate("ρ = " + f"{rho:.2f}", xy=(5, 0.2),  xycoords='data', xytext=(0.25, 0.05), textcoords='axes fraction')
    
    for part, c in zip(ref['bodies'], pool_palette()):
        part.set_facecolor(c)
        part.set_edgecolor('black')
        
    return ref

def plt_time_histogram(data_obj, ax, keys, bins=80, labels=None):
    
    if labels is None:
        labels = keys
    
    hist_limit, fref = [], []
    hist_labels = np.arange(bins)
    
    data = pd.concat([data_obj.expt_data[(i, "WT")].obs.loc[:, ["Pool"] + keys] for i in range(1, 3)])
def _minmax_time(adata, key='time_pca_pt'):
    return adata.obs[key].min(), adata.obs[key].max()


def _get_pt_hist(data_df, bins=80, key='time_pca_pt'):
    cuts = np.arange(bins + 1) / bins
    return [np.bincount(pd.cut(data_df.loc[data_df['Pool'] == x, key], 
                               cuts, labels=np.arange(bins)).dropna(),
                        minlength=bins) for x in range(1, 9)]


def _make_method_heatmap_data(pdata):
    raw = pdata.all_data.uns['rho'].loc[(slice(None), "WT"), :].T
    raw.columns = pd.MultiIndex.from_tuples([(1, "raw"), (2, "raw")])
    
    denoised = pdata.all_data.uns['denoised_rho'].loc[(slice(None), "WT"), :].T.reindex(raw.index)
    denoised.columns = pd.MultiIndex.from_tuples([(1, "denoised"), (2, "denoised")])
    
    return pd.concat((raw, denoised), axis=1)


def _plt_pt_violins(pdata, ax, pt_key, include_labels=True):
    pca_pt_data = {k: v[pt_key] for k, v in pdata.obs[[pt_key, 'Pool']].groupby("Pool")}
    ref = ax.violinplot([pca_pt_data[j] for j in range(1,9)], showmeans=False, showmedians=True, showextrema=False)
    ax.set_xticks(np.arange(8) + 1, labels=np.arange(8) + 1)
    ax.set_yticks([0.0, 0.5, 1.0], labels=[0.0, 0.5, 1.0] if include_labels else ['', '', ''])
    
    if include_labels:
        ax.set_ylabel("PCA1 PT")
    
    rho = spearman_rho_pools(pdata.obs['Pool'], pdata.obs[pt_key])
    ax.annotate("ρ = " + f"{rho:.2f}", xy=(5, 0.2),  xycoords='data', xytext=(0.25, 0.05), textcoords='axes fraction')
    
    for part, c in zip(ref['bodies'], pool_palette()):
        part.set_facecolor(c)
        part.set_edgecolor('black')
        
    return ref

def plt_time_histogram(data_obj, ax, keys, bins=80, labels=None):
    
    if labels is None:
        labels = keys
    
    hist_limit, fref = [], []
    hist_labels = np.arange(bins)
    
    data = pd.concat([data_obj.expt_data[(i, "WT")].obs.loc[:, ["Pool"] + keys] for i in range(1, 3)])

    for i, k in enumerate(keys):
        
        bottom_line = None
        for j, hist_data in enumerate(_get_pt_hist(data, bins=bins, key=k)):
            
            bottom_line = np.zeros_like(hist_data) if bottom_line is None else bottom_line
            
            # Flip across the X axis for the second expt
            if i == 1:
                hist_data *= -1
                                
            fref.append(ax.bar(hist_labels, 
                               hist_data, 
                               bottom=bottom_line, 
                               width=0.5, 
                               label=i, color=pool_palette()[j]))
                        
            bottom_line = bottom_line + hist_data
            
            hist_limit.append(np.max(np.abs(bottom_line)))

    hist_limit = max(hist_limit)
    ax.set_xticks(np.arange(11) * int(bins / 10))
    ax.set_xticklabels(np.arange(11) / 10, rotation=90)
    ax.set_yticks([-5000, 0, 5000], labels=[5000, 0, 5000])
    ax.set_ylim(-1 * hist_limit, hist_limit)
    ax.set_ylabel("# Cells")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel("Pseudotime")
    ax.annotate(f"{labels[0]}", xy=(80, 500),  xycoords='data', xytext=(0.05, 0.88), textcoords='axes fraction')
    ax.annotate(f"{labels[1]}", xy=(80, -500),  xycoords='data', xytext=(0.05, 0.025), textcoords='axes fraction')
    ax.axhline(0, 0, 1, linestyle='-', linewidth=1.0, c='black')
    ax.set_xlim(0, bins)
                        
    return fref
    for i, k in enumerate(keys):
        
        bottom_line = None
        for j, hist_data in enumerate(_get_pt_hist(data, bins=bins, key=k)):
            
            bottom_line = np.zeros_like(hist_data) if bottom_line is None else bottom_line
            
            # Flip across the X axis for the second expt
            if i == 1:
                hist_data *= -1
                                
            fref.append(ax.bar(hist_labels, 
                               hist_data, 
                               bottom=bottom_line, 
                               width=0.5, 
                               label=i, color=pool_palette()[j]))
                        
            bottom_line = bottom_line + hist_data
            
            hist_limit.append(np.max(np.abs(bottom_line)))

    hist_limit = max(hist_limit)
    ax.set_xticks(np.arange(11) * int(bins / 10))
    ax.set_xticklabels(np.arange(11) / 10, rotation=90)
    ax.set_yticks([-5000, 0, 5000], labels=[5000, 0, 5000])
    ax.set_ylim(-1 * hist_limit, hist_limit)
    ax.set_ylabel("# Cells")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel("Pseudotime")
    ax.annotate(f"{labels[0]}", xy=(80, 500),  xycoords='data', xytext=(0.05, 0.88), textcoords='axes fraction')
    ax.annotate(f"{labels[1]}", xy=(80, -500),  xycoords='data', xytext=(0.05, 0.025), textcoords='axes fraction')
    ax.axhline(0, 0, 1, linestyle='-', linewidth=1.0, c='black')
    ax.set_xlim(0, bins)
                        
    return fref