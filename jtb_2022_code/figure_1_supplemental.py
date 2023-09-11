import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

from jtb_2022_code.utils.Figure_deseq import DESeq2
from jtb_2022_code.figure_constants import *
from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.utils.figure_data import sum_for_pseudobulk, calc_group_props

supp_1_panel_labels = {'ncells': "A", 
                       'ncounts': "B", 
                       'umap_1': "C",
                       'umap_2': "D",
                       'umap_3': "E",
                       'umap_4': "F",
                       'umap_5': "G",
                       'umap_6': "H",}


supp_1_panel_titles = {'ncells': "# Cells",
                       'ncounts': "# Counts [UMI] / Cell",
                       'umap_1': "Time (Pool)",
                       'umap_2': "Cell Cycle",
                       'umap_3': "Genotype",
                       'umap_4': "# Counts [UMI] / Cell",
                       'umap_5': "RP/RiBi Expression",
                       'umap_6': "iESR Expression"}


supp_1_2_panel_labels = {'hm': "A", 
                         'cc': "B", 
                         'pr': "C"}


def _mean_by_pool(adata, lfc_threshold=None):
    
    _counts = adata.layers['counts']
    _umi = _counts.sum(axis=1).A1 / 3099
    _counts /= _umi[:, None]
    
    _mean_expression = pd.DataFrame(
        np.vstack(
            [
                _counts[adata.obs['Pool'] == i, :].mean(axis=0).A1
                for i in range(1, 9)
            ]
        ),
        columns=adata.var_names
    )

    _mean_expression = _mean_expression.loc[:, np.all(_mean_expression != 0, axis=0)]
    _mean_expression /= _mean_expression.iloc[0, :]
    _mean_expression = np.log2(_mean_expression)
    
    if lfc_threshold is not None:
        _mean_expression = _mean_expression.loc[:, np.any(np.abs(_mean_expression) > lfc_threshold, axis=0)]

    return _mean_expression

def _get_cc_bar_data(adata):

    cc_counts = adata.obs[['CC', 'Pool']].value_counts().reindex(CC_COLS, level=0)
    pool_counts = adata.obs[['Pool']].value_counts()
    cc_counts = cc_counts.divide(pool_counts)
            
    return cc_counts.reset_index().pivot(
        index='Pool',
        columns='CC',
        values='count'
    )

def _get_prop_bar_data(adata, counts=True):
    
    if counts:
        _counts = pd.DataFrame({
            c: adata.layers['counts'][:, adata.var[c]].sum(axis=1).A1
            for c in GENE_CAT_COLS
        })
        _counts["Pool"] = adata.obs['Pool'].values
        return _counts.groupby("Pool").agg('mean')
    
    else:
        return adata.obs[GENE_CAT_COLS + ['Pool']].groupby('Pool').agg('median')


def figure_1_supplement_1_plot(data, save=True):

    fig_refs = {}

    fig = plt.figure(figsize=(5, 8), dpi=MAIN_FIGURE_DPI)

    axd = {
        'ncells': fig.add_axes([0.1, 0.875, 0.7, 0.1]),
        'ncounts': fig.add_axes([0.1, 0.725, 0.7, 0.1]),
        'legend_a': fig.add_axes([0.81, 0.725, 0.13, 0.25]),
        'umap_1': fig.add_axes([0.05, 0.5, 0.30, 0.175]),
        'umap_2': fig.add_axes([0.55, 0.5, 0.30, 0.175]),
        'umap_3': fig.add_axes([0.05, 0.275, 0.30, 0.175]),
        'umap_4': fig.add_axes([0.55, 0.275, 0.30, 0.175]),
        'umap_5': fig.add_axes([0.05, 0.05, 0.30, 0.175]),
        'umap_6': fig.add_axes([0.55, 0.05, 0.30, 0.175]),
        'umap_1_legend': fig.add_axes([0.35, 0.5, 0.1, 0.175]),    
        'umap_2_legend': fig.add_axes([0.85, 0.5, 0.1, 0.175]),
        'umap_3_legend': fig.add_axes([0.35, 0.275, 0.1, 0.175]),    
        'umap_4_legend': fig.add_axes([0.86, 0.275, 0.025, 0.175]),
        'umap_5_legend': fig.add_axes([0.36, 0.05, 0.025, 0.175]),    
        'umap_6_legend': fig.add_axes([0.86, 0.05, 0.025, 0.175]),
    }

    axd['umap_1_legend'].legend(
        title="Time (Pool)",
        handles=[patches.Patch(color=hexcolor) for hexcolor in pool_palette()],
        labels=[str(x) for x in range(1, 9)],
        frameon=False,
        loc='center left',
        fontsize=8,
        title_fontsize=8,
        handlelength=1,
        handleheight=1,
        borderaxespad=0
    )

    axd['umap_2_legend'].legend(
        title="Phase",
        handles=[patches.Patch(color=hexcolor) for hexcolor in cc_palette()],
        labels=["G1/M", "G1", "S", "G2", "M"],
        frameon=False,
        loc='center left',
        fontsize=8,
        title_fontsize=8,
        handlelength=1,
        handleheight=1,
        borderaxespad=0
    )

    axd['umap_3_legend'].legend(
        title="Genotype",
        handles=[patches.Patch(color=hexcolor) for hexcolor in strain_palette()],
        labels=["WT", "fpr1"],
        frameon=False,
        loc='center left',
        fontsize=8,
        title_fontsize=8,
        handlelength=1,
        handleheight=1,
        borderaxespad=0
    )
    axd['umap_1_legend'].axis('off')
    axd['umap_2_legend'].axis('off')
    axd['umap_3_legend'].axis('off')

    ### PANEL A ###

    cell_counts = data.all_data.obs[['Experiment', 'Gene', 'Pool']].value_counts()

    width = 0.20
    fig_refs['ncells_1'] = axd['ncells'].bar(np.arange(8) - width * 1.5, 
                                             cell_counts.loc[1].loc["WT"].sort_index().values, 
                                             width, label='1 [WT]', color=expt_palette(long=True)[0])

    fig_refs['ncells_2'] = axd['ncells'].bar(np.arange(8) - width * 0.5, 
                                             cell_counts.loc[2].loc["WT"].sort_index().values, 
                                             width, label='2 [WT]', color=expt_palette(long=True)[1])

    fig_refs['ncells_3'] = axd['ncells'].bar(np.arange(8) + width * 0.5, 
                                             cell_counts.loc[1].loc["fpr1"].sort_index().values, 
                                             width, label='1 [fpr1]', color=expt_palette(long=True)[2])

    fig_refs['ncells_4'] = axd['ncells'].bar(np.arange(8) + width * 1.5, 
                                             cell_counts.loc[2].loc["fpr1"].sort_index().values, 
                                             width, label='2 [fpr1]', color=expt_palette(long=True)[3])

    axd['ncells'].set_xticks(np.arange(8), labels=np.arange(8) + 1)

    fig_refs['ncells_legend'] = axd['legend_a'].legend(
        title="Expt. Rep.",
        handles=[patches.Patch(color=hexcolor) for hexcolor in expt_palette(long=True)],
        labels=['1 [WT]', '2 [WT]', '1 [fpr1]', '2 [fpr1]'],
        frameon=False,
        loc='center left',
        fontsize=8,
        title_fontsize=8
    )
    axd['legend_a'].axis('off')
    ### PANEL B ###

    gcols = ['Experiment', 'Gene', 'Pool']
    cell_umis = [(e, g, p, data['n_counts'].values) for (e, g, p), data in data.all_data.obs[gcols + ['n_counts']].groupby(gcols)]
    cell_umis = {(e, g): [d1 for e1, g1, _, d1 in cell_umis if (e1 == e) and (g1 == g)] for e, g, _, _ in cell_umis}

    fig_refs['ncounts'] = [axd['ncounts'].violinplot(cell_umis[(1, "WT")], positions = np.arange(8) - width * 1.5, 
                                                     widths=width, showmeans=False, showmedians=True, showextrema=False),
                           axd['ncounts'].violinplot(cell_umis[(2, "WT")], positions = np.arange(8) - width * 0.5, 
                                                     widths=width, showmeans=False, showmedians=True, showextrema=False),
                           axd['ncounts'].violinplot(cell_umis[(1, "fpr1")], positions = np.arange(8) + width * 0.5, 
                                                     widths=width, showmeans=False, showmedians=True, showextrema=False),
                           axd['ncounts'].violinplot(cell_umis[(2, "fpr1")], positions = np.arange(8) + width * 1.5, 
                                                     widths=width, showmeans=False, showmedians=True, showextrema=False)]

    for i, parts in enumerate(fig_refs['ncounts']):
        for pc in parts['bodies']:
            pc.set_facecolor(expt_palette(long=True)[i])
            pc.set_alpha(0.75)

    axd['ncounts'].set_xticks(np.arange(8), labels=np.arange(8) + 1)
    axd['ncounts'].set_ylim([0, 10000])

    ### PANELS C-H ###

    umap_adata = ad.AnnData(np.zeros((data.all_data.X.shape[0], 1)), dtype=data.all_data.X.dtype)
    umap_adata.obsm['X_umap'] = data.all_data.obsm['X_umap'].copy()
    umap_adata.obs = data.all_data.obs.copy()
    umap_adata.obs['Ribosomal'] = umap_adata.obs['RP'] + umap_adata.obs['RiBi']

    fig_refs['umap_1'] = plot_umap(
        umap_adata,
        axd['umap_1'],
        color="Pool",
        palette=pool_palette(),
        alpha=0.2,
        size=1
    )

    fig_refs['umap_2'] = plot_umap(
        umap_adata,
        axd['umap_2'],
        color="CC",
        palette=cc_palette(),
        alpha=0.2,
        size=1
    )

    fig_refs['umap_3'] = plot_umap(
        umap_adata,
        axd['umap_3'],
        color="Gene",
        palette=strain_palette(),
        alpha=0.2,
        size=1
    )

    fig_refs['umap_4'] = plot_umap(
        umap_adata,
        axd['umap_4'],
        color="n_counts",
        cmap='viridis',
        alpha=0.2,
        size=1,
        vmin=0,
        vmax=10000
    )

    fig_refs['umap_5'] = plot_umap(
        umap_adata,
        axd['umap_5'],
        color="Ribosomal",
        cmap='viridis',
        alpha=0.2,
        size=1,
        vmin=0,
        vmax=0.6
    )

    fig_refs['umap_6'] = plot_umap(
        umap_adata,
        axd['umap_6'],
        color="iESR",
        cmap='viridis',
        alpha=0.2,
        size=1,
        vmin=0,
        vmax=0.1
    )

    for i in range(4, 7):
        fig_refs[f'umap_{i}_cbar'] = plt.colorbar(
            fig_refs[f'umap_{i}'],
            cax=axd[f'umap_{i}_legend'],
        )
        axd[f'umap_{i}_legend'].tick_params(labelsize=8) 
        fig_refs[f'umap_{i}_cbar'].solids.set(alpha=1)

    for ax_id, label in supp_1_panel_labels.items():
        axd[ax_id].set_title(supp_1_panel_titles[ax_id], size=8)
        axd[ax_id].set_title(label, loc='left', weight='bold', size=10)
        axd[ax_id].tick_params(axis='both', which='major', labelsize=8)


    if save:
        fig.savefig(FIGURE_1_1_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white')

    return fig

def figure_1_supplement_2_plot(data, save=True):

    fig_refs = {}

    fig = plt.figure(figsize=(5, 7), dpi=MAIN_FIGURE_DPI)
    def cluster_on_rows(dataframe, **kwargs):

        return dataframe.index[dendrogram(
            linkage(
                pdist(
                    dataframe.values,
                    **kwargs
                ),
                method='ward'
            ),
            no_plot=True
        )['leaves']]

    _l = 0.17
    _w = 0.16
    _h = 0.14

    axd = {
        'expr_1': fig.add_axes([_l, 0.76, _w, _h]),
        'expr_2': fig.add_axes([_l + _w, 0.76, _w, _h]),
        'expr_3': fig.add_axes([_l + 2 * _w, 0.76, _w, _h]),
        'expr_4': fig.add_axes([_l + 3 * _w, 0.76, _w, _h]),
        'expr_cbar': fig.add_axes([_l + 4 * _w + 0.01, 0.76, 0.025, _h]),
        'cc_1': fig.add_axes([_l, 0.54, _w, _h]),
        'cc_2': fig.add_axes([_l + _w, 0.54, _w, _h]),
        'cc_3': fig.add_axes([_l + 2 *_w, 0.54, _w, _h]),
        'cc_4': fig.add_axes([_l + 3 * _w, 0.54, _w, _h]),
        'cc_legend': fig.add_axes([_l + 4 * _w, 0.54, 0.13, _h]),    
        'cat_1': fig.add_axes([_l, 0.32, _w, _h]),
        'cat_2': fig.add_axes([_l + _w, 0.32, _w, _h]),    
        'cat_3': fig.add_axes([_l + 2 * _w, 0.32, _w, _h]),
        'cat_4': fig.add_axes([_l + 3 * _w, 0.32, _w, _h]),    
        'cat_legend': fig.add_axes([_l + 4 * _w, 0.32, 0.13, _h]),
        'cat_prop_1': fig.add_axes([_l, 0.1, _w, _h]),
        'cat_prop_2': fig.add_axes([_l + _w, 0.1, _w, _h]),    
        'cat_prop_3': fig.add_axes([_l + 2 * _w, 0.1, _w, _h]),
        'cat_prop_4': fig.add_axes([_l + 3 * _w, 0.1, _w, _h]),
        'pool_legend': fig.add_axes([_l + 4 * _w, 0.05, 0.13, 0.25]),
    }

    axd['pool_legend'].imshow(
        plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME), aspect='equal'
    )
    axd['pool_legend'].axis('off')
    
    axd['expr_1'].set_title("A", loc='left', weight='bold', size=10, x=-0.1)
    axd['cc_1'].set_title("B", loc='left', weight='bold', size=10, x=-0.1)
    axd['cat_1'].set_title("C", loc='left', weight='bold', size=10, x=-0.1)
    axd['cat_prop_1'].set_title("D", loc='left', weight='bold', size=10, x=-0.1)

    axd['cc_1'].set_ylabel("% Cells in Phase", size=8)
    axd['cc_1'].tick_params(axis='both', which='major', labelsize=8)

    axd['cat_1'].set_ylabel("Mean Counts (UMI)", size=8)
    axd['cat_1'].tick_params(axis='both', which='major', labelsize=8)

    axd['cat_prop_1'].set_ylabel("% Gene Expression", size=8)
    axd['cat_prop_1'].tick_params(axis='both', which='major', labelsize=8)

    axd['expr_2'].set_xlabel("Time (Pool)", size=8, x=1)
    axd['cc_2'].set_xlabel("Time (Pool)", size=8, x=1)
    axd['cat_2'].set_xlabel("Time (Pool)", size=8, x=1)
    axd['cat_prop_2'].set_xlabel("Time (Pool)", size=8, x=1)
    
    plot_y_order = cluster_on_rows(
        _mean_by_pool(data.expt_data[(1, "WT")], lfc_threshold=0.5).T
    )

    for i, k in enumerate([(1, "WT"), (2, "WT"), (1, "fpr1"), (2, "fpr1")]):

        i += 1

        fig_refs[f'expr_{i}'] = axd[f'expr_{i}'].imshow(
            _mean_by_pool(
                data.expt_data[k]
            ).reindex(
                plot_y_order,
                axis=1
            ).fillna(0).values.T,
            cmap='bwr',
            vmin=-5,
            vmax=5,
            aspect='auto', interpolation='none'
        )
        axd[f'expr_{i}'].set_yticks([])
        axd[f'expr_{i}'].set_xticks(list(range(0, 8)), list(range(1, 9)), size=8)
        axd[f'expr_{i}'].set_title(f"Expt. {k[0]}: {k[1]}", size=8)

        fig_refs[f'cc_{i}'] = plot_stacked_barplot(
            _get_cc_bar_data(data.expt_data[k]),
            axd[f'cc_{i}'],
            CC_COLS[::-1],
            palette=cc_palette()[::-1]
        )
        axd[f'cc_{i}'].set_xticks(list(range(0, 8)), list(range(1, 9)), size=8)

        fig_refs[f'cat_{i}'] = plot_stacked_barplot(
            _get_prop_bar_data(data.expt_data[k]),
            axd[f'cat_{i}'],
            GENE_CAT_COLS[::-1],
            palette=gene_category_palette()[::-1]
        )
        axd[f'cat_{i}'].set_xticks(list(range(0, 8)), list(range(1, 9)), size=8)
        axd[f'cat_{i}'].set_ylim(0, 6000)

        fig_refs[f'cat_prop_{i}'] = plot_stacked_barplot(
            _get_prop_bar_data(data.expt_data[k], counts=False),
            axd[f'cat_prop_{i}'],
            GENE_CAT_COLS[::-1],
            palette=gene_category_palette()[::-1]
        )
        axd[f'cat_prop_{i}'].set_xticks(list(range(0, 8)), list(range(1, 9)), size=8)

        if i > 1:
            axd[f'cc_{i}'].set_yticks([], [])
            axd[f'cat_{i}'].set_yticks([], [])
            axd[f'cat_prop_{i}'].set_yticks([], [])

    fig_refs[f'expr_cbar'] = plt.colorbar(
        fig_refs[f'expr_1'],
        cax=axd[f'expr_cbar'],
        label="Log$_2$ FC"
    )
    axd[f'expr_cbar'].tick_params(labelsize=8)

    axd['cc_legend'].legend(
        title="Phase",
        handles=[patches.Patch(color=hexcolor) for hexcolor in cc_palette()],
        labels=CC_COLS,
        frameon=False,
        loc='center left',
        fontsize=8,
        title_fontsize=8,
        handlelength=1,
        handleheight=1,
        borderaxespad=0
    )
    axd['cc_legend'].axis('off')

    axd['cat_legend'].legend(
        title="Category",
        handles=[patches.Patch(color=hexcolor) for hexcolor in gene_category_palette()],
        labels=GENE_CAT_COLS,
        frameon=False,
        loc='center left',
        fontsize=8,
        title_fontsize=8,
        handlelength=1,
        handleheight=1,
        borderaxespad=0
    )
    axd['cat_legend'].axis('off')

    if save:
        fig.savefig(FIGURE_1_2_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white')

    return fig