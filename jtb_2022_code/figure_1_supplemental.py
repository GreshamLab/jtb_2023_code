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

from jtb_2022_code.utils.Figure_deseq import DESeq2, hclust
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


def _get_hm_data(m_dict, add_spacer=True):
    
    hm_spacer = pd.Series(0.0, index=m_dict[1, "WT"].index, name=9)
    return pd.concat((pd.concat((m_dict[(i, g)], hm_spacer), axis=1) if add_spacer else m_dict[(i, g)]
                     for g in ["WT", "fpr1"] for i in [1, 2]), axis=1)


def _get_cc_bar_data(data, add_spacer=True):
    cc_props = []
    cc_spacer = pd.Series([0.0] * len(CC_COLS), index=pd.MultiIndex.from_product([[9], CC_COLS]))

    for g in ["WT", "fpr1"]:
        for i in [1, 2]:
            cc_counts = data.expt_data[(i, g)].obs[['CC', 'Pool']].value_counts().reindex(CC_COLS, level=0)
            pool_counts = data.expt_data[(i, g)].obs[['Pool']].value_counts()
            cc_counts = cc_counts.divide(pool_counts)
            
            if add_spacer:
                cc_counts = pd.concat((cc_counts, cc_spacer))
        
            cc_props.append(cc_counts)
                        
    return pd.concat(cc_props, axis=1)
            

def _get_prop_bar_data(data, add_spacer=True):
    gene_props = []
    gene_spacer = pd.DataFrame([[0.0] * len(GENE_CAT_COLS)], columns=GENE_CAT_COLS, index=[9])
    for g in ["WT", "fpr1"]:
        for i in [1, 2]:
            pr_median = data.expt_data[(i, g)].obs[GENE_CAT_COLS + ['Pool']].groupby('Pool').agg('median')
            pr_median = pr_median.divide(pr_median.sum(axis=1), axis=0)
            pr_median = pd.concat((pr_median, gene_spacer))
            gene_props.append(pr_median)
    return pd.concat(gene_props, axis=1)

def _add_avlines(ax, locs, **kwargs):
    for l in locs:
        ax.axvline(l, **kwargs)


def figure_1_supplement_1_plot(data, save=True):

    fig_refs = {}

    fig, axd = plt.subplot_mosaic([['ncells', 'ncells'],
                                   ['ncounts', 'ncounts'],
                                   ['umap_1', 'umap_2'],
                                   ['umap_3', 'umap_4'],
                                   ['umap_5', 'umap_6']],
                                  gridspec_kw=dict(width_ratios=[1, 1], 
                                                   height_ratios=[0.5, 0.5, 1, 1, 1]), 
                                  figsize=(8, 9), dpi=300)
                                  #constrained_layout=True)

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

    fig_refs['ncells_lax'] = add_legend_axis(axd['ncells'], size='10%')
    fig_refs['ncells_legend'] = fig_refs['ncells_lax'].legend(title="Expt. Rep.",
                                                              handles=[patches.Patch(color=hexcolor) for hexcolor in expt_palette(long=True)],
                                                              labels=['1 [WT]', '2 [WT]', '1 [fpr1]', '2 [fpr1]'],
                                                              frameon=False,
                                                              loc='center left')
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


    fig_refs['ncounts_lax'] = add_legend_axis(axd['ncounts'], size='10%')
    fig_refs['ncounts_legend'] = fig_refs['ncounts_lax'].legend(title="Expt. Rep.",
                                                                handles=[patches.Patch(color=hexcolor) for hexcolor in expt_palette(long=True)],
                                                                labels=['1 [WT]', '2 [WT]', '1 [fpr1]', '2 [fpr1]'],
                                                                frameon=False,
                                                                loc='center left')

    for i, hexcolor in enumerate(expt_palette(long=True)):
        fig_refs['ncounts_legend'].legendHandles[i].set_color(hexcolor)

    ### PANELS C-H ###

    umap_adata = ad.AnnData(np.zeros((data.all_data.X.shape[0], 1)))
    umap_adata.obsm['X_umap'] = data.all_data.obsm['X_umap'].copy()
    umap_adata.obs = data.all_data.obs.copy()

    # Explicitly squeeze data because scanpy's handling of vmin/vmax is dogshit
    umap_adata.obs['Ribosomal'] = umap_adata.obs['RP'] + umap_adata.obs['RiBi']

    umap_adata.obs['n_counts'] = squeeze_data(umap_adata.obs['n_counts'], 10000, 0)
    umap_adata.obs['Ribosomal'] = squeeze_data(umap_adata.obs['Ribosomal'], 0.6, 0)
    umap_adata.obs['iESR'] = squeeze_data(umap_adata.obs['iESR'], 0.1, 0)

    fig_refs['umap_1'] = sc.pl.umap(umap_adata, ax=axd['umap_1'], color="Pool", palette=pool_palette(), 
                                    show=False, alpha=0.25, size=2, sort_order=False, legend_loc='none')
    fig_refs['umap_1_legend'] = add_legend(add_legend_axis(axd['umap_1']), pool_palette(), umap_adata.obs['Pool'].dtype.categories.values)
    fig_refs['umap_2'] = sc.pl.umap(umap_adata, ax=axd['umap_2'], color="CC", palette=cc_palette(), 
                                    show=False, alpha=0.25, size=2, sort_order=False, legend_loc='none')
    fig_refs['umap_2_legend'] = add_legend(add_legend_axis(axd['umap_2']), cc_palette(), umap_adata.obs['CC'].dtype.categories.values)
    fig_refs['umap_3'] = sc.pl.umap(umap_adata, ax=axd['umap_3'], color="Gene", palette=strain_palette(), 
                                    show=False, alpha=0.25, size=2, sort_order=False, legend_loc='none')
    fig_refs['umap_3_legend'] = add_legend(add_legend_axis(axd['umap_3']), strain_palette(), umap_adata.obs['Gene'].dtype.categories.values)
    fig_refs['umap_4'] = sc.pl.umap(umap_adata, ax=axd['umap_4'], color="n_counts", 
                                    show=False, alpha=0.25, size=2, sort_order=False)#, legend_loc='none')
    #fig_refs['umap_4_cbar'] = plt.colorbar(fig_refs['umap_4'].collections[0], cax=add_legend_axis(axd['umap_4']), pad=0.01, fraction=0.08, aspect=30, use_gridspec=True)
    fig_refs['umap_5'] = sc.pl.umap(umap_adata, ax=axd['umap_5'], color="Ribosomal", 
                                    show=False, alpha=0.25, size=2, sort_order=False)#, legend_loc='none')
    #fig_refs['umap_5_cbar'] = plt.colorbar(fig_refs['umap_5'].collections[0], cax=add_legend_axis(axd['umap_5']), pad=0.01, fraction=0.08, aspect=30, use_gridspec=True)
    fig_refs['umap_6'] = sc.pl.umap(umap_adata, ax=axd['umap_6'], color="iESR", 
                                    show=False, alpha=0.25, size=2, sort_order=False)#, legend_loc='none')
    #fig_refs['umap_6_cbar'] = plt.colorbar(fig_refs['umap_6'].collections[0], cax=add_legend_axis(axd['umap_6']), pad=0.01, fraction=0.08, aspect=30, use_gridspec=True)

    for ax_id, label in supp_1_panel_labels.items():
        axd[ax_id].set_title(supp_1_panel_titles[ax_id])
        axd[ax_id].set_title(label, loc='left', weight='bold')

    fig.tight_layout()    

    if save:
        fig.savefig(FIGURE_1_1_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white')

    return fig

def figure_1_supplement_2_plot(data, save=True):

    if 'DESeq_Pseudobulk' not in data.expt_data[(1, "WT")].uns:
        ### DESeq Based on Pseudobulk ###
        gb, mb = sum_for_pseudobulk(data.all_data, ['Pool', 'Experiment', 'Gene', 'Replicate'])

        results = {}

        for i in [1, 2]:
            for g in ["WT", "fpr1"]:
                idx = mb['Experiment'] == str(i)
                idx &= mb['Gene'] == g

                print(f"Running DESeq2 (Experiment {i}: {g}) on {idx.sum()} observations")
                de_obj = DESeq2(gb[idx], mb[idx], "~Pool", threads=4).run(fitType = "local", quiet=True)
                results[(i, g)] = de_obj.multiresults(lambda y: ("Pool", y, "1"), list(map(str, range(2, 9))), "Pool", lfcThreshold=FIGURE_1A_LFC_THRESHOLD)

        plot_genes = results[(1, "WT")].loc[results[(1, "WT")]['padj'] < FIGURE_1A_PADJ_THRESHOLD, :].index.unique()
        plot_genes = plot_genes.union(results[(2, "WT")].loc[results[(2, "WT")]['padj'] < FIGURE_1A_PADJ_THRESHOLD, :].index.unique())

        plot_matrix = {(i, g): results[(i, g)].loc[results[(i, g)].index.isin(plot_genes), :].pivot(
            columns="Pool", values="log2FoldChange"
        ).reindex(
            list(map(str, range(1, 9))), axis=1
        ).fillna(0) for i in [1, 2] for g in ["WT", "fpr1"]}

        print("Clustering Results")
        plot_hclust = hclust(plot_matrix[(1, "WT")])
        plot_y_order = plot_hclust['labels'][plot_hclust['order'] - 1]

        for k, v in plot_matrix.items():
            plot_matrix[k] = plot_matrix[k].reindex(plot_y_order, axis=0)

        for k, e_data in data.expt_data.items():
            e_data.uns['DESeq_Pseudobulk'] = plot_matrix[k]
            
        data.save()
    
    else:
        plot_matrix = {k: v.uns['DESeq_Pseudobulk'] for k, v in data.expt_data.items()}

    ### BUILD PLOT ###
    fig_refs = {}

    fig, axd = plt.subplot_mosaic([['hm', 'hm_cbar'],
                                   ['cc', 'cc_cbar'],
                                   ['pr', 'pr_cbar']],
                                  gridspec_kw=dict(width_ratios=[4, 0.1], 
                                                   height_ratios=[1, 1, 1],
                                                   wspace=0, hspace=0.01), 
                                  figsize=(8, 6), dpi=300,
                                  constrained_layout=True)

    ### PANEL A HEATMAP ###
    fig_refs['hm'] = axd['hm'].imshow(squeeze_data(_get_hm_data(plot_matrix).iloc[:, :-1], FIGURE_1A_MINMAX), 
                                      cmap='bwr', aspect='auto', interpolation='none')

    axd['hm'].set_xticks(list(range(0, 9 * 4))[:-1], labels=((list(range(1, 9)) + [""] ) * 4)[:-1])
    axd['hm'].set_yticks([], labels=[])
    axd['hm'].set_ylabel("Genes")
    axd['hm'].set_xlabel("Time [Pool]")
    axd['hm'].set_title(" " + "           ".join(f"Expt {i}: {g}" for g in ["WT", 'fpr1'] for i in [1, 2]))
    _add_avlines(axd['hm'], [8, 17, 26], color="black", linewidth=1)

    ### PANEL B CELL CYCLE STACKED BARPLOT ###
    pool_bottoms = np.zeros(9 * 4)
    pool_data = _get_cc_bar_data(data)
    for c, cc in enumerate(CC_COLS[::-1]):
        pools = pool_data.loc[(slice(None), cc), :].values.T.flatten()
        fig_refs["cc"] = axd["cc"].bar(list(range(0, 9 * 4)), pools, label=cc, bottom=pool_bottoms, 
                                       color=cc_palette()[::-1][c])
        pool_bottoms += pools

    axd["cc"].set_ylabel("% Cells in Phase")
    axd["cc"].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))    
    axd["cc"].set_xticks(list(range(0, 9 * 4)), labels=(list(range(1, 9)) + [""] ) * 4)
    axd["cc"].set_xlim(-0.5, 34.5)
    axd['cc'].set_xlabel("Time [Pool]")
    _add_avlines(axd['cc'], [8, 17, 26], color="black", linewidth=1)

    ### PANEL C GENE CATEGORY STACKED BARPLOT ###
    pool_bottoms = np.zeros(9 * 4)
    pr_data = _get_prop_bar_data(data)
    for p, pr in enumerate(GENE_CAT_COLS[::-1]):
        pools = pr_data.loc[:, pr].T.values.flatten()
        fig_refs["pr"] = axd["pr"].bar(list(range(0, 9 * 4)), pools, label=pr, bottom=pool_bottoms, 
                                       color=gene_category_palette()[::-1][p])
        pool_bottoms += pools

    axd["pr"].set_ylabel("% Gene Expression")
    axd["pr"].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))       
    axd["pr"].set_xticks(list(range(0, 9 * 4))[:-1], labels=((list(range(1, 9)) + [""] ) * 4)[:-1])
    axd["pr"].set_xlim(-0.5, 34.5)
    axd['pr'].set_xlabel("Time [Pool]")
    _add_avlines(axd['pr'], [8, 17, 26], color="black", linewidth=1)

    ### COLORBARS ###
    fig_refs['hm_cbar'] = fig.colorbar(fig_refs['hm'], cax=axd['hm_cbar'], orientation="vertical", aspect=40)
    fig_refs['hm_cbar'].set_label("Log2 FC")

    axd['cc_cbar'].axis('off')
    fig_refs['cc_legend'] = axd['cc_cbar'].legend(title="Cell Cycle",
                                                  handles=[patches.Patch(color=hexcolor) for hexcolor in cc_palette()],
                                                  labels=CC_COLS,
                                                  frameon=False,
                                                  loc='center left')

    axd['pr_cbar'].axis('off')
    fig_refs['pr_legend'] = axd['pr_cbar'].legend(title="Gene Category",
                                                  handles=[patches.Patch(color=hexcolor) for hexcolor in gene_category_palette()],
                                                  labels=GENE_CAT_COLS,
                                                  frameon=False,
                                                  loc='center left')

    for i, hexcolor in enumerate(cc_palette()):
        fig_refs['cc_legend'].legendHandles[i].set_color(hexcolor)

    for i, hexcolor in enumerate(gene_category_palette()):
        fig_refs['pr_legend'].legendHandles[i].set_color(hexcolor)

    ### PANEL LABELS ###
    for ax_id, label in supp_1_2_panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold')

    if save:
        fig.savefig(FIGURE_1_2_SUPPLEMENTAL_FILE_NAME + ".png", facecolor='white')

    return fig