import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import re
pd.set_option('precision',3)
np.set_printoptions(precision=3, linewidth=1000)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

color_pallet = "deep"
plt.style.use('seaborn-ticks')
sns.set_color_codes(color_pallet)
plt.rcParams['svg.fonttype'] = 'none'

import scanpy as sc

def sparsifier(pcs,devs):
    c = 0
    sparse_pc = np.zeros([len(pcs),len(pcs[0])])
    for pc in pcs:
        mean = np.mean(pc, axis=0)
        sd = np.std(pc, axis=0)
        a = np.asarray([x if (x > mean + (devs * sd)) else 0 for x in pc])
        b = np.asarray([x if (x < mean - (devs * sd)) else 0 for x in pc])
        sparse_pc[c] = a+b
        c+=1
    return(sparse_pc)

def sparse_gen(data,pcs=100):
    sc.pp.pca(data, n_comps=pcs)
    loadings = data.varm['PCs'].T
    sparse_pcs = sparsifier(loadings.copy(), 6)
    sparse_pca=data.X@sparse_pcs.T
    sparse = sparse_pca.T.copy()

    return sparse, sparse_pcs

def data_prep(file):
    adata = sc.read(file)
    adata.X = adata.X.astype(float)
    data = adata.copy()
    sc.pp.normalize_per_cell(data)
    sc.pp.log1p(data)
    exp2 = data.copy()

    return exp2

def get_corr(sparsified_dat):
    rhos = np.zeros([sparse.shape[0],sparse.shape[0]])
    for px in range(0, len(sparsified_dat)):
        for py in range(0, len(sparsified_dat)):
            rhos[px][py] = np.asarray(sp.stats.spearmanr(sparsified_dat[px],sparsified_dat[py]))[0]
    return rhos

def genes_in_pc(sparse_loadings,adata):
    _genes_per_pc = []
    for i in range(0,len(sparse_loadings)):
        p = sparse_loadings[i]
        cluster_genes = []
        for g in adata.var_names[np.nonzero(p)]:
            cluster_genes.append(g)
        cluster_genes= np.unique(cluster_genes)
        _genes_per_pc.append(cluster_genes)

    return _genes_per_pc

def load_genes_per_pc(f):
    file = f
    exp = data_prep(file)
    sparse, sparse_pcs = sparse_gen(exp)

    genes_per_pc = genes_in_pc(sparse_pcs,exp)

    unique_genes_per_pc = np.zeros(100)
    for px in range(1,len(genes_per_pc)):
            unique_genes_per_pc[px]=(len(genes_per_pc[px])-len(set(genes_per_pc[px]).intersection(np.unique(np.concatenate(np.asarray(genes_per_pc[:px]))))))/len(genes_per_pc[px])
    unique_genes_per_pc[0] = len(genes_per_pc[0])/len(genes_per_pc[0])

    overlap = np.zeros(100)
    for px in range(1,len(genes_per_pc)):
            overlap[px]=(len(set(genes_per_pc[px]).intersection(np.unique(np.concatenate(np.asarray(genes_per_pc[:px]))))))/len(genes_per_pc[px])

    df = pd.DataFrame({'PC':[str(i) for i in range(1,len(overlap)+1)],
                      'Unique Genes': unique_genes_per_pc,
                      'Overlapping Genes':overlap})
    return df

def genes_per_pc_plt(df,ax):

    #fig,ax = plt.subplots(figsize=(8,5))
    df.set_index('PC').plot(kind='bar', stacked=True, color=['purple', 'orange'],width=1,ax=ax,legend=False,align='edge',linewidth=0,edgecolor='none',grid=False)
    #plt.yticks(fontsize=16)
    #plt.xticks(ticks=[0,20,40,60,80,99], labels = ['1','20','40','60','80','100'],fontsize=16)
    #plt.legend(fontsize = 16)


    return ax

def load_dewakss_dat():
    files = ['../Data/optimal_pcs_rep1_WT.csv','../Data/optimal_pcs_rep2_WT.csv','../Data/optimal_pcs_rep1_frp1.csv','../Data/optimal_pcs_rep2_frp1.csv']
    pd_dfs = []
    opt = 'optimal'
    steps = 'steps'
    for f in files:
        df = pd.read_csv(f)
        pd_dfs.append(df)
    return pd_dfs

def mk_dwks_f(dfs,key,ax):
    colors = ['b','orange','g','r']
    d=0
    sns.set_style("whitegrid")
    for df in dfs:
        sns.lineplot(df['steps'],df[key],ax=ax)
        sns.scatterplot(np.asarray([df['optimal'][0]]), np.asarray([df[key][int(df['optimal'][0])]]), marker='x',s=80,ax=ax)
        d+=1
    return ax

def axis_labels_evc(ax,key):
    ax[key].set_xticks([0,20,40,60,80,99])
    ax[key].set_xticklabels(['0','20','40','60','80','100'],fontsize=12)
    ax[key].set_ylabel('EVC', fontsize = 13.5,labelpad=1)
    ax[key].set_xlabel('PCs',fontsize = 13.5,labelpad=1)

def axis_labels_mse(ax,key):
    ax[key].set_xticks([0,20,40,60,80,99])
    ax[key].set_xticklabels(['0','20','40','60','80','100'],fontsize=12)
    ax[key].set_yticklabels(['','0.1','0.2','0.3'],fontsize=12)
    ax[key].set_ylabel('MSE', fontsize = 13.5,labelpad=1)
    ax[key].set_xlabel('PCs',fontsize = 13.5,labelpad=1)

def axis_labels(ax,key):
    ax[key].set_xticks([0,20,40,60,80,99])
    ax[key].set_xticklabels(['1','20','40','60','80','100'],fontsize=12)
    ax[key].set_yticks([0,.5,1])
    ax[key].set_yticklabels(['0','50%','100%'],fontsize=12)
    ax[key].set_ylabel('% Genes', fontsize = 13.5,labelpad=1)
    ax[key].set_xlabel('PCs',fontsize = 13.5,labelpad=1)

def add_legend(ax, colors, labels, marker, title=None):
    fakeplots = [ax.scatter([], [], c=c, label=l,marker=marker) for c, l in zip(colors, labels)]
    ax.axis('off')
    return ax.legend(frameon=False,
                     loc='center left',
                     ncol=1,
                     columnspacing=0,
                     mode=None,
                     title=title)

def supplemental_figure_4_plot_data():

    #load dewakss data
    dfs=load_dewakss_dat()

    #load overlapping/unique genes data
    df1=load_genes_per_pc('../Data/2021_RAPA_TIMECOURSE_FIGS_1_WT.h5ad')
    df2=load_genes_per_pc('../Data/2021_RAPA_TIMECOURSE_FIGS_2_WT.h5ad')
    df3=load_genes_per_pc('../Data/2021_RAPA_TIMECOURSE_FIGS_1_fpr1.h5ad')
    df4=load_genes_per_pc('../Data/2021_RAPA_TIMECOURSE_FIGS_2_fpr1.h5ad')

    #define figure
    fig_refs={}

    layout = [['evcs','evcs', 'mse', 'mse', 'ev_legend'],
              ['rep1_wt','rep1_wt','rep2_wt','rep2_wt','g_legend'],
              ['rep1_frp1','rep1_frp1','rep2_frp1','rep2_frp1','g_legend'],]
    fig, axd = plt.subplot_mosaic(layout,
                                  gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.15],
                                                   height_ratios=[1.5, 2, 2],
                                                   wspace=0.01, hspace=0.025),
                                  figsize=(9, 6), dpi=300,
                                  constrained_layout=True)

    panel_labels =  {'evcs': "A",
                    'mse': "B",
                    'ev_legend': "",
                    'rep1_wt': "C",
                    'rep2_wt': "D",
                    'g_legend': "",
                    'rep1_frp1': "E",
                    'rep2_frp1': "F",}

    panel_titles = {'evcs': "",
                    'mse': "",
                    'ev_legend': "",
                    'rep1_wt': "Expt. 1: WT",
                    'rep2_wt': "Expt. 2: WT",
                    'g_legend': "",
                    'rep1_frp1': "fpr1",
                    'rep2_frp1': "fpr1",}
    for ax_id, label in panel_labels.items():
        axd[ax_id].set_title(label, loc='left', weight='bold')
    for ax_id, label in panel_titles.items():
        axd[ax_id].set_title(label)

    #plot dewakss figure
    keys = ['evcs','mse']
    for k in keys:
        fig_refs[k]=mk_dwks_f(dfs,k,axd[k])

    #plot barplots
    fig_refs['rep1_wt']=genes_per_pc_plt(df1,axd['rep1_wt'])
    fig_refs['rep2_wt']=genes_per_pc_plt(df2,axd['rep2_wt'])
    fig_refs['rep1_frp1']=genes_per_pc_plt(df3,axd['rep1_frp1'])
    fig_refs['rep2_frp1']=genes_per_pc_plt(df4,axd['rep2_frp1'])

    #add axes
    axis_labels_evc(axd,'evcs')
    axis_labels_mse(axd,'mse')
    axis_labels(axd,'rep1_wt')
    axis_labels(axd,'rep2_wt')
    axis_labels(axd,'rep1_frp1')
    axis_labels(axd,'rep2_frp1')

    #add legend
    fig_refs['ev_legend'] = add_legend(axd['ev_legend'],
                                    ['blue','orange','green','red'],
                                    ['REP 1 WT','REP 2 WT','REP 1 fpr1','REP 2 fpr1'],
                                      'o')
    fig_refs['g_legend'] = add_legend(axd['g_legend'],
                                        ['orange','purple'],
                                        ['Overlapping \nGenes','Unique \nGenes'],
                                         's')
    #save figure
    fig.savefig('supplemental_fig_variance' + ".png", facecolor="white")
