import numpy as _np
import anndata as _ad
import pandas as _pd
import scanpy as _sc
import os as _os
from joblib import parallel_backend as _parallel_backend
from umap.umap_ import fuzzy_simplicial_set as _fuzzy_simplicial_set
from umap.umap_ import nearest_neighbors as _nearest_neighbors
import scipy.sparse as _sps
from pandas.api.types import is_numeric_dtype as _is_numeric
from scipy.stats import spearmanr as _spearmanr

from .adata_common import *
from .projection_common import *


def do_pca_pt(data, key="pca_pt"):
    # PCA-based pseudotime
    if 'X_pca' not in data.obsm:
        _sc.pp.pca(data, n_comps=1)
        pca_pt = interval_normalize(data.obsm['X_pca'])
        del data.obsm['X_pca']
        del data.uns['pca']
        del data.varm['PCs']
    else:
        pca_pt = interval_normalize(data.obsm['X_pca'][:, 0])
        
    if spearman_rho_pools(data.obs['Pool'], pca_pt) < 0:
        pca_pt = _np.abs(1 - pca_pt)
        
    data.obs[key] = pca_pt
        
    return data


def do_pca_on_groups(data, npcs, layer="counts"):
    
    data.obsm["X_pca"] = _np.zeros((data.shape[0], npcs))
    
    for g in data.obs['Gene'].unique():
        for i in [1, 2]:
            print(f"PCA Experiment {i} [{g}]")
            s_idx = data.obs['Experiment'] == i
            s_idx &= data.obs['Gene'] == g

            sdata = get_clean_anndata(data, s_idx, layer=layer)
            do_pca(sdata, npcs)
            
            data.obsm["X_pca"][s_idx, :] = sdata.obsm["X_pca"]
    
    return data


def spearman_rho_pools(pool_vector, pt_vector, average_1_2_pools=True):
    
    if average_1_2_pools:
        pool_vector = pool_vector.copy()
        pool_vector[pool_vector == 2] = 1
    
    return _spearmanr(pool_vector, pt_vector)[0]
    