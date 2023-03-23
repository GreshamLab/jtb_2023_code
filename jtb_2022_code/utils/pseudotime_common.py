import numpy as _np
import anndata as _ad
import pandas as _pd
import scanpy as _sc
import os as _os
from joblib import parallel_backend as _parallel_backend
from scipy.stats import spearmanr as _spearmanr

from inferelator_velocity import calc_velocity

from .adata_common import *
from .projection_common import *
from ..figure_constants import *

PCA_PT = "pca_pt"


def get_pca_pt(data, pca_key='X_pca'):
    # PCA-based pseudotime
    if pca_key != 'X_pca' and pca_key not in data.obsm:
        raise ValueError(f"PCA key {pca_key} is not in data")
        
    elif pca_key == 'X_pca' and pca_key not in data.obsm:
        _sc.pp.pca(data, n_comps=1)
        pca_pt = interval_normalize(data.obsm['X_pca'])
        del data.obsm['X_pca']
        del data.uns['pca']
        del data.varm['PCs']
    else:
        pca_pt = interval_normalize(data.obsm[pca_key][:, 0])
        
    if spearman_rho_pools(data.obs['Pool'], pca_pt) < 0:
        pca_pt = _np.abs(1 - pca_pt)
        
    return pca_pt


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
            data.uns['pca'] = sdata.uns['pca'].copy()
    
    return data


def spearman_rho_pools(pool_vector, pt_vector, average_1_2_pools=True, ignore_na_inf=False):
 
    if not ignore_na_inf and (any(_pd.isna(pt_vector)) or any(_np.isinf(pt_vector))):
        return _np.nan
    
    if _np.nanmin(pt_vector) == _np.nanmax(pt_vector):
        return 0
 
    if average_1_2_pools:
        pool_vector = pool_vector.copy()
        pool_vector[pool_vector == 2] = 1

    return _spearmanr(pool_vector, pt_vector)[0]


def calc_rhos(adata, obsm_key):
    n = adata.obsm[obsm_key].shape[1]
    p = adata.obs["Pool"]
    return list(map(lambda y: (adata.obsm[obsm_key].columns[y],
                               spearman_rho_pools(p, adata.obsm[obsm_key].iloc[:, y])), 
                    range(n)))
    
    
def spearman_rho_grid(data, obsm_key, uns_key, average_1_2_pools=True):
       
    df = data.apply_to_expts(calc_rhos, obsm_key)
    df = _pd.DataFrame(df)
    df.columns = df.iloc[0, :].apply(lambda x: x[0])
    df.columns = _pd.MultiIndex.from_tuples(list(map(lambda x: x.split("_"), df.columns)))
    df.columns.set_names(['PCs', 'Neighbors'], inplace=True)
    df.index = _pd.MultiIndex.from_tuples(data.expts)
    df = df.applymap(lambda x: x[1])
    data.all_data.uns[uns_key] = df
    
    for k in data.expts:
        data.expt_data[k].uns[uns_key] = df.loc[k, :].copy()
    
    return data

def add_time_rho(data_obj):
    data.all_data.uns['rho']['time'] = 0.
    data.all_data.uns['denoised_rho']['time'] = 0.

    for k in data.expts:
        expt_ref = data.expt_data[k]
        data.all_data.uns['rho'].loc[k, 'time'] = spearman_rho_pools(expt_ref.obs['Pool'], expt_ref.obs['program_rapa_time'])
        data.all_data.uns['denoised_rho'].loc[k, 'time'] = spearman_rho_pools(expt_ref.obs['Pool'], expt_ref.obs['program_rapa_time_denoised'])
        
    return data
