import numpy as _np
import anndata as _ad
import pandas as _pd
import scanpy as _sc
import os as _os
from joblib import parallel_backend as _parallel_backend
from scipy.stats import spearmanr as _spearmanr

from inferelator_prior.velocity.calc import calc_velocity
from inferelator_prior.velocity.times import assign_times_from_pseudotime, assign_times_from_pseudotime_sliding

from .adata_common import *
from .projection_common import *
from ..figure_constants import *

PCA_PT = "pca_pt"


def do_pca_pt(data, pt_key=PCA_PT, pca_key='X_pca'):
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
        
    data.obs[pt_key] = pca_pt
        
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


def calculate_times_velocities(data,
                               transform_expr=None,
                                   pt_obs_key=PCA_PT, 
                               time_obs_key=None, 
                               layer="X", 
                               distance_key="distances", 
                               nn=max(N_NEIGHBORS), 
                               force=False):
    
    if time_obs_key is None:
        time_obs_key = "time_" + pt_obs_key
    
    layer_out = layer + "_velocity"
    
    if time_obs_key not in data.obs or force:
        rho = spearman_rho_pools(data.obs['Pool'], data.obs[pt_obs_key])
        do_time_assign_by_pool(data, pt_obs_key=pt_obs_key, time_obs_key=time_obs_key, force=force)
    
    if layer_out not in data.layers or force:
        lref = data.X if layer == "X" else data.layers[layer]
        lref = transform_expr(lref) if transform_expr is not None else lref
        
        data.layers[layer_out] = calc_velocity(lref, 
                                               data.obs[time_obs_key].values, 
                                               data.obsp[distance_key],
                                               nn, wrap_time=None)
    
    return data


def do_time_assign_by_pool(adata, pt_obs_key='pca_pt', time_obs_key=None, quantiles=(0.05, 0.95), 
                           force=False, max_outlier=10):
    
    if time_obs_key is None:
        time_obs_key = "time_" + pt_obs_key
        
    if time_obs_key in adata.obs and not force:
        return adata
    
    pseudotime = adata.obs[pt_obs_key].values
    labels = adata.obs['Pool'].values
    
    if spearman_rho_pools(labels, pseudotime) < 0:
        pseudotime = _np.abs(pseudotime - _np.max(pseudotime))
        
    real_time, outlier_mask = assign_times_from_pseudotime_sliding(pseudotime, 
                                                                   labels, 
                                                                   list(range(1, 9)), 
                                                                   POOL_TIMES, 
                                                                   edges=quantiles,
                                                                   mask_outliers=True,
                                                                   trim_outliers=False)
    
    adata.obs[time_obs_key] = real_time
    
    if max_outlier is not None:
    
        for g, start_time, stop_time in POOL_TIMES:

            _g_outlier = real_time > (stop_time + max_outlier)
            _g_outlier |= real_time < (start_time - max_outlier)
            _g_outlier &= outlier_mask
            _g_outlier &= adata.obs['Pool'] == g
            
            adata.obs.loc[_g_outlier, time_obs_key] = _np.nan
            
    else:
        
        adata.obs.loc[outlier_mask, time_obs_key] = _np.nan
    
    return adata
