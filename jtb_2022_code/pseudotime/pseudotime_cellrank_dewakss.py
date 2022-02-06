import pandas as _pd
import numpy as _np
import scvelo as _scv
import scanpy as _sc
import anndata as _ad
from scipy.stats import spearmanr as _spearmanr

import cellrank
import dewakss.denoise as _dd
from cellrank.tl.kernels import CytoTRACEKernel

from joblib import parallel_backend as _parallel_backend
from ..utils.pseudotime_common import *


OBSM_COLUMNS = _pd.Index([str(x) + "_" + str(y) for x in N_PCS for y in N_NEIGHBORS])
CELLRANK_DEWAKSS_OBS_COL = 'cellrank_dewakss_pt'


def _do_cellrank_dewakss(data):
    print("\tNormalizing Data")
    data.X = data.X.astype(float)
    _sc.pp.normalize_per_cell(data)
    _sc.pp.log1p(data)
    
    if "X_pca" in data.obsm and data.obsm["X_pca"].shape[1] >= max(N_PCS):
        pass
    else:
        print("\tPreprocessing (PCA)")
        _sc.pp.pca(data, n_comps=max(N_PCS), use_highly_variable=False, zero_center=False)
        
    print("\tDEWAKSS:")
    with _parallel_backend("loky", inner_max_num_threads=1):
        _sc.pp.neighbors(data, n_neighbors=max(N_NEIGHBORS), n_pcs=max(N_PCS))

        denoiseer = _dd.DEWAKSS(data, n_pcs=N_PCS , n_neighbors=N_NEIGHBORS, settype=_np.float64,
                                use_global_err=False, modest='max', recompute_pca=False)
        denoiseer.fit(data)
        denoiseer.transform(data, copy=False)
        
        # Set the dewakss graphs & denoised expression data on default keys
        data.obsp['distances'] = data.obsp['denoised_distances'].copy()
        data.obsp['connectivities'] = data.obsp['denoised_connectivities'].copy()
        data.X = data.layers['denoised'].copy()
        
        _sc.pp.pca(data, n_comps=max(N_PCS), use_highly_variable=False, zero_center=False)
        _sc.tl.umap(data)
    
    # CytoTRACE
    print("\tCytotrace")
    ctk = CytoTRACEKernel(data, layer="X")
    ctk.compute_transition_matrix(threshold_scheme="soft", nu=0.5)
    ctk.compute_projection(basis="umap")

    
def cellrank_dewakss(adata, layer="counts"):
    
    if CELLRANK_DEWAKSS_OBS_COL not in adata.obs:
        adata.obs[CELLRANK_DEWAKSS_OBS_COL] = _np.zeros(adata.shape[0])
    
    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)
    
    for g in adata.obs['Gene'].unique():
        for i in [1, 2]:
            print(f"Processing Experiment {i} [{g}]")
            s_idx = adata.obs['Experiment'] == i
            s_idx &= adata.obs['Gene'] == g

            sdata = get_clean_anndata(adata, s_idx, layer=layer, include_pca=True)
            
            _do_cellrank_dewakss(sdata)
            
            adata.obs.loc[s_idx, CELLRANK_DEWAKSS_OBS_COL] = sdata.obs['ct_pseudotime']
            
            rho = _spearmanr(sdata.obs['Pool'], sdata.obs['ct_pseudotime'])[0]
            print(f"Cellrank PT & Pool rho = {rho}")
            
    return adata
