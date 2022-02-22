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
from ..utils.adata_common import get_clean_anndata
from ..utils.dewakss_common import run_dewakss
from .pseudotime_cellrank import do_cellrank, CELLRANK_OBSM_COL

OBSM_COLUMNS = _pd.Index([str(x) + "_" + str(y) for x in N_PCS for y in N_NEIGHBORS])
CELLRANK_DEWAKSS_OBS_COL = "denoised_" + CELLRANK_OBSM_COL


def do_cytotrace_denoised(adata):
    
    if CELLRANK_DEWAKSS_OBS_COL in adata.obs:
        return adata
    
    ddata = get_clean_anndata(adata, 
                              layer='denoised', 
                              include_pca=True, 
                              replace_neighbors_with_dewakss=True)
    _sc.tl.umap(ddata)
    do_cellrank(ddata, layer="X")
    
    adata.obs[CELLRANK_DEWAKSS_OBS_COL] = ddata.obs['ct_pseudotime'].copy()

    
def cellrank_dewakss(adata, layer="counts"):
    
    if CELLRANK_DEWAKSS_OBS_COL not in adata.obs:
        adata.obs[CELLRANK_DEWAKSS_OBS_COL] = _np.zeros(adata.shape[0])
    else:
        return adata
    
    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)
    
    for g in adata.obs['Gene'].unique():
        for i in [1, 2]:
            print(f"Processing Experiment {i} [{g}]")
            s_idx = adata.obs['Experiment'] == i
            s_idx &= adata.obs['Gene'] == g

            sdata = get_clean_anndata(adata, s_idx, layer=layer, include_pca=True)
            
            run_dewakss(data)
            do_cytotrace_denoised(data)
            
            adata.obs.loc[s_idx, CELLRANK_DEWAKSS_OBS_COL] = sdata.obs['ct_pseudotime']
            
            rho = _spearmanr(sdata.obs['Pool'], sdata.obs['ct_pseudotime'])[0]
            print(f"Experiment {i} [{g}] Cellrank rho = {rho}")
            
    return adata
