import pandas as _pd
import numpy as _np
import scvelo as _scv
import scanpy as _sc
import anndata as _ad
from scipy.stats import spearmanr as _spearmanr

from joblib import parallel_backend as _parallel_backend
from jtb_2022_code.utils.pseudotime_common import *
from jtb_2022_code.utils.dewakss_common import run_dewakss

DPT_OBS_COL = 'dpt_pseudotime'


def _do_dpt_dewakss(data, n_dcs=15):
    
    run_dewakss(data)
        
    do_pca_pt(data)

    # Set the dewakss graphs & denoised expression data on default keys
    data.obsp['distances'] = data.obsp['denoised_distances'].copy()
    data.obsp['connectivities'] = data.obsp['denoised_connectivities'].copy()
    data.X = data.layers['denoised'].copy()
    
    # Get a root cell
    data.uns['iroot'] = data.obs[PCA_PT].argmin()
    print(f"\tSelected root cell: {data.uns['iroot']}")

    with _parallel_backend("loky", inner_max_num_threads=1):
        _sc.tl.diffmap(data, n_comps=n_dcs)
        _sc.tl.dpt(data, n_dcs=n_dcs)

    
def _dpt_by_group(adata, n_comps=15, layer="counts"):
    
    pt = _np.zeros(adata.shape[0])
    
    for g in adata.obs['Gene'].unique():
        for i in [1, 2]:
            print(f"Processing Experiment {i} [{g}]")
            s_idx = adata.obs['Experiment'] == i
            s_idx &= adata.obs['Gene'] == g

            sdata = get_clean_anndata(adata, s_idx, layer=layer, include_pca=True)
            _do_dpt_dewakss(sdata, n_dcs=n_comps)
            
            pt[s_idx] = sdata.obs[DPT_OBS_COL]
            
            rho = spearman_rho_pools(sdata.obs['Pool'], sdata.obs[DPT_OBS_COL])
            print(f"Scanpy DPT PT & Pool rho = {rho}")
            
    return pt


def dpt_dewakss(adata, layer="counts"):
        
    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)
    adata.obs[DPT_OBS_COL] = _dpt_by_group(adata, layer=layer)

    return adata
