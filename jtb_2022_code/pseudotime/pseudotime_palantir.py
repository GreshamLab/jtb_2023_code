import pandas as _pd
import numpy as _np
import scanpy as _sc
import anndata as _ad
from scipy.stats import spearmanr as _spearmanr

import palantir
import harmony 

from joblib import parallel_backend as _parallel_backend
from ..utils.pseudotime_common import *

# Reset random seed
_np.random.seed(42)

OBSM_COLUMNS = _pd.Index(["_".join((str(x), str(z))) 
                          for x in N_PCS for z in N_NEIGHBORS])
PALANTIR_OBSM_COL = 'palantir_pt'


def _do_palantir(data, npcs, ncomps=15, nns=30):

    # Diffusion map on PCs
    with _parallel_backend("loky", inner_max_num_threads=1):

        print("\tDiffusion Mapping")
        _pca_df = _pd.DataFrame(data.obsm['X_pca'][:, 0:npcs], index=data.obs_names)
        dm_res = palantir.utils.run_diffusion_maps(_pca_df, n_components=ncomps, knn=nns)

        print("\tMultiscale Space")
        ms_data = palantir.utils.determine_multiscale_space(dm_res)
        fdl = harmony.plot.force_directed_layout(dm_res['kernel'], data.obs_names)

        # Get pseudotimes based on PCA
        do_pca_pt(data)
        start_cell = ms_data.index[data.obs['pca_pt'].argmin()]

        print(f"\tPalantir (Start Cell: {start_cell})")
        pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=500, knn=nns)
        data.obs[PALANTIR_OBSM_COL] = pr_res.pseudotime

    return data


def _do_palantir_preprocessing(data, npcs):
    print("\tNormalizing Data")
    data.X = data.X.astype(float)
    _sc.pp.normalize_per_cell(data)
    _sc.pp.log1p(data)
   
    if "X_pca" in data.obsm and data.obsm["X_pca"].shape[1] >= npcs:
        pass
    else:
        print("\tPreprocessing (PCA)")
        _sc.pp.pca(data, n_comps=npcs)    

    
def _palantir_by_group(adata, layer="counts", npc=50, n_comps=15, nns=30):
    
    pt = _np.zeros(adata.shape[0])
    
    for g in adata.obs['Gene'].unique():
        for i in [1, 2]:
            print(f"Processing Experiment {i} [{g}]")
            s_idx = adata.obs['Experiment'] == i
            s_idx &= adata.obs['Gene'] == g

            sdata = get_clean_anndata(adata, s_idx, layer=layer, include_pca=True)
            _do_palantir_preprocessing(sdata, npc)
            _do_palantir(sdata, npc, n_comps, nns)
            
            pt[s_idx] = sdata.obs[PALANTIR_OBSM_COL]
            
            rho = _spearmanr(sdata.obs['Pool'], sdata.obs[PALANTIR_OBSM_COL])[0]
            print(f"Palantir PT & Pool rho = {rho}")
            
    return pt

def palantir_grid_search(adata, layer="counts", n_pcs=None, nc=15, dcs_equal_pcs=False):
    
    n_pcs = N_PCS if n_pcs is None else n_pcs
    
    if PALANTIR_OBSM_COL not in adata.obsm:
        adata.obsm[PALANTIR_OBSM_COL] = _pd.DataFrame(_np.zeros((adata.shape[0],
                                                                 len(OBSM_COLUMNS))),
                                                      columns=OBSM_COLUMNS,
                                                      index=adata.obs_names)
    
    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)

    for npc in n_pcs:
        for nn in N_NEIGHBORS:
            print(f"Grid search: {npc} PCs, {nc} Comps, {nn} Neighbors")
            obsm_column = "_".join((str(npc), str(nn)))
            palantir_pt = _palantir_by_group(adata, layer=layer, npc=npc, nns=nn,
                                             n_comps=npc if dcs_equal_pcs else nc)
            adata.obsm[PALANTIR_OBSM_COL].loc[:, obsm_column] = palantir_pt

    return adata
