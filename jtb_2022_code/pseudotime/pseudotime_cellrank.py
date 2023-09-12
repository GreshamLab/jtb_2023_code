import pandas as _pd
import numpy as _np
import scvelo as _scv
import scanpy as _sc
from scipy.stats import spearmanr as _spearmanr

import cellrank
from cellrank.tl.kernels import CytoTRACEKernel

from joblib import parallel_backend as _parallel_backend
from jtb_2022_code.utils.pseudotime_common import do_pca_on_groups
from jtb_2022_code.utils.adata_common import get_clean_anndata

from jtb_2022_code.figure_constants import (
    N_PCS,
    N_NEIGHBORS
)


OBSM_COLUMNS = _pd.Index([
    str(x) + "_" + str(y)
    for x in N_PCS for y in N_NEIGHBORS
])
CELLRANK_OBSM_COL = "cellrank_pt"


def do_cellrank(data, layer="Ms"):
    ctk = CytoTRACEKernel(data, layer=layer)
    ctk.compute_transition_matrix(threshold_scheme="soft", nu=0.5)
    ctk.compute_projection(basis="umap")


def _do_cellrank_preprocess(data, npcs, nns):
    print("\tNormalizing Data")
    data.X = data.X.astype(float)
    _sc.pp.normalize_per_cell(data)
    _sc.pp.log1p(data)

    if "X_pca" in data.obsm and data.obsm["X_pca"].shape[1] >= npcs:
        print("\tPreprocessing (Neighbors)")
    else:
        print("\tPreprocessing (PCA & Neighbors)")
        _sc.pp.pca(
            data,
            n_comps=npcs,
            use_highly_variable=False,
            zero_center=False
        )

    with _parallel_backend("loky", inner_max_num_threads=1):
        _sc.pp.neighbors(data, n_neighbors=nns, n_pcs=npcs)
        _sc.tl.umap(data)

    # use scVelo's `moments` function for imputation
    # note that hack we're using here:
    # we're copying our `.X` matrix into the layers because
    # that's where `scv.tl.moments`
    # expects to find counts for imputation
    print("\tImputing")
    data.layers["spliced"] = data.X.copy()
    data.layers["unspliced"] = data.X.copy()
    _scv.pp.moments(data, n_pcs=npcs, n_neighbors=nns)

    # CytoTRACE
    print("\tCytotrace")
    do_cellrank(data)


def _cellrank_by_group(adata, npc=50, nns=15, layer="counts"):
    pt = _np.zeros(adata.shape[0])

    for g in adata.obs["Gene"].unique():
        for i in [1, 2]:
            print(f"Processing Experiment {i} [{g}]")
            s_idx = adata.obs["Experiment"] == i
            s_idx &= adata.obs["Gene"] == g

            sdata = get_clean_anndata(
                adata,
                s_idx,
                layer=layer,
                include_pca=True
            )
            _do_cellrank_preprocess(sdata, npc, nns)
            do_cellrank(sdata)

            pt[s_idx] = sdata.obs["ct_pseudotime"]

            rho = _spearmanr(sdata.obs["Pool"], sdata.obs["ct_pseudotime"])[0]
            print(f"Experiment {i} [{g}] Cellrank rho = {rho}")

    return pt


def cellrank_grid_search(adata, layer="counts"):
    if CELLRANK_OBSM_COL not in adata.obsm:
        adata.obsm[CELLRANK_OBSM_COL] = _pd.DataFrame(
            _np.zeros((adata.shape[0], len(OBSM_COLUMNS))),
            columns=OBSM_COLUMNS,
            index=adata.obs_names,
        )

    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)

    for npc in N_PCS:
        for nn in N_NEIGHBORS:
            print(f"Grid search: {npc} PCs, {nn} Neighbors")
            obsm_column = str(npc) + "_" + str(nn)
            cellrank_pt = _cellrank_by_group(
                adata,
                layer=layer,
                npc=npc,
                nns=nn
            )
            adata.obsm[CELLRANK_OBSM_COL].loc[:, obsm_column] = cellrank_pt

    return adata
