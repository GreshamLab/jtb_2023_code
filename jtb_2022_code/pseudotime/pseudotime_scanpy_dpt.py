import pandas as _pd
import numpy as _np
import scanpy as _sc

from joblib import parallel_backend as _parallel_backend
from jtb_2022_code.utils.pseudotime_common import (
    spearman_rho_pools
)

from ..utils.projection_common import do_pc1_min_cell
from jtb_2022_code.utils.pseudotime_common import do_pca_on_groups
from jtb_2022_code.utils.adata_common import get_clean_anndata

from jtb_2022_code.figure_constants import (
    N_PCS,
    N_NEIGHBORS,
    VERBOSE
)

OBSM_COLUMNS = _pd.Index([
    str(x) + "_" + str(y)
    for x in N_PCS for y in N_NEIGHBORS
])
DPT_OBS_COL = "dpt_pseudotime"


def do_dpt(data, n_dcs=15):
    # Get a root cell
    data.uns["iroot"] = do_pc1_min_cell(data)

    if VERBOSE:
        print(f"\tSelected root cell: {data.uns['iroot']}")

    with _parallel_backend("loky", inner_max_num_threads=1):
        _sc.tl.diffmap(data, n_comps=n_dcs)
        _sc.tl.dpt(data, n_dcs=n_dcs)


def _do_dpt_preprocess(data, npcs, nns):
    data.X = data.X.astype(float)
    _sc.pp.normalize_per_cell(data)
    _sc.pp.log1p(data)

    if "X_pca" in data.obsm and data.obsm["X_pca"].shape[1] >= npcs:
        pass
    else:
        print("\tPreprocessing PCA")
        _sc.pp.pca(data, n_comps=npcs)

    with _parallel_backend("loky", inner_max_num_threads=1):
        _sc.pp.neighbors(data, n_neighbors=nns, n_pcs=npcs)


def _dpt_by_group(adata, npc=50, nns=15, n_comps=15, layer="counts"):
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
            _do_dpt_preprocess(sdata, npc, nns)
            do_dpt(sdata, n_dcs=n_comps)

            pt[s_idx] = sdata.obs[DPT_OBS_COL]

            rho = spearman_rho_pools(
                sdata.obs["Pool"],
                sdata.obs[DPT_OBS_COL]
            )
            print(f"Experiment {i} [{g}] Scanpy DPT rho = {rho}")

    return pt


def dpt_grid_search(adata, layer="counts", nc=15, dcs_equal_pcs=False):
    if DPT_OBS_COL not in adata.obsm:
        adata.obsm[DPT_OBS_COL] = _pd.DataFrame(
            _np.zeros((adata.shape[0], len(OBSM_COLUMNS))),
            columns=OBSM_COLUMNS,
            index=adata.obs_names,
        )

    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)

    for npc in N_PCS:
        for nn in N_NEIGHBORS:
            print(f"Grid search: {npc} PCs, {nn} Neighbors")
            obsm_column = str(npc) + "_" + str(nn)
            dpt_pt = _dpt_by_group(
                adata,
                layer=layer,
                npc=npc,
                nns=nn,
                n_comps=npc if dcs_equal_pcs else nc,
            )
            adata.obsm[DPT_OBS_COL].loc[:, obsm_column] = dpt_pt

    return adata
