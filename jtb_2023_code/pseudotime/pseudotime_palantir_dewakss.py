import numpy as _np
import scanpy as _sc

from ..utils.pseudotime_common import (
    spearman_rho_pools,
    do_pca_on_groups
)
from ..utils.adata_common import get_clean_anndata
from ..utils.dewakss_common import run_dewakss
from .pseudotime_palantir import _do_palantir, PALANTIR_OBSM_COL

from jtb_2023_code.figure_constants import N_PCS

PALANTIR_DEWAKSS_OBSM_COL = "denoised_" + PALANTIR_OBSM_COL


def do_palantir_denoised(adata, n_pcs, nns, n_comps=15):
    if PALANTIR_DEWAKSS_OBSM_COL in adata.obs:
        return adata

    ddata = get_clean_anndata(
        adata,
        layer="denoised",
        include_pca=True,
        replace_neighbors_with_dewakss=True
    )

    if "X_pca" not in ddata.obsm:
        _sc.pp.pca(ddata, n_comps=n_pcs)

    _do_palantir(ddata, n_pcs, ncomps=n_comps, nns=nns)
    adata.obs[PALANTIR_DEWAKSS_OBSM_COL] = ddata.obs[PALANTIR_OBSM_COL]

    return adata


def _palantir_by_group(adata, n_comps=15, layer="counts", dcs_equal_pcs=True):
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
            run_dewakss(sdata)

            n_pcs = int(sdata.uns["denoised"]["params"]["n_pcs"])
            n_neighbors = int(sdata.obs["denoised_n"].median())
            n_comps = n_pcs if dcs_equal_pcs else n_comps

            do_palantir_denoised(sdata, n_pcs, n_neighbors, n_comps=n_comps)

            pt[s_idx] = sdata.obs[PALANTIR_DEWAKSS_OBSM_COL]

            rho = spearman_rho_pools(
                sdata.obs["Pool"], sdata.obs[PALANTIR_DEWAKSS_OBSM_COL]
            )
            print(f"Experiment {i} [{g}] Scanpy DPT rho = {rho}")

    return pt


def palantir_dewakss(adata, layer="counts", dcs_equal_pcs=True):
    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)
    adata.obs[PALANTIR_DEWAKSS_OBSM_COL] = _palantir_by_group(
        adata, layer=layer, dcs_equal_pcs=dcs_equal_pcs
    )

    return adata
