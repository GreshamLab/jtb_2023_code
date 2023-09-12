import numpy as _np
import scanpy as _sc

from ..utils.pseudotime_common import (
    spearman_rho_pools,
    do_pca_on_groups
)
from ..utils.adata_common import get_clean_anndata
from ..utils.dewakss_common import run_dewakss
from .pseudotime_scanpy_dpt import do_dpt, DPT_OBS_COL

from jtb_2023_code.figure_constants import (
    N_PCS
)

DPT_DEWAKSS_OBS_COL = "denoised_" + DPT_OBS_COL


def do_dpt_denoised(adata, n_comps=15):
    if DPT_DEWAKSS_OBS_COL in adata.obs:
        return adata

    ddata = get_clean_anndata(
        adata,
        layer="denoised",
        include_pca=True,
        replace_neighbors_with_dewakss=True
    )

    if "X_pca" not in ddata.obsm:
        _sc.pp.pca(ddata, n_comps=adata.uns["denoised"]["params"]["n_pcs"])

    do_dpt(ddata, n_dcs=n_comps)

    adata.obs[DPT_DEWAKSS_OBS_COL] = ddata.obs[DPT_OBS_COL]

    return adata


def _dpt_by_group(adata, n_comps=15, layer="counts", dcs_equal_pcs=False):
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

            n_comps = (
                int(sdata.uns["denoised"]["params"]["n_pcs"])
                if dcs_equal_pcs
                else n_comps
            )
            do_dpt_denoised(sdata, n_comps=n_comps)

            pt[s_idx] = sdata.obs[DPT_DEWAKSS_OBS_COL]

            rho = spearman_rho_pools(
                sdata.obs["Pool"],
                sdata.obs[DPT_DEWAKSS_OBS_COL]
            )
            print(f"Experiment {i} [{g}] Scanpy DPT rho = {rho}")

    return pt


def dpt_dewakss(adata, layer="counts", dcs_equal_pcs=False):
    do_pca_on_groups(adata, _np.max(N_PCS), layer=layer)
    adata.obs[DPT_DEWAKSS_OBS_COL] = _dpt_by_group(adata, layer=layer)

    return adata
