import scanpy as _sc

from .adata_common import (
    get_clean_anndata
)
from ..figure_constants import (
    VERBOSE,
    N_PCS
)


def do_pca(adata, n_pcs, normalize=False):
    # If there is no projection in the data object, make it
    if "X_pca" not in adata.obsm:
        if VERBOSE:
            print(f"Calculating PCA with {n_pcs} PCs")

        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        if normalize:
            adata.X = adata.X.astype(float)
            _sc.pp.normalize_per_cell(adata)
            _sc.pp.log1p(adata)

        # Do PCA, kNN, and UMAP with constant parameters
        _sc.pp.pca(adata, n_comps=n_pcs)

    return adata


def do_pc1_min_cell(adata):

    if "X_pca" not in adata.obsm:
        do_pca(adata, 1)

    return adata.obsm['X_pca'][:, 0].ravel().argmin()


def do_umap(adata, n_pcs, nns, min_dist=0.1):
    # If there is no projection in the data object, make it
    if "X_pca" not in adata.obsm:
        do_pca(adata, n_pcs)

    if "X_umap" not in adata.obsm:
        if VERBOSE:
            print(
                f"Projecting UMAP with {n_pcs} PCs,",
                f"{nns} Neighbors,",
                f"and {min_dist} min_dist",
            )

        _sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=nns)
        _sc.tl.umap(adata, min_dist=min_dist)

    return adata


def do_denoised_pca(adata, n_pcs=max(N_PCS), force=False):
    if "denoised_pca" in adata.obsm and not force:
        return adata

    dn_data = get_clean_anndata(adata, layer="denoised")
    do_pca(dn_data, n_pcs, normalize=False)

    adata.obsm["denoised_pca"] = dn_data.obsm["X_pca"].copy()
    adata.uns["denoised_pca"] = dn_data.uns["pca"].copy()
    adata.varm["denoised_PCs"] = dn_data.varm["PCs"].copy()

    return adata
