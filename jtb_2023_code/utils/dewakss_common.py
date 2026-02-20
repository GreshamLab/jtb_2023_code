import scanpy as _sc
import numpy as _np
import dewakss.denoise as _dd
import scipy.sparse as _sps

from scself._noise2self.graph import local_optimal_knn
from joblib import parallel_backend

from ..figure_constants import (
    N_PCS,
    N_NEIGHBORS,
    STANDARDIZE_DEPTH
)
from .standardization import standardize

try:
    from sparse_dot_mkl import csr_matrix
    MKL=True
except ImportError:
    MKL=False


def run_dewakss(
    data,
    n_pcs=N_PCS,
    n_neighbors=N_NEIGHBORS,
    normalize=True,
    n_counts=STANDARDIZE_DEPTH
):
    if "denoised" not in data.uns:
        print("\tNormalizing Data")
        data.X = data.X.astype(float)

        if normalize:
            standardize(data, method='log', n_counts=n_counts)

        if "X_pca" in data.obsm and data.obsm["X_pca"].shape[1] >= max(n_pcs):
            pass
        else:
            print("\tPreprocessing (PCA)")
            if MKL and _sps.isspmatrix_csr(data.X):
                data.X = csr_matrix(data.X)
                _sc.pp.pca(data, n_comps=max(n_pcs))
                data.X = _sps.csr_matrix(data.X)
            else:
                _sc.pp.pca(data, n_comps=max(n_pcs))

        print("\tDEWAKSS:")
        with parallel_backend("loky", inner_max_num_threads=1):
            _sc.pp.neighbors(
                data,
                n_neighbors=max(n_neighbors),
                n_pcs=max(n_pcs)
            )

        _do_dewakss(data, n_pcs=n_pcs, n_neighbors=n_neighbors)

        print("Building denoised AnnData Object")
        data.X = data.layers["denoised"]
        _sc.pp.pca(data, n_comps=100)

        data.X = _np.expm1(data.X)
        data.obsm["denoised_pca"] = data.obsm["X_pca"]

        data.obsp["optimal_distances"] = local_optimal_knn(
            data.obsp["distances"],
            data.obsp["denoised_distances"].getnnz(axis=1)
        )

        del data.layers["denoised"]
        del data.obsm["X_pca"]

    else:
        pass

    return data


def _do_dewakss(
    data,
    n_pcs=N_PCS,
    n_neighbors=N_NEIGHBORS
):

    with parallel_backend("loky", inner_max_num_threads=1):
        denoiseer = _dd.DEWAKSS(
            data,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            settype=_np.float64,
            use_global_err=False,
            modest="max",
            recompute_pca=False,
        )
        denoiseer.fit(data)
        denoiseer.transform(data, copy=False)

    return data
