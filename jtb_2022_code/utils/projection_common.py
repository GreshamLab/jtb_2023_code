import numpy as _np
import pandas as _pd
import pandas.api.types as _pat
import os as _os
import anndata as _ad
import scanpy as _sc
import scipy as _sp
from joblib import parallel_backend as _parallel_backend

from .adata_common import *
from .figure_common import *
from ..figure_constants import *


def do_pca(adata, n_pcs):
    # If there is no projection in the data object, make it
    if 'X_pca' not in adata.obsm:
        if VERBOSE:
            print(f"Calculating PCA with {n_pcs} PCs")
        
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        
        # Normalize and log1p
        adata.X = adata.X.astype(float)
        _sc.pp.normalize_per_cell(adata)
        _sc.pp.log1p(adata)

        # Do PCA, kNN, and UMAP with constant parameters
        _sc.pp.pca(adata, n_comps=n_pcs)
        
        # Restore count data to X
        adata.X = adata.layers['counts'].copy()
       
    return adata


def do_umap(adata, n_pcs, nns, min_dist=0.1):
    # If there is no projection in the data object, make it
    if 'X_pca' not in adata.obsm:
        do_pca(adata, n_pcs)
    
    if 'X_umap' not in adata.obsm:
        if VERBOSE:
            print(f"Projecting UMAP with {n_pcs} PCs,",
                  f"{nns} Neighbors,",
                  f"and {min_dist} min_dist")
        
        _sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=nns)
        _sc.tl.umap(adata, min_dist=min_dist)
        
    return adata
