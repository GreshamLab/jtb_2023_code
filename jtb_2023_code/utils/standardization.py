import scanpy as _sc

from jtb_2023_code.figure_constants import (
    STANDARDIZE_DEPTH,
    STANDARDIZE_V1
)
from scself import (
    TruncRobustScaler,
    standardize_data
)

def standardize(
    adata,
    n_counts=STANDARDIZE_DEPTH,
    method='log',
    standardize_v1=STANDARDIZE_V1
):

    if 'counts' not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()

    # Original depth standardization
    if standardize_v1:
        _sc.pp.normalize_total(adata, target_sum=STANDARDIZE_DEPTH)

        if (method == 'log') or (method == 'log_scale'):
            _sc.pp.log1p(adata)
        if (method == 'scale') or (method == 'log_scale'):
            scaler = TruncRobustScaler(with_centering=False)
            adata.X = scaler.fit_transform(adata.X)
            adata.var['scale_factor'] = scaler.scale_

    else:
        standardize_data(
            adata,
            method=method,
            target_sum=n_counts,
            subset_genes_for_depth=~(
                adata.var['RP'] |
                adata.var['RiBi']
            )
        )

        if (method == 'scale') or (method == 'log_scale'):
            adata.var['scale_factor'] = adata.var['X_scale_factor']

    return adata
