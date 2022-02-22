import scanpy as _sc
import numpy as _np
import dewakss.denoise as _dd
from joblib import parallel_backend as _parallel_backend

from ..figure_constants import *

def run_dewakss(data, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS):
    
    if 'denoised' not in data.uns:
        
        if 'counts' not in data.layers:
            data.layers['counts'] = data.X.copy()

        print("\tNormalizing Data")
        data.X = data.X.astype(float)
        _sc.pp.normalize_per_cell(data)
        _sc.pp.log1p(data)

        if "X_pca" in data.obsm and data.obsm["X_pca"].shape[1] >= max(n_pcs):
            pass
        else:
            print("\tPreprocessing (PCA)")
            _sc.pp.pca(data, n_comps=max(n_pcs))

        print("\tDEWAKSS:")
        with _parallel_backend("loky", inner_max_num_threads=1):
            _sc.pp.neighbors(data, n_neighbors=max(n_neighbors), n_pcs=max(n_pcs))

            denoiseer = _dd.DEWAKSS(data, n_pcs=n_pcs , n_neighbors=n_neighbors, settype=_np.float64,
                                    use_global_err=False, modest='max', recompute_pca=False)
            denoiseer.fit(data)
            denoiseer.transform(data, copy=False)
            
        data.X = data.layers['counts'].copy()
        
    else:
        pass
        
    return data
