import anndata as ad
import scanpy as sc
import jtb_2022_code
from scipy.stats import zscore
from inferelator_prior.velocity.programs import sparse_PCA

data = ad.read(jtb_2022_code.figure_constants.RAPA_SINGLE_CELL_EXPR_FILE)

for i in range(1, 3):
    for j in ["WT", "fpr1"]:
        expt = data[(data.obs['Experiment'] == i) & (data.obs['Gene'] == j), ].copy()
        expt.X = expt.X.astype(float).A
        sc.pp.normalize_per_cell(expt)
        sc.pp.log1p(expt)
        sc.pp.filter_genes(expt, min_cells=10)
        expt.X = zscore(expt.X)
        sc.pp.pca(expt, n_comps=100)
        
        sparse_PCA(expt, normalize=False)
        
        d_key = f"{i}_{j}_sparse_pca"
        data.uns[d_key] = expt.uns['sparse_pca']
        data.uns[d_key]['obsm'] = expt.obsm['X_sparsepca']
        data.uns[d_key]['varm'] = expt.varm['X_sparsepca']
        
data.write("/scratch/cj59/RAPA/2021_RAPA_SparsePCA.h5ad")