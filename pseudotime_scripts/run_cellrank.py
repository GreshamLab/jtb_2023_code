from jtb_2022_code.pseudotime.pseudotime_cellrank import cellrank_grid_search
import anndata as _ad

ADATA_FILE = "2021_RAPA_TIMECOURSE.h5ad"

if __name__ == "__main__":
    adata = _ad.read(ADATA_FILE)
    adata = cellrank_grid_search(adata, layer="X")
    adata.write("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_CELLRANK.h5ad")