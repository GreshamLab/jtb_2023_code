from jtb_2023_code.pseudotime.pseudotime_cellrank import cellrank_grid_search, CELLRANK_OBSM_COL
import anndata as _ad

ADATA_FILE = "2021_RAPA_TIMECOURSE.h5ad"

if __name__ == "__main__":
    adata = _ad.read(ADATA_FILE)
    adata = cellrank_grid_search(adata, layer="X")
    #adata.write("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_CELLRANK.h5ad")
    adata.obsm[CELLRANK_OBSM_COL].to_csv("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_CELLRANK.tsv.gz", sep="\t")
    