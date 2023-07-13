from jtb_2022_code.pseudotime.pseudotime_palantir import palantir_grid_search, PALANTIR_OBSM_COL
import anndata as _ad

ADATA_FILE = "2021_RAPA_TIMECOURSE.h5ad"

if __name__ == "__main__":
    adata = _ad.read(ADATA_FILE)
    adata = palantir_grid_search(adata, layer="X", dcs_equal_pcs=True)
    #adata.write("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_PALANTIR.h5ad")
    adata.obsm[PALANTIR_OBSM_COL].to_csv("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_PALANTIR_PCDC.tsv.gz", sep="\t")
