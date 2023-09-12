from jtb_2023_code.pseudotime.pseudotime_scanpy_dpt import dpt_grid_search, DPT_OBS_COL
import anndata as _ad

ADATA_FILE = "2021_RAPA_TIMECOURSE.h5ad"

if __name__ == "__main__":
    adata = _ad.read(ADATA_FILE)
    adata = dpt_grid_search(adata, layer="X", dcs_equal_pcs=True)
    #adata.write("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_DPT.h5ad")
    adata.obsm[DPT_OBS_COL].to_csv("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_DPT_PCDC.tsv.gz", sep="\t")