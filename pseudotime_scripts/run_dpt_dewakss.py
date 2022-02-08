from jtb_2022_code.pseudotime.pseudotime_scanpy_dpt_dewakss import dpt_dewakss, DPT_OBS_COL
import anndata as _ad

ADATA_FILE = "2021_RAPA_TIMECOURSE.h5ad"

if __name__ == "__main__":
    adata = _ad.read(ADATA_FILE)
    adata = dpt_dewakss(adata, layer="X")
    adata.write("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_DPT_DEWAKSS.h5ad")
    adata.obs[DPT_OBS_COL].to_csv("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_DPT_DEWAKSS.tsv", sep="\t")