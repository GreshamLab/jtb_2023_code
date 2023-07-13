from jtb_2022_code.pseudotime.pseudotime_palantir_dewakss import palantir_dewakss, PALANTIR_DEWAKSS_OBSM_COL
import anndata as _ad

ADATA_FILE = "2021_RAPA_TIMECOURSE.h5ad"

if __name__ == "__main__":
    adata = _ad.read(ADATA_FILE)
    adata = palantir_dewakss(adata, layer="X")
    #adata.write("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_PALANTIR_DEWAKSS.h5ad")
    adata.obs[PALANTIR_DEWAKSS_OBSM_COL].to_csv("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_PALANTIR_DEWAKSS.tsv.gz", sep="\t")
