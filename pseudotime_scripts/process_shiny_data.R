require("readr")

data.path <- '/scratch/cj59/RAPA/'
wide.file.names <- list(
    expression="2021_RAPA_EXPRESSION.tsv.gz",
    denoised="2021_RAPA_DENOISED.tsv.gz",
    velocity="2021_RAPA_VELOCITY.tsv.gz",
    tfa_expression="2021_RAPA_TFA_EXPRESSION.tsv.gz",
    tfa_denoised="2021_RAPA_TFA_DENOISED.tsv.gz",
    tfa_velocity="2021_RAPA_TFA_VELOCITY.tsv.gz",
    tfa_decay_constant="2021_RAPA_TFA_DECAY_20MIN.tsv.gz",
    tfa_decay_latent="2021_RAPA_TFA_DECAY_LATENT.tsv.gz"
)

agg.file.names <- list(
    rapamycin="2021_RAPA_AGGREGATE_RAPAMYCIN.tsv.gz",
    cell_cycle="2021_RAPA_AGGREGATE_CELL_CYCLE.tsv.gz"
)
    
metadata.file.names <- list(
    metadata="2021_RAPA_METADATA.tsv.gz",
    gene_metadata="2021_RAPA_GENE_METADATA.tsv.gz"
)

load.tsv <- function(x) {readr::read_tsv(x)}