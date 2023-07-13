from inferelator import (inferelator_workflow,
                         inferelator_verbose_level,
                         MPControl)

from inferelator.velocity_workflow import VelocityWorkflow
from inferelator.preprocessing.single_cell import normalize_expression_to_median

import gc
import anndata as ad

from jtb_2022_code.figure_constants import (
    INFERELATOR_DATA_FILE,
    INFERELATOR_PRIORS_FILE,
    INFERELATOR_TF_NAMES_FILE,
    INFERELATOR_GOLD_STANDARD_FILE,
    EXPRESSION_TSV_FILES,
    LATENT_DATA_FILES,
    METADATA_DATA_FILES,
    ScratchFile
)

inferelator_verbose_level(1)

INPUT_DIR = str(ScratchFile(""))
OUTPUT_PATH = str(ScratchFile(""))

REGRESSION = 'bbsr'

def set_up_workflow(wkf):

    wkf.set_file_paths(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_PATH,
        priors_file=INFERELATOR_PRIORS_FILE,
        tf_names_file=INFERELATOR_TF_NAMES_FILE,
        gold_standard_file=INFERELATOR_GOLD_STANDARD_FILE
    )

    return wkf


MPControl.set_multiprocess_engine("local")

print("Loading data")
inferelator_adata = ad.read(INFERELATOR_DATA_FILE)

print(f"Saving expression {EXPRESSION_TSV_FILES['expression']}")
inferelator_adata.to_df().to_csv(
    EXPRESSION_TSV_FILES['expression'],
    sep="\t",
    compression={'method': 'gzip', 'compresslevel': 1}
)

print(f"Saving denoised {EXPRESSION_TSV_FILES['denoised']}")
inferelator_adata.to_df(layer='denoised').to_csv(
    EXPRESSION_TSV_FILES['denoised'],
    sep="\t",
    compression={'method': 'gzip', 'compresslevel': 1}
)

print(f"Saving velocity {EXPRESSION_TSV_FILES['velocity']}")
inferelator_adata.to_df(layer='velocity').to_csv(
    EXPRESSION_TSV_FILES['velocity'],
    sep="\t",
    compression={'method': 'gzip', 'compresslevel': 1}
)

print(f"Saving metadata {METADATA_DATA_FILES['metadata']}")
inferelator_adata.obs.to_csv(
    METADATA_DATA_FILES['metadata'],
    sep="\t",
    compression={'method': 'gzip', 'compresslevel': 1}
)

print(f"Saving metadata {METADATA_DATA_FILES['gene_metadata']}")
inferelator_adata.var.to_csv(
    METADATA_DATA_FILES['gene_metadata'],
    sep="\t",
    compression={'method': 'gzip', 'compresslevel': 1}
)

del inferelator_adata
gc.collect()

worker = set_up_workflow(
    inferelator_workflow(regression=REGRESSION, workflow="single-cell")
)
worker.set_expression_file(h5ad=INFERELATOR_DATA_FILE)
worker.set_count_minimum(0.05)
worker.add_preprocess_step(normalize_expression_to_median)

worker.set_tfa(
    tfa_output_file=LATENT_DATA_FILES['tfa_expression']
)

worker.startup()

del worker


gc.collect()

worker = set_up_workflow(
    inferelator_workflow(regression=REGRESSION, workflow="single-cell")
)
worker.set_expression_file(h5ad=INFERELATOR_DATA_FILE, h5_layer='denoised')
worker.set_tfa(
    tfa_output_file=LATENT_DATA_FILES['tfa_denoised']
)

worker.startup()

del worker

gc.collect()

worker = set_up_workflow(
    inferelator_workflow(regression=REGRESSION, workflow=VelocityWorkflow)
)
worker.set_expression_file(h5ad=INFERELATOR_DATA_FILE, h5_layer='denoised')
worker.set_velocity_parameters(
    velocity_file_name=INFERELATOR_DATA_FILE,
    velocity_file_type="h5ad",
    velocity_file_layer='velocity'
)
worker.set_tfa(
    tfa_output_file=LATENT_DATA_FILES['tfa_velocity']
)

worker.startup()

del worker

gc.collect()

worker = set_up_workflow(
    inferelator_workflow(regression=REGRESSION, workflow=VelocityWorkflow)
)
worker.set_expression_file(h5ad=INFERELATOR_DATA_FILE, h5_layer='denoised')
worker.set_velocity_parameters(
    velocity_file_name=INFERELATOR_DATA_FILE,
    velocity_file_type="h5ad",
    velocity_file_layer='velocity'
)
worker.set_decay_parameters(
    global_decay_constant=.0150515
)
worker.set_tfa(
    tfa_output_file=LATENT_DATA_FILES['tfa_decay_constant']
)

worker.startup()

del worker

gc.collect()

worker = set_up_workflow(
    inferelator_workflow(regression=REGRESSION, workflow=VelocityWorkflow)
)
worker.set_expression_file(h5ad=INFERELATOR_DATA_FILE, h5_layer='denoised')
worker.set_velocity_parameters(
    velocity_file_name=INFERELATOR_DATA_FILE,
    velocity_file_type="h5ad",
    velocity_file_layer='velocity'
)
worker.set_decay_parameters(
    decay_constant_file=INFERELATOR_DATA_FILE,
    decay_constant_file_type="h5ad",
    decay_constant_file_layer='decay_constants'
)
worker.set_tfa(
    tfa_output_file=LATENT_DATA_FILES['tfa_decay_latent']
)

worker.startup()

del worker
