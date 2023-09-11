import numpy as _np
import os as _os         
from .utils.figure_filenames import *

DataFile.set_path("~/Documents/jtb_2022_code/Data/")
FigureFile.set_path("~/Documents/jtb_2022_code/Figures/")
ScratchFile.set_path("/scratch/cj59/RAPA/")

MAIN_FIGURE_DPI = 300
SUPPLEMENTAL_FIGURE_DPI = 300

# Print status updates
VERBOSE = True

# Metadata for bulk expression data
RAPA_BULK_EXPR_FILE = str(DataFile("20210312_RAPA_BULK_TIMECOURSE.tsv.gz"))
RAPA_BULK_EXPR_FILE_META_DATA_COLS = ["Oligo", "Time", "Replicate"]
RAPA_BULK_EXPR_FILE_TIMES = [2.5, 5, 7.5, 10, 15, 30, 45, 60, 90, 120]

# Single cell expression data filenames
RAPA_SINGLE_CELL_EXPR_FILE = str(ScratchFile("2021_RAPA_TIMECOURSE.h5ad"))
RAPA_SINGLE_CELL_EXPR_PROCESSED = str(ScratchFile("2021_RAPA_TIMECOURSE_FIGS.h5ad"))
RAPA_SINGLE_CELL_VELOCITY = str(ScratchFile("2021_RAPA_VELOCITY_FIGS.h5ad"))
RAPA_SINGLE_CELL_DENOISED = str(ScratchFile("2021_RAPA_DENOISED_FIGS.h5ad"))

ELIFE_SINGLE_CELL_FILE = str(ScratchFile("ELIFE_2020_SINGLE_CELL.h5ad"))
ELIFE_SINGLE_CELL_FILE_PROCESSED = str(ScratchFile("2020_ELIFE_STATIC.h5ad"))

# For formatting (needs {e} and {g})
RAPA_SINGLE_CELL_EXPR_BY_EXPT = str(ScratchFile("2021_RAPA_TIMECOURSE_FIGS_{e}_{g}.h5ad"))
RAPA_SINGLE_CELL_VELOCITY_BY_EXPT = str(ScratchFile("2021_RAPA_VELOCITY_FIGS_{e}_{g}.h5ad"))
RAPA_SINGLE_CELL_DENOISED_BY_EXPT = str(ScratchFile("2021_RAPA_DENOISED_FIGS_{e}_{g}.h5ad"))

# Inferelator files
INFERELATOR_DATA_FILE = str(ScratchFile("2021_INFERELATOR_DATA.h5ad"))
INFERELATOR_PRIORS_FILE = str(DataFile("JOINT_PRIOR_20230701.tsv.gz"))
INFERELATOR_TF_NAMES_FILE = str(ScratchFile("tf_names_yeastract.txt"))

# Model files
SUPIRFACTOR_COUNT_MODEL = str(ModelFile("RAPA_COUNT_RNN_MODEL.h5"))
SUPIRFACTOR_VELOCITY_DYNAMICAL_MODEL = str(ModelFile("RAPA_VELOCITY_RNN_MODEL.h5"))
SUPIRFACTOR_DECAY_MODEL = str(ModelFile("RAPA_DECAY_MODEL.h5"))
SUPIRFACTOR_BIOPHYSICAL_MODEL = str(ModelFile("RAPA_BIOPHYSICAL_MODEL.h5"))

MODEL_RESULTS_FILE = str(DataFile("SUPIRFACTOR_RESULTS_ALL.tsv.gz"))
MODEL_LOSSES_FILE = str(DataFile("SUPIRFACTOR_LOSSES_ALL.tsv.gz"))

# Pseudotime TSV files keyed by (method, is_dewakss), value (file name, has_index)
PSEUDOTIME_FILES = {
    ('dpt', False): (str(DataFile("2021_RAPA_TIMECOURSE_DPT.tsv.gz")), True),
    ('cellrank', False): (str(DataFile("2021_RAPA_TIMECOURSE_CELLRANK.tsv.gz")), True),
    ('monocle', False): (str(DataFile("2021_RAPA_TIMECOURSE_MONOCLE.tsv.gz")), False),
    ('palantir', False): (str(DataFile("2021_RAPA_TIMECOURSE_PALANTIR.tsv.gz")), True),
    ('dpt', True): (str(DataFile("2021_RAPA_TIMECOURSE_DPT_DEWAKSS.tsv.gz")), True),
    ('cellrank', True): (str(DataFile("2021_RAPA_TIMECOURSE_CELLRANK_DEWAKSS.tsv.gz")), True),
    ('monocle', True): (str(DataFile("2021_RAPA_TIMECOURSE_MONOCLE_DEWAKSS.tsv.gz")), False),
    ('palantir', True): (str(DataFile("2021_RAPA_TIMECOURSE_PALANTIR_DEWAKSS.tsv.gz")), True)
}

# Existing decay constant data files
# {DataSet: (File type, Gene Column, Half-life Column, Excel loading engine)}
DECAY_CONSTANT_FILES = {
    'Neymotin2014': ('tsv', "Syst", "thalf", None),
    'Chan2018': ('tsv', "gene_id", ["halflife_160412_r1", "halflife_160412_r2"], None),
    'Geisberg2014': ('tsv', "systematic_name", "halflife", None),
    'Munchel2011': ('tsv', "Systematic Name", "Half-life [min]", None),
    'Miller2011': ('tsv', "X1", "wt", None)
}
                   
DECAY_CONSTANT_LINKS = {
    'Neymotin2014': "https://rnajournal.cshlp.org/content/suppl/2014/08/08/rna.045104.114.DC1/TableS5.xls",
    'Chan2018': "https://cdn.elifesciences.org/articles/32536/elife-32536-fig1-data2-v4.txt",
    'Geisberg2014': "https://www.cell.com/cms/10.1016/j.cell.2013.12.026/attachment/5d358c57-4ca0-4216-be37-3cc5c909b375/mmc1.xlsx",
    'Munchel2011': "https://www.molbiolcell.org/doi/suppl/10.1091/mbc.e11-01-0028/suppl_file/mc-e11-01-0028-s10.xls",
    'Miller2011': "https://www.embopress.org/action/downloadSupplement?doi=10.1038%2Fmsb.2010.112&file=msb2010112-sup-0001.txt"
}
                        
# Gene metadata filenames
GENE_GROUP_FILE = str(DataFile("STable6.tsv"))
GENE_NAMES_FILE = str(DataFile("yeast_gene_names.tsv"))

# Group columns
CC_COLS = ['M-G1', 'G1', 'S', 'G2', 'M']
AGG_COLS = ['RP', 'RiBi', 'iESR', 'Mito']
OTHER_GROUP_COL = 'Other'
CELLCYCLE_GROUP_COL = 'Cell Cycle'
GENE_CAT_COLS = ['RP', 'RiBi', 'iESR', CELLCYCLE_GROUP_COL, OTHER_GROUP_COL]

# ADATA keys
RAPA_TIME_COL = 'program_rapa_time'
CC_TIME_COL = 'program_cc_time'
RAPA_GRAPH_OBSP = 'program_rapa_distances'
CC_GRAPH_OBSP = 'program_cc_distances'
RAPA_VELO_LAYER = 'rapamycin_velocity'
CC_VELO_LAYER = 'cell_cycle_velocity'

# Umap parameters
UMAP_NPCS = 50
UMAP_NNS = 200
UMAP_MIN_DIST = 0.2

# Input schematic FIGS
FIG1B_FILE_NAME = str(SchematicFile("Figure1B_RAW.png"))
FIG2A_FILE_NAME = str(SchematicFile("Figure2A_RAW.png"))
FIG3A_FILE_NAME = str(SchematicFile("Figure3A_RAW.png"))
SFIG2A_FILE_NAME = str(SchematicFile("SupplementalFigure2A_RAW.png"))
SFIG3A_FILE_NAME = str(SchematicFile("SupplementalFigure3A_RAW.png"))
SFIG3B_FILE_NAME = str(SchematicFile("SupplementalFigure3B_RAW.png"))
FIG_RAPA_LEGEND_FILE_NAME = str(SchematicFile("Figure_RAPA_Legend.png"))
FIG_CC_LEGEND_FILE_NAME = str(SchematicFile("Figure_CC_Legend.png"))
FIG_RAPA_LEGEND_VERTICAL_FILE_NAME = str(SchematicFile("Figure_RAPA_Legend_Vertical.png"))
FIG_CC_LEGEND_VERTICAL_FILE_NAME = str(SchematicFile("Figure_CC_Legend_Vertical.png"))
FIG_EXPT_LEGEND_VERTICAL_FILE_NAME = str(SchematicFile("Figure_EXPT_Legend_Vertical.png"))
FIG_DEEP_LEARNING_FILE_NAME = str(SchematicFile("Deep_Learning_Model.png"))
FIG_DYNAMICAL_FILE_NAME = str(SchematicFile("Dynamical_Model.png"))

# Color Palettes for Categorical Data
POOL_PALETTE = "YlGnBu"
EXPT_PALETTE = "Dark2"
GENE_PALETTE = "Dark2"
CC_PALETTE = "Set2"
GENE_CAT_PALETTE = "Set1"

CATEGORY_COLORS = ["gray", "skyblue", "lightgreen"]
CLUSTER_PALETTE = 'tab20'
PROGRAM_PALETTE = 'tab10'

# Output file names (without extensions)
FIGURE_1_FILE_NAME = str(FigureFile("Figure_1"))
FIGURE_1_1_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_1_1"))
FIGURE_1_2_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_1_2"))
FIGURE_2_FILE_NAME = str(FigureFile("Figure_2"))
FIGURE_2_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_2"))
FIGURE_3_FILE_NAME = str(FigureFile("Figure_3"))
FIGURE_3_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_3"))
FIGURE_4_FILE_NAME = str(FigureFile("Figure_4"))
FIGURE_4_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_4"))
FIGURE_5_FILE_NAME = str(FigureFile("Figure_5"))
FIGURE_5_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_5"))
FIGURE_6_FILE_NAME = str(FigureFile("Figure_6"))
FIGURE_6_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_6"))

# Search space for grid searches
N_PCS = _np.arange(5, 115, 10)
N_NEIGHBORS = _np.arange(20, 190, 10)
N_COMPS = _np.array([5, 10, 15])

## FIGURE CONSTANTS ##
FIGURE_1A_MINMAX = 5
FIGURE_1A_LFC_THRESHOLD = _np.log2(1.25)
FIGURE_1A_PADJ_THRESHOLD = 0.01

### TIME CONSTANTS ###
CC_LENGTH_DATA_FILE = str(DataFile("Supplemental_Growth_Curve_FY45.tsv"))
CC_LENGTH = 88

### SELECT GENES FOR FIGURES ###
FIGURE_4_GENES = ["YKR039W", "YOR063W"]
FIGURE_5_GENES = ["YGR109C", "YNR009W", "YIL131C", "YPR119W"]

# FROM SPELLMAN98 #
# ADJUSTED TO 88 MIN #

CC_TIME_ORDER = {
    'M-G1': ('G1', 7, 22.5),
    'G1': ('S', 22.5, 39.5),
    'S': ('G2', 39.5, 56.5),
    'G2': ('M', 56.5, 77.5), 
    'M': ('M-G1', 77.5, 95)
}


RAPA_TIME_ORDER = {
    '12': ('3', -5, 5),
    '3': ('4', 5, 15),
    '4': ('5', 15, 25),
    '5': ('6', 25, 35), 
    '6': ('7', 35, 45),
    '7': ('8', 45, 55)
}

METRIC_SCALE_LIMS = {
    'cosine': [0, 2],
    'euclidean': [0, None],
    'manhattan': [0, None],
    'information': [0, 1]
}