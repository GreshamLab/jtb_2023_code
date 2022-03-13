import numpy as _np
import os as _os         
from .utils.figure_filenames import *

DataFile.set_path("~/Documents/jtb_2022_code/Data/")
FigureFile.set_path("~/Documents/jtb_2022_code/Figures/")
 
# Print status updates
VERBOSE = True

# Metadata for bulk expression data
RAPA_BULK_EXPR_FILE = str(DataFile("~/Documents/R/rapa_20210628/data/20210312_RAPA_BULK_TIMECOURSE.tsv.gz"))
RAPA_BULK_EXPR_FILE_META_DATA_COLS = ["Oligo", "Time", "Replicate"]
RAPA_BULK_EXPR_FILE_TIMES = [2.5, 5, 7.5, 10, 15, 30, 45, 60, 90, 120]

# Single cell expression data filenames
RAPA_SINGLE_CELL_EXPR_FILE = str(DataFile("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE.h5ad"))
RAPA_SINGLE_CELL_EXPR_PROCESSED = str(DataFile("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_FIGS.h5ad"))

# For formatting (needs {e} and {g})
RAPA_SINGLE_CELL_EXPR_BY_EXPT = str(DataFile("/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_FIGS_{e}_{g}.h5ad"))

# Pseudotime TSV files keyed by (method, is_dewakss), value (file name, has_index)
PSEUDOTIME_FILES = {('dpt', False): (str(DataFile("2021_RAPA_TIMECOURSE_DPT.tsv.gz")), True),
                    ('cellrank', False): (str(DataFile("2021_RAPA_TIMECOURSE_CELLRANK.tsv.gz")), True),
                    ('monocle', False): (str(DataFile("2021_RAPA_TIMECOURSE_MONOCLE.tsv.gz")), False),
                    ('palantir', False): (str(DataFile("2021_RAPA_TIMECOURSE_PALANTIR.tsv.gz")), True),
                    ('dpt', True): (str(DataFile("2021_RAPA_TIMECOURSE_DPT_DEWAKSS.tsv")), True),
                    ('cellrank', True): (str(DataFile("2021_RAPA_TIMECOURSE_CELLRANK_DEWAKSS.tsv")), True),
                    ('monocle', True): (str(DataFile("2021_RAPA_TIMECOURSE_MONOCLE_DEWAKSS.tsv.gz")), False)}

# Existing decay constant data files
# {DataSet: (File type, Gene Column, Half-life Column, Excel loading engine)}
DECAY_CONSTANT_FILES = {'Neymotin2014': ('tsv', "Syst", "thalf", None),
                        'Chan2018': ('tsv', "gene_id", ["halflife_160412_r1", "halflife_160412_r2"], None),
                        'Geisberg2015': ('excel', "systematic name", "Half-Life           (in minutes)", 'openpyxl'),
                        'Munchel2011': ('excel', "Systematic Name", "Half-life [min]", 'xlrd'),
                        'Miller2011': ('tsv', "X1", "wt", None)}
                        
DECAY_CONSTANT_LINKS = {'Neymotin2014': "https://rnajournal.cshlp.org/content/suppl/2014/08/08/rna.045104.114.DC1/TableS5.xls",
                        'Chan2018': "https://cdn.elifesciences.org/articles/32536/elife-32536-fig1-data2-v4.txt",
                        'Geisberg2015': "https://www.cell.com/cms/10.1016/j.cell.2013.12.026/attachment/5d358c57-4ca0-4216-be37-3cc5c909b375/mmc1.xlsx",
                        'Munchel2011': "https://www.molbiolcell.org/doi/suppl/10.1091/mbc.e11-01-0028/suppl_file/mc-e11-01-0028-s10.xls",
                        'Miller2011': "https://www.embopress.org/action/downloadSupplement?doi=10.1038%2Fmsb.2010.112&file=msb2010112-sup-0001.txt"}
                        
# Gene metadata filenames
GENE_GROUP_FILE = str(DataFile("STable6.tsv"))
GENE_NAMES_FILE = str(DataFile("yeast_gene_names.tsv"))

# Group columns
CC_COLS = ['M-G1', 'G1', 'S', 'G2', 'M']
AGG_COLS = ['RP', 'RiBi', 'iESR', 'Mito']
OTHER_GROUP_COL = 'Other'
CELLCYCLE_GROUP_COL = 'Cell Cycle'
GENE_CAT_COLS = ['RP', 'RiBi', 'iESR', CELLCYCLE_GROUP_COL, OTHER_GROUP_COL]

# Umap parameters
UMAP_NPCS = 50
UMAP_NNS = 200
UMAP_MIN_DIST = 0.2

# Input schematic FIG1B
FIG1B_FILE_NAME = str(DataFile("Figure1B_RAW.png"))

# Color Palettes for Categorical Data
POOL_PALETTE = "YlGnBu"
EXPT_PALETTE = "Dark2"
GENE_PALETTE = "Dark2"
CC_PALETTE = "Set2"
GENE_CAT_PALETTE = "Set1"

# Output file names (without extensions)
FIGURE_1_FILE_NAME = str(FigureFile("Figure_1"))
FIGURE_1_1_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_1_1"))
FIGURE_1_2_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_1_2"))
FIGURE_2_FILE_NAME = str(FigureFile("Figure_2"))
FIGURE_2_1_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_2_1"))
FIGURE_2_2_SUPPLEMENTAL_FILE_NAME = str(FigureFile("Supplemental_Figure_2_2"))
FIGURE_3_FILE_NAME = str(FigureFile("Figure_3"))

# Search space for grid searches
N_PCS = _np.arange(5, 115, 10)
N_NEIGHBORS = _np.arange(15, 115, 10)
N_COMPS = _np.array([5, 10, 15])

## FIGURE CONSTANTS ##
FIGURE_1A_MINMAX = 4
FIGURE_1A_LFC_THRESHOLD = _np.log2(1.25)
FIGURE_1A_PADJ_THRESHOLD = 0.01

### TIME CONSTANTS ###
POOL_TIMES = [(1, 0, 20),
              (2, 0, 20),
              (3, 20, 30),
              (4, 30, 40),
              (5, 40, 50),
              (6, 50, 60),
              (7, 60, 70),
              (8, 70, 80)]