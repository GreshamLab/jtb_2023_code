# jtb_2023_code

This repository contains figure generation code for the manuscript "Simultaneous estimation of gene regulatory network structure and RNA kinetics from single cell gene expression".

Model code is in the [supirfactor-dynamical](https://github.com/GreshamLab/supirfactor-dynamical) package.
Velocity and time inference code is in the [inferelator-velocity](https://github.com/flatironinstitute/inferelator-velocity) package.
Sequencing data is deposited in NCBI GEO under accession [GSE242556](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242556).

![Figure 1](https://github.com/GreshamLab/jtb_2023_code/blob/main/Figures/Figure_1.png?raw=true)

## Overview

This project is the analysis and plotting layer for the manuscript. It packages processed data, computes shared derived quantities such as projections and decay summaries, loads model outputs from external packages, and renders the main and supplemental figures.

The repository is organized around a small Python package, `jtb_2023_code`, plus checked-in figure assets and helper scripts.

## Repository layout

- `/home/runner/work/jtb_2023_code/jtb_2023_code/jtb_2023_code`: main Python package
  - `generate_figures.py`: top-level figure generation pipeline
  - `preprocess.py`: builds the processed shared dataset used by the figures
  - `package_data.py`: exports a packaged `AnnData` object with counts, embeddings, velocities, and decay values
  - `figure_1.py` through `figure_6.py`: main figure builders
  - `figure_*_supplemental.py`: supplemental figure builders
  - `figure_constants.py`: file paths, plotting constants, selected genes, and other central configuration
  - `utils/`: shared data loading, plotting, projections, decay, pseudotime, and model helper code
  - `pseudotime/`: wrappers for different pseudotime methods
  - `old_data/`: support for older published datasets used in later figures
- `/home/runner/work/jtb_2023_code/jtb_2023_code/Data`: reference tables shipped with the repository
- `/home/runner/work/jtb_2023_code/jtb_2023_code/Figures`: rendered main and supplemental figure outputs
- `/home/runner/work/jtb_2023_code/jtb_2023_code/Schematic`: static schematic artwork used inside the figures
- `/home/runner/work/jtb_2023_code/jtb_2023_code/scripts`: standalone runners for pseudotime workflows

## Main workflow

At a high level, the figure pipeline does the following:

1. Loads single-cell expression data into a shared `FigureSingleCellData` object.
2. Computes or loads PCA, UMAP, pseudotime, decay, and related derived data.
3. Loads model outputs and predictions from external modeling packages.
4. Builds the manuscript figures and writes them to the figure output directory.

The main orchestrator is `jtb_2023_code/generate_figures.py`, which preprocesses the dataset and then calls each figure module in sequence.

## Installation

The package metadata is defined in `/home/runner/work/jtb_2023_code/jtb_2023_code/setup.py`.

Primary Python dependencies include:

- `numpy`
- `scipy`
- `scanpy`
- `pandas`
- `joblib`
- `anndata`
- `matplotlib`
- `pydeseq2`
- `supirfactor-dynamical`
- `inferelator-velocity`

Install the repository in an environment that already has the scientific Python stack available:

```bash
pip install -e /home/runner/work/jtb_2023_code/jtb_2023_code
```

## Data and path configuration

The code expects several data, figure, scratch, schematic, and model paths. These defaults are defined in `jtb_2023_code/figure_constants.py` and `jtb_2023_code/utils/figure_filenames.py`.

Some defaults point to author-specific local or scratch locations, so running the pipeline in a new environment usually requires overriding paths on the command line. The path helper supports:

- `-d` for the data directory
- `-f` for the figure directory
- `-scratch` for scratch files
- `-s` for the schematic directory
- `-m` for the model directory

## Usage

Examples of the main entry points:

### Preprocess the shared figure dataset

```bash
python /home/runner/work/jtb_2023_code/jtb_2023_code/jtb_2023_code/preprocess.py \
  -d /path/to/data \
  -scratch /path/to/scratch
```

### Generate all figures

```bash
python /home/runner/work/jtb_2023_code/jtb_2023_code/jtb_2023_code/generate_figures.py \
  -d /path/to/data \
  -f /path/to/figures \
  -scratch /path/to/scratch \
  -s /path/to/schematics \
  -m /path/to/models
```

### Package data for downstream use

```bash
python /home/runner/work/jtb_2023_code/jtb_2023_code/jtb_2023_code/package_data.py \
  -d /path/to/data \
  -scratch /path/to/scratch \
  --output_file /path/to/output/data.h5ad
```

## Notes

- This repository focuses on manuscript analysis and figure production rather than the core dynamical model implementation.
- The checked-in `Figures/` directory provides examples of the expected outputs.
- The `scripts/` directory contains method-specific pseudotime runners and may require additional environment setup depending on the tool being used.

## License

This repository is distributed under the MIT License. See `/home/runner/work/jtb_2023_code/jtb_2023_code/LICENSE`.
