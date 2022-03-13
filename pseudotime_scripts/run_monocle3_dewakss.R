require(monocle3)
require(tidyr)
require(dplyr)
require(readr)

# File paths
WORKING.PATH <<- "/scratch/cj59/"
DATA.PATH <<- 'RAPA'
COUNTS.FILE <<- "2021_RAPA_TIMECOURSE_DENOISED_EXPRESSION.tsv"
META.DATA.COLUMNS <<- c('Gene', 'Replicate', 'Pool', 'Experiment')

print("Loading count data")
count_data <- readr::read_delim(file.path(WORKING.PATH, DATA.PATH, COUNTS.FILE), "\t", escape_double = FALSE, trim_ws = TRUE)
count_data[, '...1'] <- NULL

print("Extracting metadata")
meta_data <- as.data.frame(dplyr::select(count_data, all_of(META.DATA.COLUMNS)))
rownames(meta_data) <- rownames(count_data)

count_data <- dplyr::select(count_data, -any_of(META.DATA.COLUMNS))

OUT.MONOCLE.FILE <- '/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_MONOCLE_DEWAKSS.tsv'

# Get the experiment and gene tags, and the associated root cell
# Which is precomputed from the PC1 pseudotime
# {1: 1, WT, 9213; 2: 1, fpr1, 3814; etc...}
unpack.int <- function(x) {
  if (x %% 2 == 0) {gene <- "fpr1"} else {gene <- "WT"}
  e <- floor((x - 1) / 2) + 1
  rc <- c('9214', '3815', '82689', '70873')[x] # 1-index argmin(PC1)
  opt_nn <- c(75, 75, 85, 75)[x]
  opt_pcs <- c(85, 45, 105, 45)[x]
  return(list(e, gene, rc, opt_nn, opt_pcs))
}

# Run monocle on a subset of the data and return the object
do.monocle3 <- function(x) {
  metas <- unpack.int(x)
  n_pcs <- metas[[5]]
  n.neighbors <- metas[[4]]
  keepers <- (meta_data$Gene == metas[[2]]) & (meta_data$Experiment == metas[[1]])

  keep.counts <- t(count_data[keepers, ])
  colnames(keep.counts) <- rownames(count_data)[keepers]
  print(paste0("Rooting on cell ", toString(metas[[3]])))

  print("Select & Make CDS")
  cds <- new_cell_data_set(keep.counts, cell_metadata = meta_data[keepers, ])
  print("Preprocess CDS")
  cds <- preprocess_cds(cds, num_dim = n_pcs)
  print("DimRed CDS")
  cds <- reduce_dimension(cds, umap.n_neighbors = n.neighbors, max_components=2)
  print("Cluster CDS")
  cds <- cluster_cells(cds, k=n.neighbors)
  print("Learn Graph")
  cds <- learn_graph(cds)
  print("Order Graph")
  cds <- order_cells(cds, root_cells=metas[[3]])
  return(cds)
}

# Put pseudotime into the appropriate rows
assign.metadata.pt <- function(x, pt, pt.metadata) {
    metas <- unpack.int(x)
    print(paste0("Processing PT for ", toString(metas[[1]]), "_", toString(metas[[2]])))

    pt.metadata[(meta_data$Gene == metas[[2]]) & (meta_data$Experiment == metas[[1]])] <- pt 
    return(pt.metadata)
}

monocle3_pseudotime <- data.frame(matrix(0, ncol=1, nrow=nrow(count_data)),
                                  row.names = rownames(count_data))
colnames(monocle3_pseudotime) <- c('dewakss')

expts <- lapply(1:4, do.monocle3)

for (i in 1:4) {
  monocle3_pseudotime[, 'dewakss'] <- assign.metadata.pt(i, 
                                                         expts[[i]]@principal_graph_aux[["UMAP"]]$pseudotime,
                                                         monocle3_pseudotime[, 'dewakss'])
}

write_tsv(x = monocle3_pseudotime, file = OUT.MONOCLE.FILE)
