require(monocle3)
require(tidyr)
require(dplyr)
require(readr)

# File paths
WORKING.PATH <<- "/scratch/cj59/"
DATA.PATH <<- 'RAPA'
COUNTS.FILE <<- "2021_RAPA_TIMECOURSE.tsv.gz"
META.DATA.COLUMNS <<- c('Gene', 'Replicate', 'Pool', 'Experiment')

print("Loading count data")
count_data <- readr::read_delim(file.path(WORKING.PATH, DATA.PATH, COUNTS.FILE), "\t", escape_double = FALSE, trim_ws = TRUE)
count_data[, '...1'] <- NULL

print("Extracting metadata")
meta_data <- as.data.frame(dplyr::select(count_data, all_of(META.DATA.COLUMNS)))
rownames(meta_data) <- rownames(count_data)

count_data <- dplyr::select(count_data, -any_of(META.DATA.COLUMNS))
meta_data['UMI'] <- rowSums(count_data)

OUT.MONOCLE.FILE <- '/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_MONOCLE_PCNN_PT.tsv'

N_PCS <- seq(5, 105, 10)
N_NEIGHBORS <- seq(15, 105, 10)

# Get the experiment and gene tags, and the associated root cell
# Which is precomputed from the PC1 pseudotime
# {1: 1, WT, 9213; 2: 1, fpr1, 3814; etc...}
unpack.int <- function(x) {
  if (x %% 2 == 0) {gene <- "fpr1"} else {gene <- "WT"}
  e <- floor((x - 1) / 2) + 1
  rc <- c('9214', '3815', '82689', '70873')[x] # 1-index argmin(PC1)
  return(list(e, gene, rc))
}

# Run monocle on a subset of the data and return the object
do.monocle3 <- function(x, n.neighbors=15, n_pcs=100, umap.comps=2) {
  metas <- unpack.int(x)
  keepers <- (meta_data$Gene == metas[[2]]) & (meta_data$Experiment == metas[[1]])

  keep.counts <- t(count_data[keepers, ])
  colnames(keep.counts) <- rownames(count_data)[keepers]
  print(paste0("Rooting on cell ", toString(metas[[3]])))

  print("Select & Make CDS")
  cds <- new_cell_data_set(keep.counts, cell_metadata = meta_data[keepers, ])
  print("Preprocess CDS")
  cds <- preprocess_cds(cds, num_dim = n_pcs)
  print("DimRed CDS")
  cds <- reduce_dimension(cds, umap.n_neighbors = n.neighbors, max_components=umap.comps)
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

# Make column labels for the grid search
print(paste0("Preparing for ", toString(length(N_NEIGHBORS) * length(N_PCS)), " element grid search"))
column.labels <- apply(expand.grid(N_PCS, N_NEIGHBORS), 1, paste, collapse="_")
monocle3_pseudotime <- data.frame(matrix(0, ncol=length(column.labels), nrow=nrow(count_data)),
                                  row.names = rownames(count_data))
colnames(monocle3_pseudotime) <- column.labels

# Do a grid search for neighbors and PCs
for (n.neigh in N_NEIGHBORS) {
  for (n.pcs in N_PCS) {
    column.label <- paste(n.pcs, n.neigh, sep="_")
    print(paste("Neighbors:", n.neigh, "PCs:", n.pcs, "Column:", column.label))
    expts <- lapply(1:4, do.monocle3, n.neighbors = n.neigh, n_pcs = n.pcs)
    
    for (i in 1:4) {
      monocle3_pseudotime[, column.label] <- assign.metadata.pt(i, 
                                                                expts[[i]]@principal_graph_aux[["UMAP"]]$pseudotime,
                                                                monocle3_pseudotime[, column.label])
    }
    unconnected <- sum(is.infinite(monocle3_pseudotime[, column.label]))
    print(paste(unconnected, "unconnected (infinite) cells"))
  }
  write_tsv(x = monocle3_pseudotime, file = OUT.MONOCLE.FILE)
}
