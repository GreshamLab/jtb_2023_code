require(monocle3)
require(tidyr)
require(dplyr)
require(readr)

# File paths
WORKING.PATH <<- "."
DATA.PATH <<- 'Data'
META.DATA.COLUMNS <<- c('Gene', 'Replicate', 'Pool', 'Experiment')

load_count_tsv <- function(file.name, meta.data.columns=NULL) {
  c.data <- data.frame(readr::read_delim(file.path(WORKING.PATH, DATA.PATH, file.name), "\t", escape_double = FALSE, trim_ws = TRUE))
  c.data <- fix_imported_data_frame(c.data)
  if (!is.null(meta.data.columns)) {
    m.data <- c.data[meta.data.columns]
    c.data <- c.data[, !(colnames(c.data) %in% meta.data.columns)]
    m.data['UMI'] <- rowSums(c.data)
    return(list(c.data, m.data))
  }
  return(c.data)
}

fix_imported_data_frame <- function(broken.data.frame) {
  if ('X1' %in% colnames(broken.data.frame)) {broken.data.frame$X1 <- NULL}
  if ('---1' %in% colnames(broken.data.frame)) {broken.data.frame$X1 <- NULL}
  rownames(broken.data.frame) <- gsub(".", "-", rownames(broken.data.frame), fixed = TRUE)
  colnames(broken.data.frame) <- gsub(".", "-", colnames(broken.data.frame), fixed = TRUE)
  colnames(broken.data.frame)[colnames(broken.data.frame) == "X15S_rRNA"] <-  "15S_rRNA"
  colnames(broken.data.frame)[colnames(broken.data.frame) == "X21S_rRNA"] <-  "21S_rRNA"
  return(broken.data.frame)
}

if (!exists("count_data")) {
  count_data <<- load_count_tsv(COUNTS.FILE, meta.data.columns = META.DATA.COLUMNS)
  meta_data <<- count_data[[2]]
  count_data <<- count_data[[1]]
}

OUT.MONOCLE.FILE <- '/scratch/cj59/RAPA/2021_RAPA_TIMECOURSE_MONOCLE_PCNN_PT.tsv'

N_PCS <- seq(5, 105, 10)
N_NEIGHBORS <- seq(15, 105, 10)

pt.data <- read.table(file.path(WORKING.PATH, DATA.PATH, RAW.PT.FILE), sep="\t", header=T, stringsAsFactors = F)
['9213', '3814', '82688', '70872']

# Get the experiment and gene tags, and the associated root cell
# Which is precomputed from the PC1 pseudotime
unpack.int <- function(x) {
  if (x %% 2 == 0) {gene <- "fpr1"} else {gene <- "WT"}
  e <- floor((x - 1) / 2) + 1
  rc <- c('9213', '3814', '82688', '70872') # From PCA1 pseudotime
  return(list(e, gene, rc))
}

# Run monocle on a subset of the data and return the object
do.monocle3 <- function(x, n.neighbors=15, n_pcs=100, umap.comps=2) {
  metas <- unpack.int(x)
  keepers <- (meta_data$Gene == metas[[2]]) & (meta_data$Experiment == metas[[1]])

  keep.counts <- t(count_data[keepers, ])
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
    pt.metadata[(meta_data$Gene == metas[[2]]) & (meta_data$Experiment == metas[[1]])] <- pt 
    return(pt.metadata)
}

# Make column labels for the grid search
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
    
    for (i in 1:2) {
      monocle3_pseudotime[, column.label] <- assign.metadata.pt(i, 
                                                                expts[[i]]@principal_graph_aux[["UMAP"]]$pseudotime,
                                                                monocle3_pseudotime[, column.label])
    }
    unconnected <- sum(is.infinite(monocle3_pseudotime[, column.label]))
    print(paste(unconnected, "unconnected (infinite) cells"))
  }
  write_tsv(x = monocle3_pseudotime, file = OUT.MONOCLE.FILE)
}
