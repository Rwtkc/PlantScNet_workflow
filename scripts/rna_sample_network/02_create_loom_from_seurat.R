#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(SCopeLoomR)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) stop("Missing value for ", flag)
  args[[hit + 1]]
}

seurat_path <- get_arg("--seurat")
output_dir <- get_arg("--output-dir", "result")
group_col <- get_arg("--group", "seurat_clusters")
min_cells_per_gene <- as.integer(get_arg("--min-cells-per-gene", "3"))

if (is.null(seurat_path)) {
  stop("Required argument: --seurat")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(seurat_path)
mat <- GetAssayData(obj, slot = "data")
rownames(mat) <- gsub("-", "_", rownames(mat))

cellmeta <- obj@meta.data
cellmeta[[group_col]] <- as.character(cellmeta[[group_col]])
saveRDS(cellmeta, file.path(output_dir, "cellmeta.rds"))

mat_dense <- as.matrix(mat)
n_counts <- rowSums(mat_dense, na.rm = TRUE)
n_cells <- rowSums(mat_dense > 0, na.rm = TRUE)
genes_keep <- names(n_cells)[n_counts > 0 & n_cells > min_cells_per_gene]
mat_dense <- mat_dense[genes_keep, , drop = FALSE]

loom_path <- file.path(output_dir, "exprMat.loom")
loom <- build_loom(loom_path, dgem = mat_dense)

add_cell_annotation <- function(loom, cell_annotation) {
  cell_annotation <- data.frame(cell_annotation)
  for (drop_col in c("nGene", "nUMI")) {
    if (drop_col %in% colnames(cell_annotation)) {
      cell_annotation <- cell_annotation[, colnames(cell_annotation) != drop_col, drop = FALSE]
    }
  }
  if (ncol(cell_annotation) <= 0) stop("No cell annotation columns remain")
  if (!all(get_cell_ids(loom) %in% rownames(cell_annotation))) {
    stop("Cell IDs are missing in metadata")
  }
  cell_annotation <- cell_annotation[get_cell_ids(loom), , drop = FALSE]
  for (cn in colnames(cell_annotation)) {
    add_col_attr(loom = loom, key = cn, value = cell_annotation[, cn])
  }
  invisible(loom)
}

loom <- add_cell_annotation(loom, cellmeta)
close_loom(loom)

writeLines(rownames(mat_dense), file.path(output_dir, "loom_genes.txt"))
message("Saved: ", loom_path)
