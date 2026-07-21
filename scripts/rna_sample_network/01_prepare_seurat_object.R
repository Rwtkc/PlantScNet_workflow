#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) stop("Missing value for ", flag)
  args[[hit + 1]]
}

matrix_path <- get_arg("--matrix")
features_path <- get_arg("--features")
barcodes_path <- get_arg("--barcodes")
output_path <- get_arg("--output", "seurat_obj.rds")
project_name <- get_arg("--project", "PlantScNet_sample")
gene_column <- as.integer(get_arg("--gene-column", "1"))
min_cells <- as.integer(get_arg("--min-cells", "3"))
min_features <- as.integer(get_arg("--min-features", "200"))
resolution <- as.numeric(get_arg("--resolution", "0.5"))
mt_pattern <- get_arg("--mt-pattern", "^MT-")

if (is.null(matrix_path) || is.null(features_path) || is.null(barcodes_path)) {
  stop("Required arguments: --matrix, --features, --barcodes")
}

message("Reading matrix: ", matrix_path)
mat <- readMM(matrix_path)
features <- read.delim(features_path, sep = "\t", header = FALSE, stringsAsFactors = FALSE, fill = TRUE)
barcodes <- read.table(barcodes_path, stringsAsFactors = FALSE)

if (gene_column > ncol(features)) {
  stop("--gene-column exceeds number of columns in feature table")
}

rownames(mat) <- make.unique(as.character(features[[gene_column]]))
colnames(mat) <- as.character(barcodes[[1]])

mat <- mat[rowSums(mat != 0) > 0, , drop = FALSE]
mat <- mat[, colSums(mat != 0) > 0, drop = FALSE]

obj <- CreateSeuratObject(
  counts = mat,
  project = project_name,
  min.cells = min_cells,
  min.features = min_features
)

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = mt_pattern)
obj@meta.data$nGene <- obj@meta.data$nFeature_RNA
obj@meta.data$nUMI <- obj@meta.data$nCount_RNA

obj <- SCTransform(obj, verbose = FALSE)
obj <- RunPCA(obj, verbose = FALSE)

std_dev <- obj[["pca"]]@stdev
first_diff <- diff(std_dev)
second_diff <- diff(first_diff)
elbow_point <- which.min(second_diff) + 2
elbow_point <- max(2, min(elbow_point, length(std_dev)))

obj <- FindNeighbors(obj, dims = 1:elbow_point, verbose = FALSE)
obj <- FindClusters(obj, resolution = resolution, verbose = FALSE)

saveRDS(obj, output_path)
message("Saved: ", output_path)
message("PCA dimensions used: 1:", elbow_point)
