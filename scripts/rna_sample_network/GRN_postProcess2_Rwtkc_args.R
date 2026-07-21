suppressPackageStartupMessages(library("data.table"))
suppressPackageStartupMessages(library("pbapply"))
suppressPackageStartupMessages(library("philentropy"))
suppressPackageStartupMessages(library("dplyr"))
suppressPackageStartupMessages(library("stringr"))
suppressPackageStartupMessages(library("magrittr"))


parse_args <- function(args) {
  opts <- list(
    scenicOutput = "./result/",
    outputDir = "./result/",
    group = "seurat_clusters"
  )

  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]

    if (arg %in% c("--scenicOutput", "--scenic_output")) {
      opts$scenicOutput <- args[[i + 1]]
      i <- i + 2
    } else if (arg %in% c("-o", "--output", "--outputDir", "--output_dir")) {
      opts$outputDir <- args[[i + 1]]
      i <- i + 2
    } else if (arg %in% c("--group", "-g")) {
      opts$group <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }

  opts
}


require_file <- function(path) {
  if (!file.exists(path) || file.info(path)$size <= 0) {
    stop("Required file is missing or empty: ", path, call. = FALSE)
  }
}


args <- parse_args(commandArgs(trailingOnly = TRUE))
scenicOutput <- args$scenicOutput
outputDir <- args$outputDir
group <- args$group

dir.create(outputDir, recursive = TRUE, showWarnings = FALSE)

message("scenicOutput: ", normalizePath(scenicOutput, mustWork = FALSE))
message("outputDir: ", normalizePath(outputDir, mustWork = FALSE))
message("group: ", group)

require_file(file.path(scenicOutput, "cellmeta.rds"))
require_file(file.path(scenicOutput, "AUCell.txt"))
require_file(file.path(scenicOutput, "reg.tsv"))

cellmeta <- readRDS(file = file.path(scenicOutput, "cellmeta.rds"))

if (!(group %in% colnames(cellmeta))) {
  stop("Group column not found in cellmeta: ", group, call. = FALSE)
}

message("Read cellmeta: ", nrow(cellmeta), " cells")

## Regulon activity score (RAS) matrix
rasMat <- data.table::fread(
  file.path(scenicOutput, "AUCell.txt"),
  sep = "\t",
  header = TRUE,
  data.table = FALSE,
  check.names = FALSE
)

rownames(rasMat) <- rasMat[[1]]
colnames(rasMat) <- sub("(+)", "", colnames(rasMat), fixed = TRUE)
rasMat <- as.matrix(rasMat[, -1, drop = FALSE])
storage.mode(rasMat) <- "numeric"

common_cells <- intersect(rownames(cellmeta), rownames(rasMat))
if (length(common_cells) == 0) {
  stop("No overlapping cell names between cellmeta.rds and AUCell.txt", call. = FALSE)
}

if (length(common_cells) < nrow(cellmeta)) {
  message(
    "Warning: only ", length(common_cells), " / ", nrow(cellmeta),
    " cellmeta cells are present in AUCell.txt"
  )
}

cellmeta <- cellmeta[common_cells, , drop = FALSE]
rasMat <- rasMat[common_cells, , drop = FALSE]
saveRDS(rasMat, file.path(outputDir, "rasMat.rds"))
message("20%")

## Regulon Specificity Score (RSS) matrix
cell.types <- unique(as.character(cellmeta[[group]]))
ctMat <- sapply(cell.types, function(i) {
  as.numeric(as.character(cellmeta[[group]]) == i)
})

if (is.null(dim(ctMat))) {
  ctMat <- matrix(ctMat, ncol = 1)
}

colnames(ctMat) <- cell.types
rownames(ctMat) <- rownames(cellmeta)
message("40%")

calc_rss_one <- function(reg_name) {
  x <- as.numeric(rasMat[, reg_name])
  sapply(colnames(ctMat), function(ct) {
    y <- as.numeric(ctMat[, ct])
    if (all(is.na(x)) || sum(x, na.rm = TRUE) <= 0 || sum(y, na.rm = TRUE) <= 0) {
      return(NA_real_)
    }
    suppressMessages(
      1 - philentropy::JSD(
        rbind(x, y),
        unit = "log2",
        est.prob = "empirical"
      )
    )
  })
}

rss_list <- pbapply::pblapply(colnames(rasMat), calc_rss_one)
rssMat <- do.call(rbind, rss_list)
rssMat <- as.matrix(rssMat)
rownames(rssMat) <- colnames(rasMat)
colnames(rssMat) <- colnames(ctMat)
saveRDS(rssMat, file.path(outputDir, "rssMat.rds"))
message("60%")

# TF and their targets in each regulon
reg <- read.table(
  file.path(scenicOutput, "reg.tsv"),
  sep = "\t",
  stringsAsFactors = FALSE,
  quote = "",
  comment.char = ""
)

tf_target <- lapply(4:nrow(reg), function(i) {
  x <- unname(unlist(reg[i, , drop = TRUE]))
  tt <- stringr::str_extract_all(string = x[9], pattern = "\\(.+?\\)", simplify = TRUE) %>%
    stringr::str_replace_all(pattern = "\\(|\\)|'", replacement = "") %>%
    stringr::str_split(pattern = ", ", simplify = TRUE) %>%
    magrittr::set_colnames(c("target", "value"))

  data.frame(TF = x[1], tt, stringsAsFactors = FALSE) %>%
    dplyr::mutate(value = as.numeric(value))
})

message("80%")

tf_target <- do.call("rbind", tf_target) %>% unique()
colnames(tf_target)[3] <- "importance_score"
saveRDS(tf_target, file.path(outputDir, "tf_target.rds"))
write.table(
  tf_target,
  file = file.path(outputDir, "tf_target.txt"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

message("100%")
