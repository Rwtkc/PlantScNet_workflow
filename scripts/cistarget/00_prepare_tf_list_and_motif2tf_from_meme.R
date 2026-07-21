#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) stop("Missing value for ", flag)
  args[[hit + 1]]
}

meme_path <- get_arg("--meme")
out_tbl <- get_arg("--out-tbl", "motif2TF.tbl")
out_tf <- get_arg("--out-tf", "tf_list.txt")
out_motifs <- get_arg("--out-motifs", "motif_ids.txt")
source_name <- get_arg("--source-name", "PlantTFDB")
source_version <- get_arg("--source-version", "1.0")

if (is.null(meme_path) || !file.exists(meme_path) || file.info(meme_path)$size == 0) {
  stop("Required non-empty MEME file: --meme")
}

lines <- readLines(meme_path, warn = FALSE)
motif_lines <- grep("^MOTIF[[:space:]]+", lines, value = TRUE)
if (!length(motif_lines)) {
  stop("No MOTIF lines found in MEME file: ", meme_path)
}

split_lines <- strsplit(motif_lines, "[[:space:]]+")
gene_name <- vapply(split_lines, function(x) x[[2]], character(1))
motif_name <- vapply(split_lines, function(x) {
  if (length(x) >= 3) x[[3]] else x[[2]]
}, character(1))

df <- data.frame(
  motif_name = motif_name,
  gene_name = gene_name,
  stringsAsFactors = FALSE
)

dup_idx <- duplicated(df$motif_name) | duplicated(df$motif_name, fromLast = TRUE)
df$motif_name[dup_idx] <- paste0(df$motif_name[dup_idx], "_", df$gene_name[dup_idx])

df$`#motif_id` <- df$motif_name
df$motif_description <- df$gene_name
df$source_name <- source_name
df$source_version <- source_version
df$motif_similarity_qvalue <- 0
df$similar_motif_id <- "None"
df$similar_motif_description <- "None"
df$orthologous_identity <- 1
df$orthologous_gene_name <- "None"
df$orthologous_species <- "None"
df$description <- "gene is directly annotated"

df <- df[, c(
  "#motif_id",
  "motif_name",
  "motif_description",
  "source_name",
  "source_version",
  "gene_name",
  "motif_similarity_qvalue",
  "similar_motif_id",
  "similar_motif_description",
  "orthologous_identity",
  "orthologous_gene_name",
  "orthologous_species",
  "description"
)]

write.table(df, file = out_tbl, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
write.table(df$motif_name, file = out_motifs, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(df$motif_description, file = out_tf, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)

message("motif2TF table: ", out_tbl)
message("motif ID list: ", out_motifs)
message("TF list: ", out_tf)
