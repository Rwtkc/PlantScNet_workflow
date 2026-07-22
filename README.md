# PlantScNet 调控网络推断流程

_PlantScNet 从单细胞输入数据到物种级候选调控网络的完整计算流程。_

---

## 1. 主要软件版本

本流程整理时使用的主要软件版本如下：

| 流程部分 | 软件版本 |
| --- | --- |
| scRNA-seq 样本处理 | R 4.1.3; Seurat 4.3.0; Matrix 1.5.4.1; data.table 1.14.8; SCopeLoomR 0.13.0 |
| scATAC-seq 样本处理 | R 4.5.3; Signac 1.17.1; Seurat 5.5.0; Matrix 1.7.5; data.table 1.17.8; GenomicRanges 1.62.1 |
| Genome interval 和 cisTarget 相关处理 | Python 3.10.16; samtools 1.23.1; bedtools 2.31.1; NumPy 2.2.6; pandas 2.2.3; pyarrow 20.0.0 |
| 物种级 XGBoost 整合 | Python 3.11.13; XGBoost 3.0.1; Optuna 4.6.0; scikit-learn 1.7.1; NumPy 2.3.1; pandas 2.3.2; SciPy 1.16.0 |

---

## 2. 这个目录包含什么

当前目录以拟南芥为示例物种，保留了完整流程所需的代表性输入文件和通用脚本模板。

### 2.1 拟南芥示例输入文件

| 文件 | 用途 |
| --- | --- |
| `Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz` | 拟南芥 genome FASTA，用于构建 FASTA index 和提取 promoter 序列 |
| `Arabidopsis_thaliana.TAIR10.63.gtf.gz` | 拟南芥 GTF 注释，用于提取 gene 坐标和 TSS |
| `ath_tf.txt` | 拟南芥 TF list，用于限制 GRNBoost2 的 candidate regulator 集合 |
| `ath.tbl` | motif-to-TF 注释表，用于 pySCENIC `ctx`，也可作为 ATAC motif-to-TF 映射示例 |
| `ath.meme` | 拟南芥 motif 文件；如需重新构建 cisTarget database，可由该文件整理为 Cluster-Buster motif 输入 |
| `ath.feather` | 已构建好的拟南芥 gene-based cisTarget ranking database，可直接用于 pySCENIC `ctx` |

### 2.2 脚本目录

| 路径 | 内容 |
| --- | --- |
| `scripts/cistarget/` | promoter BED/FASTA 提取和 gene-based cisTarget database 构建模板 |
| `scripts/rna_sample_network/` | scRNA 数据整理、Seurat 对象构建、loom 生成、GRNBoost2 和 pySCENIC 运行模板 |
| `scripts/scatac_sample_network/` | 基于 cluster-enriched peaks、promoter overlap 和 motif overlap 构建 scATAC 样本级候选边的模板 |
| `scripts/xgboost_integration/` | 跨样本特征矩阵和 XGBoost 物种级整合模板 |

---

## 3. 运行环境建议

建议把流程拆成三个 conda 环境，避免 R、pySCENIC、cisTarget 建库依赖互相冲突：

| 环境类型 | 用途 |
| --- | --- |
| cisTarget 建库环境 | 运行 `create_cistarget_motif_databases.py`、`cbust`、`samtools`、`bedtools` 和 MEME Suite/FIMO |
| R 分析环境 | 运行 Seurat、Signac、SCopeLoomR、GenomicRanges 和 R 后处理脚本 |
| Python 网络与模型环境 | 运行 Arboreto/GRNBoost2、pySCENIC、矩阵处理、Optuna 和 XGBoost |

环境名称可以自行设置；运行某一步前，激活包含该步骤依赖的软件环境即可。

cisTarget 建库依赖官方仓库：

```text
https://github.com/aertslab/create_cisTarget_databases
```

---

## 4. 完整运行顺序

### Step 1. 准备拟南芥参考文件

当前目录已经包含拟南芥示例输入：

```text
Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
Arabidopsis_thaliana.TAIR10.63.gtf.gz
ath_tf.txt
ath.tbl
ath.meme
ath.feather
```

如果换成其他物种，需要准备同等内容，并保证 genome annotation、表达矩阵 gene ID、TF list、motif-to-TF 表和训练标签使用同一套或可映射的基因 ID。

其中 `ath_tf.txt` 和 `ath.tbl` 应与 `ath.meme` 来自同一套 motif/TF 注释。PlantScNet 示例中从 MEME 文件的 `MOTIF` 行解析 TF gene ID 和 motif ID，并据此生成 pySCENIC 使用的 TF list、motif-to-TF table 以及 cisTarget 建库使用的 motif ID list：

```bash
Rscript scripts/cistarget/00_prepare_tf_list_and_motif2tf_from_meme.R \
  --meme ath.meme \
  --out-tbl ath.tbl \
  --out-tf ath_tf.txt \
  --out-motifs resources/ath/motif_ids.txt
```

该步骤用于保证 GRNBoost2 的 TF list、pySCENIC `ctx` 的 motif-to-TF 注释、scATAC motif-to-TF 映射和 cisTarget ranking database 使用一致的 motif/TF 命名。

### Step 2. 构建 promoter BED 和 promoter FASTA

使用 GTF 和 genome FASTA 提取 strand-aware 2 kb upstream promoter：

```bash
bash scripts/cistarget/01_build_promoter_fasta.sh \
  --gtf Arabidopsis_thaliana.TAIR10.63.gtf.gz \
  --genome Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz \
  --out-prefix resources/ath/promoters_2kb \
  --upstream 2000
```

输出：

```text
resources/ath/promoters_2kb.bed
resources/ath/promoters_2kb.fa
```

这两个文件后续分别用于：

- scRNA cisTarget database 构建或核对
- scATAC peak-to-gene promoter overlap

### Step 3. 准备或重建 cisTarget ranking database

如果已经使用当前目录中的 `ath.feather`，可以直接进入 Step 4。

如果需要从 promoter FASTA 和 motif 文件重新构建 gene-based cisTarget database，需要先从 `ath.meme` 整理出 Cluster-Buster 格式 motif 文件和 motif list，然后调用 `create_cistarget_motif_databases.py`：

```bash
bash scripts/cistarget/02_build_gene_based_cistarget_db.sh \
  --fasta resources/ath/promoters_2kb.fa \
  --motif-dir resources/ath/motifs_cb \
  --motif-list resources/ath/motif_ids.txt \
  --script /path/to/create_cisTarget_databases/create_cistarget_motif_databases.py \
  --out-prefix resources/ath/cistarget/ath_genes
```

实际运行前需要先下载官方建库脚本：

```bash
git clone https://github.com/aertslab/create_cisTarget_databases
```

### Step 4. 构建 scRNA 样本级网络

公共 scRNA-seq/snRNA-seq 数据的原始下载格式可能不同。进入 PlantScNet scRNA 样本级网络流程前，需要先将每个样本整理为 10x-like 三件套：

```text
matrix.mtx
features.tsv
barcodes.tsv
```

其中 `matrix.mtx` 为 gene-by-cell count matrix，`features.tsv` 至少第一列为 gene ID，`barcodes.tsv` 第一列为 cell barcode。具体格式可直接参考 10x Genomics filtered feature-barcode matrix 的输出形式。

每个 scRNA 样本先从整理后的 count matrix、features 和 barcodes 构建 Seurat 对象：

```bash
Rscript scripts/rna_sample_network/01_prepare_seurat_object.R \
  --matrix data/ath/raw/sample_001/matrix.mtx \
  --features data/ath/raw/sample_001/features.tsv \
  --barcodes data/ath/raw/sample_001/barcodes.tsv \
  --output data/ath/raw/sample_001/seurat_obj.rds
```

随后将 Seurat 对象导出为 loom 文件：

```bash
Rscript scripts/rna_sample_network/02_create_loom_from_seurat.R \
  --seurat data/ath/raw/sample_001/seurat_obj.rds \
  --output-dir data/ath/raw/sample_001/result
```

再运行 GRNBoost2 和 pySCENIC：

```bash
bash scripts/rna_sample_network/03_run_grnboost2_pyscenic.sh \
  --workdir data/ath/raw/sample_001/result \
  --tf-list ath_tf.txt \
  --rankings ath.feather \
  --motif2tf ath.tbl
```

脚本会依次完成 GRNBoost2 推断、pySCENIC motif enrichment、AUCell 计算和样本级 TF-target 表整理。

输出的样本级网络用于后续跨样本整合，核心列为：

```text
TF    target    importance_score
```

### Step 5. 构建 scATAC 样本级候选网络

公共 scATAC-seq 数据的原始格式差异较大，因此本流程不限定原始输入格式。进入 PlantScNet scATAC 样本级网络构建前，需要先将每个样本整理为 cluster-enriched peak BED 文件。

每个样本对应一个输入目录，目录中每个 BED 文件对应一个 cluster 的 enriched peaks：

```text
data/ath/atac/sample_001/enriched_peaks/
├── sample_001_cluster_0.bed
├── sample_001_cluster_1.bed
└── sample_001_cluster_2.bed
```

每个 BED 文件为无表头、制表符分隔文本，至少包含三列：

```text
chrom    start    end
Chr1     10500    10820
Chr1     22430    22910
Chr2     85100    85640
```

文件名用于标记该 peak set 的 cluster 来源，不要求包含 peak 数量。原始数据可以来自 H5、RDS、matrix/barcode/peak 文件或其他格式；只要最终整理为上述 BED 目录，即可进入下面的统一流程。

scATAC 网络构建还需要 promoter 注释和 motif hit 注释：

```text
resources/ath/promoters_2kb.bed
resources/ath/motif_hits.bed
```

`promoters_2kb.bed` 来自 Step 2，第四列为 target gene ID。`motif_hits.bed` 使用 MEME Suite 的 `fimo` 扫描 promoter FASTA 后生成：

```bash
bash scripts/scatac_sample_network/01_scan_promoter_motifs_with_fimo.sh \
  --promoter-fasta resources/ath/promoters_2kb.fa \
  --promoter-bed resources/ath/promoters_2kb.bed \
  --motif-meme ath.meme \
  --outdir resources/ath/fimo \
  --output-bed resources/ath/motif_hits.bed \
  --thresh 1e-4
```

输出的 `motif_hits.bed` 为无表头、制表符分隔文本，至少前四列为：

```text
chrom    start    end      motif_id
Chr1     10520    10531    MP00119
Chr1     22480    22491    MP00120
Chr2     85240    85251    MP00100
```

其中前三列为 motif hit 的基因组坐标，第四列为 motif ID。`ath.tbl` 用于把 motif ID 映射到 TF。

随后将 cluster-enriched peak BED 与 promoter 和 motif hit 注释相交。

一个候选 TF-target 边需要由同一个开放 peak 同时支持：

1. peak 与 target gene promoter 重叠
2. peak 覆盖 motif hit
3. motif 可以通过 `ath.tbl` 映射到 TF

运行 peak-motif-promoter 交集：

```bash
python scripts/scatac_sample_network/02_peak_motif_gene_to_tf_target.py \
  --input-dir data/ath/atac/sample_001/enriched_peaks \
  --motif-bed resources/ath/motif_hits.bed \
  --promoter-bed resources/ath/promoters_2kb.bed \
  --motif-tf-file ath.tbl \
  --output-dir data/ath/atac/sample_001/tf_target_intermediate \
  --workers 8
```

最后对 TF-target 候选边进行样本内排序：

```bash
python scripts/scatac_sample_network/03_rank_scatac_tf_target_list.py \
  --input data/ath/atac/sample_001/tf_target_intermediate/high_confidence_tf_target_with_clusters.tsv \
  --output-prefix data/ath/atac/sample_001/tf_target
```

默认评分为：

```text
importance_score = cluster_support * log2(1 + total_support_peaks)
```

其中：

- `cluster_support` 表示支持该候选边的 cluster 数
- `total_support_peaks` 表示支持该候选边的非重复 peak 数

简化输出同样使用：

```text
TF    target    importance_score
```

### Step 6. 合并样本级网络为跨样本特征矩阵

同一物种、同一模态的多个样本级 TF-target 文件合并为特征矩阵。每一行是一条候选 TF-target 边，每个样本贡献一组特征，而不是只贡献一个分数。

模板中的 XGBoost wrapper 会调用同一目录下的矩阵合并脚本和 Step1-Step4 脚本。如果没有额外的增强特征脚本，流程会使用基础 TF-target 矩阵继续运行。

### Step 7. 构建参考标签

XGBoost 训练需要参考 TF-target 标签：

- 拟南芥使用整理后的 Arabidopsis regulatory reference
- 非拟南芥物种可通过 Arabidopsis-to-target BBH 或同源映射投影标签

训练前必须检查：

```text
feature matrix 中的 TF-target
reference labels 中的 TF-target
```

二者需要有足够 overlap，否则应先排查 gene ID 版本、大小写、前缀和映射关系。

### Step 8. XGBoost 物种级整合

运行物种级整合模板：

```bash
bash scripts/xgboost_integration/run_species_xgboost_template.sh \
  --species ath \
  --samples-dir data/ath/samples \
  --gold data/ath/gold/ath_reference_labels.tsv \
  --out-dir data/ath/outputs/ath_run
```

该 wrapper 对应四步：

1. `Step1-CreateRobustDataset.py`：构建正负训练集、五折划分和训练元数据
2. `Step2-TPE_byessearch-v2.py`：使用 Optuna/TPE 搜索 XGBoost 参数
3. `Step3-makeModel-v2.py`：训练 integrated model，并与 single-sample block 比较
4. `Step4-PredictLargeData-v2.py`：对全部候选边进行预测

最终输出：

```text
final_regulatory_with_probability.tsv
```

该文件是物种级排序网络，可用于 PlantScNet 的 Browse、Search、Download 和 Tools 模块。
