import os
import sys

import loompy as lp
import pandas as pd
from pyscenic.binarization import binarize
from pyscenic.cli.utils import load_signatures


f_pyscenic_output = sys.argv[1]
regulon_file = sys.argv[2]
threads = int(sys.argv[3])
min_regulon_size = int(sys.argv[4])
outputdir = sys.argv[5]


def get_motif_logo(regulon):
    base_url = "http://motifcollections.aertslab.org/v9/logos/"
    for elem in regulon.context:
        if elem.endswith(".png"):
            return base_url + elem
    return ""


def write_regulon_exports(regulons, selected_names, out_dir):
    gmt_file = os.path.join(out_dir, "regulons.gmt")
    txt_file = os.path.join(out_dir, "regulons.txt")
    with open(gmt_file, "w", encoding="utf-8") as fo1, open(
        txt_file, "w", encoding="utf-8"
    ) as fo2:
        for regulon in regulons:
            if regulon.name not in selected_names:
                continue
            motif = get_motif_logo(regulon)
            genes = "\t".join(regulon.genes)
            tf = "%s(%sg)" % (regulon.transcription_factor, len(regulon.genes))
            fo1.write("%s\t%s\t%s\n" % (tf, motif, genes))
            fo2.write("%s\t%s\t%s\n" % (tf, motif, genes.replace("\t", ",")))


def load_auc_matrix_from_loom(loom_path, select_cols):
    if not os.path.exists(loom_path):
        return None
    with lp.connect(loom_path, mode="r", validate=False) as lf:
        if hasattr(lf.ca, "RegulonsAUC"):
            auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
            missing_cols = [col for col in select_cols if col not in auc_mtx.columns]
            if missing_cols:
                raise KeyError(
                    "Missing regulon columns in loom RegulonsAUC: "
                    + ", ".join(missing_cols[:10])
                )
            return auc_mtx[select_cols]
    return None


def load_auc_matrix(loom_path, output_dir, select_cols):
    auc_from_loom = load_auc_matrix_from_loom(loom_path, select_cols)
    if auc_from_loom is not None:
        return auc_from_loom

    fallback_csv = os.path.join(output_dir, "AUCell.csv")
    if os.path.exists(fallback_csv):
        auc_mtx = pd.read_csv(fallback_csv, index_col=0)
        missing_cols = [col for col in select_cols if col not in auc_mtx.columns]
        if missing_cols:
            raise KeyError(
                "Missing regulon columns in AUCell.csv: " + ", ".join(missing_cols[:10])
            )
        return auc_mtx[select_cols]

    raise AttributeError(
        "Could not find RegulonsAUC in loom and no fallback AUCell.csv exists in "
        + output_dir
    )


print(
    """
##############################################
    1. Transform regulons to gmt file ...
##############################################
"""
)
regulons = load_signatures(regulon_file)
select_cols = [reg.name for reg in regulons if len(reg.genes) >= min_regulon_size]
write_regulon_exports(regulons, set(select_cols), outputdir)

print(
    """
##############################################
    2. Collect SCENIC AUCell output ...
##############################################
"""
)
auc_mtx = load_auc_matrix(f_pyscenic_output, outputdir, select_cols)
auc_mtx.to_csv(os.path.join(outputdir, "AUCell.txt"), sep="\t")

print(
    """
######################################################
    3. Generate a binary regulon activity matrix ...
######################################################
"""
)
binary_mtx, auc_thresholds = binarize(auc_mtx, num_workers=threads)
binary_mtx.to_csv(os.path.join(outputdir, "binary_mtx.txt"), sep="\t")
auc_thresholds.to_csv(os.path.join(outputdir, "auc_thresholds.txt"), sep="\t")
