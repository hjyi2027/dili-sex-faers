"""DILIrank input: entries that share an ingredient name after normalization, and the reports they touch (change 31).

Usage: python3 21_dilirank_check.py <analysis_dir> <results_dir>
Reads data/ext/dilirank.csv (sheet "version 2" of the FDA DILIrank 2.0 workbook as CSV) through the loader of 03_build.py.
"""
import json
import os
import sys
from importlib import import_module

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

B = import_module("03_build")


def main(adir, rdir):
    used = B.load_dilirank()
    d = pd.read_csv(os.path.join(HERE, "..", "data", "ext", "dilirank.csv"), dtype=str).sort_values("LTKBID", kind="stable")
    d["cls"] = d["vDILI-Concern"].str.lower().str.replace("v", "", n=1).str.split("-").str[0]
    d["ing"] = d["CompoundName"].map(lib.norm_ingredient)
    a = pd.read_parquet(os.path.join(adir, "analysis.parquet"), columns=["ps_drug", "dilirank", "sex", "overdose"])
    a = a[a["sex"].notna() & ~a["overdose"]]
    ps = a["ps_drug"].astype(str)
    out = {"rows": int(len(d)), "class_counts": d["cls"].value_counts().to_dict(),
           "ingredients_with_two_entries_of_different_class": {}}
    for k, v in d[d["ing"].duplicated(keep=False)].groupby("ing")["cls"].agg(list).items():
        if len(set(v)) > 1:
            m = ps == k
            out["ingredients_with_two_entries_of_different_class"][k] = {
                "classes_in_ltkbid_order": v, "class_used": used[k], "reports_as_primary_suspect": int(m.sum()),
                "class_in_data": a.loc[m, "dilirank"].astype(str).value_counts().to_dict()}
    conf = list(out["ingredients_with_two_entries_of_different_class"])
    out["share_of_reports_in_these_strata_pct"] = round(100 * ps.isin(conf).sum() / len(a), 3)
    with open(os.path.join(rdir, "dilirank_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out)[:400])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
