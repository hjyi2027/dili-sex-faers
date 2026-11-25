"""Attach the indication of the primary suspect product and competing-cause terms to reports.

Usage: python3 03b_indication.py <quarter_root> <analysis_dir>

Reads the per-quarter psindi.parquet and compete.parquet written by 01b_indication.py and
keeps the rows of reports that survived deduplication. Writes <analysis_dir>/ps_indi.parquet
with one row per such report: primaryid, ps_indi ('' when not given), compete ('' when none,
else the groups joined by '+'). Post hoc change 18. No tabulation by sex happens here.
"""
import os
import sys

import pandas as pd


def main(qroot, adir):
    ids = pd.read_parquet(os.path.join(adir, "analysis.parquet"), columns=["primaryid"])["primaryid"]
    idx = pd.Index(ids)
    pi, cp = [], []
    for qd in sorted(os.listdir(qroot)):
        f1 = os.path.join(qroot, qd, "psindi.parquet")
        if not os.path.exists(f1):
            continue
        t = pd.read_parquet(f1)
        t = t[(t["ps_indi"] != "") & t["primaryid"].isin(idx)]
        t["ps_indi"] = t["ps_indi"].astype("category")
        pi.append(t)
        t = pd.read_parquet(os.path.join(qroot, qd, "compete.parquet"), columns=["primaryid", "grp"])
        cp.append(t[t["primaryid"].isin(idx)])
    cats = sorted(set().union(*[set(t["ps_indi"].cat.categories) for t in pi]))
    for t in pi:
        t["ps_indi"] = t["ps_indi"].cat.set_categories(cats)
    pi = pd.concat(pi, ignore_index=True).drop_duplicates("primaryid", keep="last")
    cp = pd.concat(cp, ignore_index=True).drop_duplicates()
    cg = cp.sort_values("grp").groupby("primaryid")["grp"].agg("+".join).rename("compete")
    out = pd.DataFrame({"primaryid": ids.to_numpy()})
    out = out.merge(pi, on="primaryid", how="left").merge(cg.reset_index(), on="primaryid", how="left")
    out["ps_indi"] = out["ps_indi"].cat.add_categories([""]).fillna("")
    out["compete"] = out["compete"].fillna("").astype("category")
    out.to_parquet(os.path.join(adir, "ps_indi.parquet"), index=False)
    print("reports", len(out), "| with indication", int((out["ps_indi"] != "").sum()),
          "| with competing-cause term", int((out["compete"] != "").sum()),
          "| distinct indications", int(out["ps_indi"].nunique()) - 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
