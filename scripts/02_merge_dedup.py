"""Merge reduced quarters, apply FDA deletions, deduplicate by caseid.

Usage: python3 02_merge_dedup.py <reduced_root> <merged_dir>

Primary deduplication (PROTOCOL section 2): one record per caseid, latest fda_dt,
ties broken by highest primaryid. Cases on any FDA deleted-case list are dropped.
Also flags second-stage duplicates for the sensitivity analysis.
No tabulation by sex happens here.
"""
import glob
import json
import os
import sys

import pandas as pd


def main(root, out):
    os.makedirs(out, exist_ok=True)
    qdirs = sorted(d for d in glob.glob(os.path.join(root, "*Q*")) if os.path.isdir(d))
    flow = {"quarters": [os.path.basename(d) for d in qdirs]}

    def read_demo(d):
        t = pd.read_parquet(os.path.join(d, "demo.parquet"))
        # event_dt may be partial (YYYY or YYYYMM). Keep it as an integer, 0 when absent.
        t["event_dt"] = pd.to_numeric(t["event_dt"], errors="coerce").fillna(0).astype("int32")
        return t
    demo = pd.concat([read_demo(d) for d in qdirs], ignore_index=True)
    for c in ["rept_cod", "sex", "occp_cod", "reporter_country", "occr_country", "quarter"]:
        demo[c] = demo[c].astype("category")
    flow["records_all_versions"] = int(len(demo))
    flow["distinct_caseids"] = int(demo["caseid"].nunique())

    dels = [os.path.join(d, "deleted.parquet") for d in qdirs
            if os.path.exists(os.path.join(d, "deleted.parquet"))]
    deleted = pd.concat([pd.read_parquet(p) for p in dels])["caseid"].unique() if dels else []
    flow["deleted_caseids_listed"] = int(len(deleted))
    is_del = demo["caseid"].isin(deleted)
    flow["records_removed_as_deleted"] = int(is_del.sum())
    demo = demo[~is_del]

    demo = demo.sort_values(["caseid", "fda_dt", "primaryid"])
    demo = demo.drop_duplicates("caseid", keep="last")
    flow["reports_after_caseid_dedup"] = int(len(demo))

    per = pd.concat(
        [pd.read_parquet(os.path.join(d, "perreport.parquet")) for d in qdirs],
        ignore_index=True).drop_duplicates("primaryid")
    per["ps_key"] = per["ps_key"].astype("category")
    rep = demo.merge(per, on="primaryid", how="left", validate="one_to_one")
    del demo, per

    # second-stage duplicate flag (sensitivity only)
    ok = rep["age_yr"].notna() & (rep["event_dt"] > 0) & rep["susp_hash"].notna() \
        & rep["pt_hash"].notna()
    key = ["sex", "age_yr", "event_dt", "reporter_country", "susp_hash", "pt_hash"]
    sub = rep.loc[ok, key + ["primaryid", "fda_dt"]].sort_values(["fda_dt", "primaryid"])
    dup = sub.duplicated(key, keep="last")
    rep["dup2"] = rep["primaryid"].isin(sub.loc[dup, "primaryid"])
    flow["second_stage_duplicates_flagged"] = int(rep["dup2"].sum())
    rep.to_parquet(os.path.join(out, "reports.parquet"), index=False)
    keep = pd.Index(rep["primaryid"])
    del rep, sub

    for name in ["suspect", "ptrows", "indirows"]:
        parts = []
        for d in qdirs:
            t = pd.read_parquet(os.path.join(d, f"{name}.parquet"))
            parts.append(t[t["primaryid"].isin(keep)])
        t = pd.concat(parts, ignore_index=True)
        for c in t.columns:
            if c != "primaryid":
                t[c] = t[c].astype("category")
        t.to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
        flow[f"{name}_rows"] = int(len(t))
        del t, parts

    ptc = pd.concat([pd.read_parquet(os.path.join(d, "ptcounts.parquet")) for d in qdirs])
    ptc = ptc.groupby("pt", as_index=False)["n"].sum().sort_values("n", ascending=False)
    ptc.to_parquet(os.path.join(out, "ptcounts.parquet"), index=False)
    flow["distinct_pts"] = int(len(ptc))

    qc = []
    for d in qdirs:
        with open(os.path.join(d, "qc.json")) as f:
            qc.append(json.load(f))
    flow["qc"] = qc
    with open(os.path.join(out, "flow.json"), "w") as f:
        json.dump(flow, f, indent=1)
    print({k: v for k, v in flow.items() if k not in ("qc", "quarters")})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
