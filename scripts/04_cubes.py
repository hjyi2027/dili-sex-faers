"""Aggregate the analysis table into count cubes that every analysis reads.

Usage: python3 04_cubes.py <analysis_dir> <cube_dir> [--permute-sex SEED]

--permute-sex shuffles the sex column among reports with known sex. It exists
only to test the downstream code without looking at real sex differences.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPORT_DIMS = ["ps_drug", "age_grp", "period", "region", "reporter", "female", "narrow",
               "broad", "pheno", "death", "severe", "overdose", "dup2", "dilirank",
               "ning_band", "alcohol_proxy", "liverdx_proxy", "wt_cat", "nsusp_band", "ca",
               "link_code"]
PAIR_DIMS = ["ingredient", "is_ps", "age_grp", "period", "region", "reporter", "female",
             "narrow", "broad", "pheno", "death", "overdose", "dup2", "nsusp_band", "ca",
             "link_code"]


MERGED = "merged"


def link_flags(cube):
    """Restore the three linkage flags from the packed code."""
    code = cube["link_code"].to_numpy()
    cube["link_drop"] = (code & 1) > 0
    cube["link_drop9"] = (code & 2) > 0
    cube["link_dropf"] = (code & 4) > 0


def main(adir, cdir, seed=None):
    os.makedirs(cdir, exist_ok=True)
    need = ["primaryid", "sex", "female", "age_yr", "wt_kg", "period", "region", "reporter", "ps_drug",
            "narrow", "broad", "pheno", "death", "severe", "overdose", "dup2", "dilirank", "ning_band",
            "alcohol_proxy", "liverdx_proxy", "country"]
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"), columns=need)
    flow = {"reports_deduplicated": int(len(rep))}
    rep = rep[rep["sex"].notna()].copy()
    flow["sex_known"] = int(len(rep))
    if seed is not None:
        rng = np.random.default_rng(seed)
        rep["female"] = rng.permutation(rep["female"].to_numpy())
        flow["PERMUTED_SEX_TEST_ONLY"] = seed
    # Post hoc data-quality dimensions (see PROTOCOL amendment 9): number of distinct
    # suspect ingredients on the report, and whether the reporter country is Canada.
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    ns = pr.groupby("primaryid", observed=True, sort=False)["ingredient"].nunique()
    del pr
    nsv = rep["primaryid"].map(ns).fillna(0)
    rep["nsusp_band"] = pd.cut(nsv, [-1, 5, 10, 20, 50, 100000],
                               labels=["0-5", "6-10", "11-20", "21-50", "51+"]).astype(str)
    del ns, nsv
    rep["ca"] = rep["country"].astype(str) == "CA"
    # Post hoc change 19: reports dropped when linked groups of near-identical long-list
    # reports are collapsed to one report (thresholds 0.8 and 0.9, see 03c_linkage.py).
    lk = pd.read_parquet(os.path.join(adir, "link.parquet"))
    # One small code keeps memory down: bit 1 = dropped at 0.8 (latest kept), bit 2 = dropped at 0.9,
    # bit 4 = dropped at 0.8 when the earliest report is kept. The cubes get the three flags back.
    rep["link_code"] = (rep["primaryid"].isin(set(lk.loc[lk["drop_j80"], "primaryid"])).astype("int8")
                        + 2 * rep["primaryid"].isin(set(lk.loc[lk["drop_j90"], "primaryid"])).astype("int8")
                        + 4 * rep["primaryid"].isin(set(lk.loc[lk["dropfirst_j80"], "primaryid"])).astype("int8"))
    del lk
    rep = rep.drop(columns=["country"])
    age = rep["age_yr"]
    rep["age_grp"] = np.select(
        [age.isna(), age < 18, age < 50, age < 65], ["missing", "under18", "18-49", "50-64"],
        default="65+")
    adult_w = rep.loc[(age >= 18) & rep["wt_kg"].notna(), "wt_kg"]
    cuts = [float(x) for x in adult_w.quantile([1 / 3, 2 / 3])]
    flow["weight_tertile_cuts_kg"] = cuts
    rep["wt_cat"] = np.select(
        [rep["wt_kg"].isna(), rep["wt_kg"] <= cuts[0], rep["wt_kg"] <= cuts[1]],
        ["missing", "T1", "T2"], default="T3")
    flow["overdose_context"] = int(rep["overdose"].sum())
    flow["adult_age_known"] = int((age >= 18).sum())
    flow["age_missing"] = int(age.isna().sum())
    flow["under18"] = int((age < 18).sum())
    flow["narrow_cases_sexknown"] = int(rep["narrow"].sum())
    flow["broad_cases_sexknown"] = int(rep["broad"].sum())
    flow["weight_known_adults"] = int(len(adult_w))

    for c in ["ps_drug", "pheno", "dilirank", "ning_band", "period", "region", "reporter",
              "age_grp", "wt_cat", "nsusp_band"]:
        rep[c] = rep[c].astype(str).astype("category")
    rep = rep.drop(columns=["age_yr", "wt_kg", "sex"])
    cube = rep.groupby(REPORT_DIMS, observed=True, sort=False).size().rename("n").reset_index()
    for c in ["ps_drug", "pheno", "dilirank", "ning_band", "period", "region", "reporter",
              "age_grp", "wt_cat", "nsusp_band"]:
        cube[c] = cube[c].astype(str)
    link_flags(cube)
    cube.to_parquet(os.path.join(cdir, "report_cube.parquet"), index=False)
    flow["report_cube_rows"] = int(len(cube))

    cols = ["age_grp", "period", "region", "reporter", "female", "narrow",
            "broad", "pheno", "death", "overdose", "dup2", "nsusp_band", "ca",
            "link_code"]
    small = rep[["primaryid"] + cols].copy()
    del rep, cube
    for c in ["age_grp", "period", "region", "reporter", "pheno", "nsusp_band"]:
        small[c] = small[c].astype("category")
    small = small.set_index("primaryid")
    pairs = pd.read_parquet(os.path.join(adir, "pairs.parquet"))
    pairs["is_ps"] = pairs["role_cod"].astype(str) == "PS"
    pairs = pairs.drop(columns=["role_cod"])
    parts, n_pairs = [], 0
    K = 16
    bucket = (pairs["primaryid"] % K).to_numpy()
    for k in range(K):
        ch = pairs[bucket == k]
        ch = ch.groupby(["primaryid", "ingredient"], observed=True, sort=False)["is_ps"].max().reset_index()
        ch = ch.join(small, on="primaryid", how="inner")
        n_pairs += len(ch)
        g = ch.groupby(PAIR_DIMS, observed=True, sort=False).size().rename("n").reset_index()
        for c in ["ingredient", "age_grp", "period", "region", "reporter", "pheno", "nsusp_band"]:
            g[c] = g[c].astype(str)
        parts.append(g)
        del ch, g
    pc = pd.concat(parts, ignore_index=True).groupby(PAIR_DIMS, observed=True, sort=False)["n"].sum().reset_index()
    link_flags(pc)
    pc.to_parquet(os.path.join(cdir, "pair_cube.parquet"), index=False)
    flow["pair_cube_rows"] = int(len(pc))
    flow["suspect_pairs_sexknown"] = int(n_pairs)
    with open(os.path.join(cdir, "flow.json"), "w") as f:
        json.dump(flow, f, indent=1)
    print(flow)


if __name__ == "__main__":
    seed = int(sys.argv[sys.argv.index("--permute-sex") + 1]) if "--permute-sex" in sys.argv else None
    if "--merged" in sys.argv:
        MERGED = sys.argv[sys.argv.index("--merged") + 1]
    main(sys.argv[1], sys.argv[2], seed)
