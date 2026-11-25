"""Reports with more than one primary suspect record (post hoc, change 26).

Usage: python3 scripts/17_multi_ps.py <data_dir> <analysis_dir> <results_dir>

FAERS marks one drug record per report as primary suspect (role code PS). From 2024 some reports carry
several PS records. scripts/01_reduce.py takes the first PS record in file order as the primary suspect
product, which defines the stratum of every drug-stratified analysis. This script
  1. counts the PS records of every report in the raw DRUG files,
  2. reports how many reports of the analysis sets carry more than one, by year of receipt,
  3. repeats the main drug-stratified contrasts without those reports.
Writes <analysis_dir>/n_ps.parquet and <results_dir>/multi_ps.json.
"""
import glob
import json
import math
import os
import sys
import zipfile
from importlib import import_module

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

R = import_module("01_reduce")
Z = 1.959963984540054


def count_ps(ddir):
    parts = []
    for z in sorted(glob.glob(os.path.join(ddir, "faers_ascii_*.zip"))):
        zf = zipfile.ZipFile(z)
        drug, _ = R.read_table(zf, R.find_member(zf, "DRUG"), ["primaryid", "role_cod"])
        drug["primaryid"] = R.to_int(drug["primaryid"])
        drug = drug.dropna(subset=["primaryid"])
        ps = drug[drug["role_cod"].str.upper().str.strip() == "PS"]
        n = ps.groupby(ps["primaryid"].astype("int64")).size()
        parts.append(n[n > 1])
    s = pd.concat(parts)
    return s.groupby(level=0).max().rename("n_ps")


def est(y, v):
    return {"or": math.exp(y), "lo": math.exp(y - Z * math.sqrt(v)), "hi": math.exp(y + Z * math.sqrt(v))}


def main(ddir, adir, rdir):
    nps = count_ps(ddir)
    nps.reset_index().rename(columns={"index": "primaryid"}).to_parquet(os.path.join(adir, "n_ps.parquet"), index=False)
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                          columns=["primaryid", "year", "sex", "female", "overdose", "narrow", "broad", "death",
                                   "pheno", "ps_drug", "age_band", "adult"])
    rep = rep[rep["sex"].notna() & ~rep["overdose"]].sort_values("primaryid").reset_index(drop=True)
    pid = rep["primaryid"].to_numpy()
    n = len(rep)
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    pr = pr[pr["primaryid"].isin(pid)].drop_duplicates()
    nsusp = np.bincount(np.searchsorted(pid, pr["primaryid"].to_numpy()), minlength=n)
    del pr
    le20 = nsusp <= 20
    multi = pd.Series(pid).map(nps).fillna(1).to_numpy() > 1
    female = rep["female"].to_numpy().astype(np.int64)
    narrow = rep["narrow"].to_numpy().astype(np.int64)
    death = rep["death"].to_numpy().astype(np.int64)
    hep = narrow.astype(bool)
    notbroad = ~rep["broad"].to_numpy() & ~hep
    phs = rep["pheno"].astype(str)
    ph = phs.isin(["HC", "CH"]).to_numpy()
    ch = (phs == "CH").to_numpy().astype(np.int64)
    adult = rep["adult"].to_numpy().astype(bool)
    dcode, dnames = pd.factorize(rep["ps_drug"].astype(str), sort=True)
    dacode, _ = pd.factorize(rep["ps_drug"].astype(str) + "|" + rep["age_band"].astype(str), sort=True)
    year = rep["year"].to_numpy()

    def mh(code, f, y):
        k = int(code.max()) + 1 if len(code) else 1
        cnt = np.bincount(code * 4 + f * 2 + y, minlength=k * 4).reshape(-1, 4).astype(float)
        return lib.mantel_haenszel(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])[:2]

    def contrasts(m):
        out = {"reports": int(m.sum()), "hepatic_reports": int((m & hep).sum())}
        out["reporting_by_drug"] = est(*mh(dcode[m], female[m], narrow[m]))
        mp = m & ph
        out["phenotype_by_drug"] = est(*mh(dcode[mp], female[mp], ch[mp]))
        mpa = mp & adult
        out["phenotype_adult_by_drug_age"] = est(*mh(dacode[mpa], female[mpa], ch[mpa]))
        out["phenotype_adult_reports"] = int(mpa.sum())
        mh_ = m & hep
        mo = m & notbroad & np.isin(dcode, np.unique(dcode[mh_]))
        yh, vh = mh(dcode[mh_], female[mh_], death[mh_])
        yo, vo = mh(dcode[mo], female[mo], death[mo])
        out["fatality_ratio_by_drug"] = est(yh - yo, vh + vo)
        return out

    res = {"rule": "first primary suspect record in file order defines the primary suspect product",
           "records_with_more_than_one_ps_all_versions": int(len(nps))}
    for name, m in (("full", np.ones(n, dtype=bool)), ("restricted", le20)):
        mm = m & multi
        yrs = pd.Series(year[mm]).value_counts().sort_index()
        res[name] = {
            "reports": int(m.sum()), "multi_ps_reports": int(mm.sum()), "multi_ps_pct": 100 * mm.sum() / m.sum(),
            "multi_ps_hepatic": int((mm & hep).sum()), "multi_ps_pct_of_hepatic": 100 * (mm & hep).sum() / (m & hep).sum(),
            "multi_ps_female_pct": 100 * female[mm].mean(), "female_pct_all": 100 * female[m].mean(),
            "first_year_with_multi_ps": int(yrs.index.min()),
            "multi_ps_by_year": {str(int(k)): int(v) for k, v in yrs.items()},
            "multi_ps_pct_from_2024": 100 * mm.sum() / (m & (year >= 2024)).sum(),
            "all_reports": contrasts(m), "single_ps_only": contrasts(m & ~multi)}
    json.dump(res, open(os.path.join(rdir, "multi_ps.json"), "w"), indent=1)
    print("multi ps:", json.dumps(res)[:1500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
