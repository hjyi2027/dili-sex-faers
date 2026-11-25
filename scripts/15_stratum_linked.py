"""Scan by primary suspect stratum under different handling of suspected duplicates (post hoc, change 23).

Usage: python3 scripts/15_stratum_linked.py <analysis_dir> <results_dir>

The scan of 14_dependence.py (restricted set) is repeated on
  restricted         reports with at most 20 suspect ingredients (reference, must equal 14_dependence.py)
  restricted_linked  the restricted set with one report kept per linked group (Jaccard index at least 0.8)
  linked             the full set with one report kept per linked group
For each set: strata with at least 20 hepatic reports in each sex, Woolf variances that treat reports as
independent, random-effects pooling by restricted maximum likelihood, likelihood-ratio test of each stratum
against the pooled value and the Benjamini-Hochberg procedure at a nominal 5%.
The flags of the reference scan are compared with those of the two linked scans.
Each stratum of the reference scan is also tested against an odds ratio of 1, and the two sets of flags
are compared.

Writes <results_dir>/stratum_scan_sets.json and <results_dir>/stratum_scan_restricted.csv (reference scan
with empirical-Bayes shrunken estimates, used for the figure).
"""
import json
import math
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

D = import_module("14_dependence")
A = import_module("05_analyze")


def main(adir, rdir):
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                          columns=["primaryid", "sex", "female", "overdose", "narrow", "ps_drug"])
    rep = rep[rep["sex"].notna() & ~rep["overdose"]].sort_values("primaryid").reset_index(drop=True)
    pid = rep["primaryid"].to_numpy()
    n = len(rep)
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    pr = pr[pr["primaryid"].isin(pid)].drop_duplicates()
    nsusp = np.bincount(np.searchsorted(pid, pr["primaryid"].to_numpy()), minlength=n)
    del pr
    le20 = nsusp <= 20
    female = rep["female"].to_numpy().astype(np.int64)
    narrow = rep["narrow"].to_numpy().astype(np.int64)
    dcode, dnames = pd.factorize(rep["ps_drug"].astype(str), sort=True)
    nd = len(dnames)
    lk = pd.read_parquet(os.path.join(adir, "link.parquet")).set_index("primaryid")
    drop = pd.Series(pid).map(lk["drop_j80"]).fillna(False).to_numpy().astype(bool)

    sets = {"restricted": le20, "restricted_linked": le20 & ~drop, "linked": ~drop}
    res, tabs = {}, {}
    for name, m in sets.items():
        cnt = np.bincount(dcode[m] * 4 + female[m] * 2 + narrow[m], minlength=nd * 4).reshape(-1, 4).astype(float)
        sc = D.scan(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0], names=dnames)
        t = sc.pop("table")
        mu, tau2 = math.log(sc["pooled_or"]), sc["tau"] ** 2
        t["eb_log_or"], t["eb_var"] = lib.eb_shrink(t["log_or"].to_numpy(), t["var"].to_numpy(), mu, tau2)
        sc.update({"reports": int(m.sum()), "hepatic_reports": int(narrow[m].sum()), "mu": mu})
        res[name], tabs[name] = sc, t
    ref = tabs["restricted"]
    # each stratum of the reference scan also tested against an odds ratio of 1 (same test and procedure)
    lrt1 = np.array([A.lrt_fixed_or(a, b, c, d, 1.0) for a, b, c, d in
                     zip(ref["f_case"], ref["f_non"], ref["m_case"], ref["m_non"])])
    ref["q_one"] = lib.bh_fdr(chi2.sf(lrt1, 1))
    fp, f1 = (ref["q_lrt"] < 0.05).to_numpy(), (ref["q_one"] < 0.05).to_numpy()
    below = (ref["log_or"] < 0).to_numpy()
    res["restricted"]["vs_one"] = {
        "flagged_vs_pooled": int(fp.sum()), "flagged_vs_one": int(f1.sum()), "flagged_by_both": int((fp & f1).sum()),
        "only_vs_pooled": int((fp & ~f1).sum()), "only_vs_one": int((~fp & f1).sum()),
        "only_vs_one_below_one": int((~fp & f1 & below).sum()),
        "only_vs_one_above_one": int((~fp & f1 & ~below).sum())}
    ref.to_csv(os.path.join(rdir, "stratum_scan_restricted.csv"), index=False)
    rflag = set(ref.loc[ref["q_lrt"] < 0.05, "stratum"].astype(str))
    rall = set(ref["stratum"].astype(str))
    mu_ref = res["restricted"]["mu"]
    rdir_sign = dict(zip(ref["stratum"].astype(str), np.sign(ref["log_or"] - mu_ref)))
    for name in ("restricted_linked", "linked"):
        t = tabs[name]
        flag = set(t.loc[t["q_lrt"] < 0.05, "stratum"].astype(str))
        allx = set(t["stratum"].astype(str))
        sign = dict(zip(t["stratum"].astype(str), np.sign(t["log_or"] - res[name]["mu"])))
        both = rflag & flag
        res[name]["vs_restricted"] = {
            "strata_in_both_scans": len(rall & allx),
            "flagged_in_both": len(both),
            "flagged_in_both_same_direction": int(sum(rdir_sign[s] == sign[s] for s in both)),
            "flagged_restricted_only": len((rflag - flag) & allx),
            "flagged_here_only": len((flag - rflag) & rall),
            "restricted_flags_not_tested_here": len(rflag - allx),
            "share_of_restricted_flags_kept_pct": 100 * len(both) / len(rflag)}
    json.dump(res, open(os.path.join(rdir, "stratum_scan_sets.json"), "w"), indent=1)
    print("stratum scan sets:", json.dumps(res)[:1200])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
