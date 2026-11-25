"""Repeated null simulations for the pooled contrasts and the drug-level scan (post hoc change 19).

Usage: python3 13_null_sims.py <analysis_dir> <results_dir> [n_permutations] [n_parametric]

Part 1. Sex is permuted across reports of the restricted set B times. Each time the script
recomputes the pooled odds ratios stratified by primary suspect drug (hepatic reporting,
cholestatic term, fatality ratio) and the primary-suspect drug-level scan. It records how
often a 95% interval excludes 1 and how many ingredients are flagged.

Part 3 repeats Part 1 with sex permuted within primary suspect drug, which keeps the mix of
women and men of every drug and removes only the within-drug association.

Part 2. The scan is simulated under a common shift: every ingredient has the pooled odds
ratio estimated from the data, with its observed numbers of reports by sex and its observed
hepatic share in men. This checks the likelihood-ratio test against the estimated pooled
value and the false discovery rate control when the reference is not 1.

Writes <results_dir>/null_sims.json.
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

A = import_module("05_analyze")
Z, MIN_CASES = A.Z, A.MIN_CASES
SEED = 20261124


def scan_from_tables(fc, fn, mc, mn):
    keep = (fc >= MIN_CASES) & (mc >= MIN_CASES)
    fc, fn, mc, mn = fc[keep], fn[keep], mc[keep], mn[keep]
    y, v = lib.woolf(fc, fn, mc, mn)
    het = lib.reml_tau2(y, v)
    psi = math.exp(het["mu"])
    lrt = np.array([A.lrt_fixed_or(a, b, c, d, psi) for a, b, c, d in zip(fc, fn, mc, mn)])
    q = lib.bh_fdr(chi2.sf(lrt, 1))
    lrt1 = np.array([A.lrt_fixed_or(a, b, c, d, 1.0) for a, b, c, d in zip(fc, fn, mc, mn)])
    q1 = lib.bh_fdr(chi2.sf(lrt1, 1))
    return {"k": int(keep.sum()), "mu": het["mu"], "tau": math.sqrt(het["tau2"]),
            "flagged": int((q < 0.05).sum()), "flagged_vs_one": int((q1 < 0.05).sum()),
            "min_p": float(chi2.sf(lrt, 1).min())}


def mh_by_code(code, n_code, female, y):
    """Mantel-Haenszel log OR and variance, women versus men, strata given by integer codes."""
    cnt = np.bincount(code * 4 + female * 2 + y, minlength=n_code * 4).reshape(-1, 4).astype(float)
    return lib.mantel_haenszel(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])[:2]


def main(adir, rdir, B=200, S=1000):
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                          columns=["primaryid", "sex", "female", "overdose", "narrow", "broad",
                                   "death", "pheno", "ps_drug"])
    rep = rep[rep["sex"].notna() & ~rep["overdose"]]
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"))
    ns = pr.groupby("primaryid", observed=True, sort=False)["ingredient"].nunique()
    rep = rep[rep["primaryid"].map(ns).fillna(0) <= 20].sort_values("primaryid").reset_index(drop=True)
    del ns
    pid = rep["primaryid"].to_numpy()
    ps = pr[(pr["role_cod"].astype(str) == "PS") & pr["primaryid"].isin(pid)][["primaryid", "ingredient"]].drop_duplicates()
    del pr
    prow = np.searchsorted(pid, ps["primaryid"].to_numpy())
    icode, inames = pd.factorize(ps["ingredient"].astype(str), sort=True)
    n_ing = len(inames)
    female = rep["female"].to_numpy().astype(np.int64)
    narrow = rep["narrow"].to_numpy().astype(np.int64)
    death = rep["death"].to_numpy().astype(np.int64)
    dcode, dnames = pd.factorize(rep["ps_drug"].astype(str), sort=True)
    n_drug = len(dnames)
    ph = rep["pheno"].astype(str).isin(["HC", "CH"]).to_numpy()
    ch = (rep["pheno"].astype(str) == "CH").to_numpy().astype(np.int64)
    hep = narrow.astype(bool)
    other = ~rep["broad"].to_numpy() & ~hep
    other &= np.isin(dcode, np.unique(dcode[hep]))
    nar_p = narrow[prow]

    def one(f):
        cnt = np.bincount(icode * 4 + f[prow] * 2 + nar_p, minlength=n_ing * 4).reshape(-1, 4).astype(float)
        sc = scan_from_tables(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])
        y1, v1 = mh_by_code(dcode, n_drug, f, narrow)
        y2, v2 = mh_by_code(dcode[ph], n_drug, f[ph], ch[ph])
        yh, vh = mh_by_code(dcode[hep], n_drug, f[hep], death[hep])
        yo, vo = mh_by_code(dcode[other], n_drug, f[other], death[other])
        return sc, (y1, v1), (y2, v2), (yh - yo, vh + vo)

    obs = one(female)
    rng = np.random.default_rng(SEED)
    scans, pooled = [], {"reporting": [], "phenotype": [], "fatality_ratio": []}
    for _ in range(B):
        sc, r, p, fr = one(rng.permutation(female))
        scans.append(sc)
        for k, (y, v) in zip(pooled, (r, p, fr)):
            pooled[k].append((y, v))
    res = {"set": "restricted", "permutations": B, "seed": SEED,
           "observed": {"scan": obs[0], "reporting_or": math.exp(obs[1][0]),
                        "phenotype_or": math.exp(obs[2][0]), "fatality_ratio": math.exp(obs[3][0])}}
    fl = np.array([s["flagged"] for s in scans])
    res["permuted_scan"] = {
        "ingredients_mean": float(np.mean([s["k"] for s in scans])),
        "flagged_mean": float(fl.mean()), "flagged_max": int(fl.max()),
        "share_with_any_flag": float((fl > 0).mean()),
        "pooled_or_mean": float(np.exp(np.mean([s["mu"] for s in scans]))),
        "tau_mean": float(np.mean([s["tau"] for s in scans])),
        "tau_max": float(np.max([s["tau"] for s in scans])),
        "flagged_vs_one_mean": float(np.mean([s["flagged_vs_one"] for s in scans]))}
    res["permuted_pooled"] = {}
    for k, vals in pooled.items():
        y = np.array([a for a, _ in vals])
        se = np.sqrt(np.array([b for _, b in vals]))
        res["permuted_pooled"][k] = {
            "or_mean": float(np.exp(y.mean())), "or_min": float(np.exp(y.min())),
            "or_max": float(np.exp(y.max())),
            "sd_log_or": float(y.std(ddof=1)), "mean_model_se": float(se.mean()),
            "share_interval_excludes_1": float((np.abs(y / se) > Z).mean())}

    # Part 2: common-shift null for the scan
    t = pd.read_csv(os.path.join(rdir, "le20_ps", "n1_ingredients.csv"))
    y0, v0 = lib.woolf(t.f_case, t.f_non, t.m_case, t.m_non)
    psi = math.exp(lib.reml_tau2(y0, v0)["mu"])
    nf = (t.f_case + t.f_non).to_numpy().astype(np.int64)
    nm = (t.m_case + t.m_non).to_numpy().astype(np.int64)
    pm = (t.m_case / nm).to_numpy()
    pf = psi * pm / (1 - pm + psi * pm)
    sims = []
    for _ in range(S):
        fc = rng.binomial(nf, pf).astype(float)
        mc = rng.binomial(nm, pm).astype(float)
        sims.append(scan_from_tables(fc, nf - fc, mc, nm - mc))
    fl = np.array([s["flagged"] for s in sims])
    res["common_shift_scan"] = {
        "simulations": S, "common_or": psi, "ingredients": int(len(t)),
        "flagged_mean": float(fl.mean()), "flagged_max": int(fl.max()),
        "share_with_any_flag": float((fl > 0).mean()),
        "pooled_or_mean": float(np.exp(np.mean([s["mu"] for s in sims]))),
        "tau_mean": float(np.mean([s["tau"] for s in sims])),
        "tau_p95": float(np.quantile([s["tau"] for s in sims], 0.95)),
        "flagged_vs_one_mean": float(np.mean([s["flagged_vs_one"] for s in sims])),
        "flagged_vs_one_min": int(np.min([s["flagged_vs_one"] for s in sims])),
        "flagged_vs_one_max": int(np.max([s["flagged_vs_one"] for s in sims]))}
    # Part 3: sex permuted within primary suspect drug, which keeps each drug's mix of women and men
    by_drug = np.argsort(dcode, kind="stable")
    f_sorted = female[by_drug]
    scans, pooled = [], {"reporting": [], "phenotype": [], "fatality_ratio": []}
    stratum_flags = []
    for _ in range(B):
        order = np.argsort(dcode + rng.random(len(dcode)))
        f = np.empty_like(female)
        f[order] = f_sorted
        sc, r, p, fr = one(f)
        scans.append(sc)
        # the same scan with the primary suspect stratum as the unit in place of the ingredient
        cnt = np.bincount(dcode * 4 + f * 2 + narrow, minlength=n_drug * 4).reshape(-1, 4).astype(float)
        stratum_flags.append(scan_from_tables(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])["flagged"])
        for k, (y, v) in zip(pooled, (r, p, fr)):
            pooled[k].append((y, v))
    fl = np.array([s["flagged"] for s in scans])
    res["within_drug_permuted_scan"] = {
        "permutations": B,
        "ingredients_mean": float(np.mean([s["k"] for s in scans])),
        "flagged_mean": float(fl.mean()), "flagged_max": int(fl.max()),
        "share_with_any_flag": float((fl > 0).mean()),
        "pooled_or_mean": float(np.exp(np.mean([s["mu"] for s in scans]))),
        "tau_mean": float(np.mean([s["tau"] for s in scans])),
        "tau_max": float(np.max([s["tau"] for s in scans]))}
    span = pd.DataFrame({"i": icode, "d": dcode[prow]}).drop_duplicates().groupby("i").size()
    res["within_drug_permuted_scan_by_stratum"] = {
        "flagged_mean": float(np.mean(stratum_flags)), "flagged_max": int(np.max(stratum_flags)),
        "ingredients": int(n_ing), "ingredients_in_more_than_one_stratum": int((span > 1).sum())}
    res["within_drug_permuted_pooled"] = {}
    for k, vals in pooled.items():
        y = np.array([a for a, _ in vals])
        se = np.sqrt(np.array([b for _, b in vals]))
        res["within_drug_permuted_pooled"][k] = {
            "or_mean": float(np.exp(y.mean())), "or_min": float(np.exp(y.min())),
            "or_max": float(np.exp(y.max())),
            "sd_log_or": float(y.std(ddof=1)), "mean_model_se": float(se.mean()),
            "share_interval_excludes_1": float((np.abs(y / se) > Z).mean())}
    with open(os.path.join(rdir, "null_sims.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("null simulations:", json.dumps({k: res[k] for k in ("permuted_scan", "common_shift_scan")})[:400])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:5]))
