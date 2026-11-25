"""Dependence between reports, the stratum-level scan and a joint fatality model (post hoc change 21).

Usage: python3 14_dependence.py <analysis_dir> <results_dir> [n_bootstrap]

A. Scan with the primary suspect stratum as the unit (restricted set). Under a permutation of sex
   within primary suspect drug this scan holds its false discovery rate, the scan by ingredient does not.
B. Cluster bootstrap. Suspected duplicate groups (record linkage, Jaccard 0.8) are the sampling units and
   every other report is its own unit. Each replicate draws Poisson(1) weights per unit. The standard
   deviation of the log estimates is compared with the model standard error.
C. Other ways of handling the linked groups, full set: collapse only groups that pass every available
   check of weight and event date, collapse only groups with a recorded age, and keep all reports with
   weight 1 / group size.
D. Joint model of the sex by hepatic interaction on fatal outcome: binomial regression with an intercept,
   a sex effect and a hepatic effect for every drug and one common interaction.

Writes <results_dir>/dependence.json and <results_dir>/stratum_scan.csv.
"""
import json
import math
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

A = import_module("05_analyze")
Z, MIN_CASES = A.Z, A.MIN_CASES
SEED = 20261125


def scan(fc, fn, mc, mn, names=None, test=True):
    keep = (fc >= MIN_CASES) & (mc >= MIN_CASES)
    fc, fn, mc, mn = fc[keep], fn[keep], mc[keep], mn[keep]
    y, v = lib.woolf(fc, fn, mc, mn)
    het = lib.reml_tau2(y, v)
    mu, tau = het["mu"], math.sqrt(het["tau2"])
    pse = math.sqrt(het["tau2"] + het["se_mu"] ** 2)
    out = {"k": int(keep.sum()), "pooled_or": math.exp(mu), "pooled_lo": math.exp(mu - Z * het["se_mu"]),
           "pooled_hi": math.exp(mu + Z * het["se_mu"]), "tau": tau,
           "pred_lo": math.exp(mu - Z * pse), "pred_hi": math.exp(mu + Z * pse)}
    if test:
        lrt = np.array([A.lrt_fixed_or(a, b, c, d, math.exp(mu)) for a, b, c, d in zip(fc, fn, mc, mn)])
        q = lib.bh_fdr(chi2.sf(lrt, 1))
        flag = q < 0.05
        out.update({"flagged": int(flag.sum()), "flagged_female_shifted": int((flag & (y > mu)).sum()),
                    "flagged_male_shifted": int((flag & (y < mu)).sum())})
        if names is not None:
            out["table"] = pd.DataFrame({"stratum": np.asarray(names)[keep], "f_case": fc, "f_non": fn,
                                         "m_case": mc, "m_non": mn, "log_or": y, "var": v, "q_lrt": q})
    return out


def main(adir, rdir, B=200):
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                          columns=["primaryid", "sex", "female", "overdose", "narrow", "broad", "death",
                                   "pheno", "ps_drug", "age_yr"])
    rep = rep[rep["sex"].notna() & ~rep["overdose"]].sort_values("primaryid").reset_index(drop=True)
    pid = rep["primaryid"].to_numpy()
    n = len(rep)
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    pr = pr[pr["primaryid"].isin(pid)].drop_duplicates()
    prow = np.searchsorted(pid, pr["primaryid"].to_numpy())
    icode, inames = pd.factorize(pr["ingredient"].astype(str), sort=True)
    del pr
    nsusp = np.bincount(prow, minlength=n)
    le20 = nsusp <= 20
    female = rep["female"].to_numpy().astype(np.int64)
    narrow = rep["narrow"].to_numpy().astype(np.int64)
    death = rep["death"].to_numpy().astype(np.int64)
    hep = narrow.astype(bool)
    notbroad = ~rep["broad"].to_numpy() & ~hep
    phs = rep["pheno"].astype(str)
    ph = phs.isin(["HC", "CH"]).to_numpy()
    ch = (phs == "CH").to_numpy().astype(np.int64)
    dcode, dnames = pd.factorize(rep["ps_drug"].astype(str), sort=True)
    nd = len(dnames)
    lk = pd.read_parquet(os.path.join(adir, "link.parquet")).set_index("primaryid")
    grp = pd.Series(pid).map(lk["group_j80"]).fillna(-1).to_numpy().astype(np.int64)
    drop = pd.Series(pid).map(lk["drop_j80"]).fillna(False).to_numpy().astype(bool)
    chk = pd.Series(pid).map(lk["check_j80"]).fillna("").to_numpy().astype(str)
    linked = grp >= 0
    gsize = np.ones(n)
    gs = pd.Series(grp[linked]).map(pd.Series(grp[linked]).value_counts()).to_numpy()
    gsize[linked] = gs
    age_known_group = np.zeros(n, dtype=bool)
    age_known_group[linked] = rep["age_yr"].notna().to_numpy()[linked]
    res = {"bootstrap_replicates": B, "seed": SEED}

    def mh(code, f, y, w=None):
        cnt = np.bincount(code * 4 + f * 2 + y, weights=w, minlength=nd * 4).reshape(-1, 4).astype(float)
        return lib.mantel_haenszel(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])[:2]

    def pooled(mask, w=None):
        """Log estimates and model variances: reporting, phenotype, fatality ratio (all by drug)."""
        m = mask
        ww = None if w is None else w[m]
        y1, v1 = mh(dcode[m], female[m], narrow[m], ww)
        mp = m & ph
        y2, v2 = mh(dcode[mp], female[mp], ch[mp], None if w is None else w[mp])
        mh_ = m & hep
        mo = m & notbroad & np.isin(dcode, np.unique(dcode[mh_]))
        yh, vh = mh(dcode[mh_], female[mh_], death[mh_], None if w is None else w[mh_])
        yo, vo = mh(dcode[mo], female[mo], death[mo], None if w is None else w[mo])
        return {"reporting": (y1, v1), "phenotype": (y2, v2), "fatality_ratio": (yh - yo, vh + vo)}

    def pair_scan(mask=None, w=None, test=True):
        """Scan over primary and secondary suspects."""
        rows = np.ones(len(prow), dtype=bool) if mask is None else mask[prow]
        ww = None if w is None else w[prow][rows]
        cnt = np.bincount(icode[rows] * 4 + female[prow][rows] * 2 + narrow[prow][rows], weights=ww,
                          minlength=len(inames) * 4).reshape(-1, 4).astype(float)
        return scan(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0], test=test)

    # ---- A. stratum-level scan, restricted set
    cnt = np.bincount(dcode[le20] * 4 + female[le20] * 2 + narrow[le20], minlength=nd * 4).reshape(-1, 4).astype(float)
    sc = scan(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0], names=dnames)
    tab = sc.pop("table")
    tab.to_csv(os.path.join(rdir, "stratum_scan.csv"), index=False)
    ing = pd.read_csv(os.path.join(rdir, "le20_ps", "n1_ingredients.csv"))
    ing_flag = set(ing.loc[ing["q_lrt"] < 0.05, "ingredient"].astype(str))
    ing_all = set(ing["ingredient"].astype(str))
    st_flag = set(tab.loc[tab["q_lrt"] < 0.05, "stratum"].astype(str))
    st_all = set(tab["stratum"].astype(str))
    sc.update({"strata_also_in_ingredient_scan": len(st_all & ing_all),
               "flagged_in_both": len(st_flag & ing_flag),
               "flagged_stratum_only_among_shared": len((st_flag - ing_flag) & ing_all),
               "flagged_ingredient_only_among_shared": len((ing_flag - st_flag) & st_all),
               "ingredient_flags_without_own_stratum": len(ing_flag - st_all)})
    res["stratum_scan"] = sc

    # ---- B. cluster bootstrap
    rng = np.random.default_rng(SEED)
    cl = grp.copy()
    cl[~linked] = grp.max() + 1 + np.arange((~linked).sum())
    _, cidx = np.unique(cl, return_inverse=True)
    ncl = cidx.max() + 1
    res["clusters"] = {"reports": int(n), "sampling_units": int(ncl), "reports_in_groups": int(linked.sum())}
    sets = {"full": np.ones(n, dtype=bool), "restricted": le20, "linked": ~drop}
    model = {k: pooled(m) for k, m in sets.items()}
    boot = {k: {c: [] for c in ("reporting", "phenotype", "fatality_ratio")} for k in ("full", "restricted")}
    for _ in range(B):
        w = rng.poisson(1.0, ncl).astype(float)[cidx]
        for k in ("full", "restricted"):
            est = pooled(sets[k], w)
            for c in est:
                boot[k][c].append(est[c][0])
    res["cluster_bootstrap"] = {}
    for k in ("full", "restricted"):
        res["cluster_bootstrap"][k] = {}
        for c in boot[k]:
            y, v = model[k][c]
            sd = float(np.std(boot[k][c], ddof=1))
            res["cluster_bootstrap"][k][c] = {
                "or": math.exp(y), "model_se": math.sqrt(v), "cluster_se": sd, "se_ratio": sd / math.sqrt(v),
                "model_lo": math.exp(y - Z * math.sqrt(v)), "model_hi": math.exp(y + Z * math.sqrt(v)),
                "cluster_lo": math.exp(y - Z * sd), "cluster_hi": math.exp(y + Z * sd)}

    # ---- C. other ways of handling the linked groups (full set)
    variants = {
        "latest_kept": (~drop, None),
        "groups_passing_checks_only": (~(drop & (chk == "pass")), None),
        "groups_with_recorded_age_only": (~(drop & age_known_group), None),
        "inverse_group_size_weights": (np.ones(n, dtype=bool), 1.0 / gsize),
    }
    res["handling"] = {}
    for name, (m, w) in [("no_handling", (np.ones(n, dtype=bool), None))] + list(variants.items()):
        est = pooled(m, w)
        y, v = est["fatality_ratio"]
        ps = pair_scan(m if w is None else None, w, test=w is None)
        d = {"reports_removed": int(n - m.sum()), "effective_reports": float(m.sum() if w is None else w.sum()),
             "fatality_ratio": math.exp(y), "scan_pooled_or": ps["pooled_or"], "scan_tau": ps["tau"], "scan_k": ps["k"]}
        if w is None:
            d.update({"fatality_lo": math.exp(y - Z * math.sqrt(v)), "fatality_hi": math.exp(y + Z * math.sqrt(v)),
                      "scan_flagged": ps["flagged"]})
        res["handling"][name] = d

    # ---- D. joint model of the sex by hepatic interaction on death
    res["joint_fatality_model"] = {}
    for k in ("full", "restricted", "linked"):
        m = sets[k] & (hep | notbroad)
        g = pd.DataFrame({"d": dcode[m], "f": female[m], "h": hep[m].astype(int), "y": death[m]})
        t = g.groupby(["d", "f", "h"])["y"].agg(["sum", "size"]).reset_index()
        t["alive"] = t["size"] - t["sum"]
        ok = t.groupby("d").filter(lambda x: len(x) == 4 and (x["sum"] > 0).all() and (x["alive"] > 0).all())
        dd, _ = pd.factorize(ok["d"], sort=True)
        K = dd.max() + 1
        X = np.zeros((len(ok), 3 * K + 1))
        r = np.arange(len(ok))
        X[r, dd] = 1.0
        X[r, K + dd] = ok["f"].to_numpy()
        X[r, 2 * K + dd] = ok["h"].to_numpy()
        X[:, -1] = ok["f"].to_numpy() * ok["h"].to_numpy()
        fit = sm.GLM(ok[["sum", "alive"]].to_numpy(), X, family=sm.families.Binomial()).fit()
        b, se = float(fit.params[-1]), float(fit.bse[-1])
        res["joint_fatality_model"][k] = {
            "drugs": int(K), "hepatic_reports": int(ok.loc[ok["h"] == 1, "size"].sum()),
            "or": math.exp(b), "lo": math.exp(b - Z * se), "hi": math.exp(b + Z * se)}
    with open(os.path.join(rdir, "dependence.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("dependence:", json.dumps({k: res[k] for k in ("stratum_scan", "joint_fatality_model")})[:600])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
