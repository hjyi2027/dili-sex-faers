"""Run analyses N1 to N4 from the count cubes.

Usage: python3 05_analyze.py <cube_dir> <result_dir> [variant]
Variants: primary (default), hp, us, broad, with_overdose, dedup2, ps_only, no_top,
and the post hoc data-quality variants le20, le10, le5 (reports with at most that many
suspect ingredients), no_ca (reporter country not Canada), le20_ps.
Exposure is always female sex (reference: male).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

ADULT = ["18-49", "50-64", "65+"]
COVS = ["age_grp", "period", "region", "reporter"]
MIN_CASES = 20        # per sex, per ingredient (PROTOCOL N1)
MIN_CELL_AGE = 10     # per sex-by-age cell, per ingredient (Amendment 6)
ANCHORS_CH = ["CLAVULANATE", "FLUCLOXACILLIN", "CHLORPROMAZINE", "ERYTHROMYCIN"]
ANCHORS_HC = ["ISONIAZID", "DICLOFENAC", "VALPROATE", "MINOCYCLINE"]
Z = 1.959964


def est(log_or, var):
    if not np.isfinite(log_or) or not np.isfinite(var):
        return {"or": None, "lo": None, "hi": None, "log_or": None, "se": None}
    se = math.sqrt(var)
    return {"or": math.exp(log_or), "lo": math.exp(log_or - Z * se),
            "hi": math.exp(log_or + Z * se), "log_or": log_or, "se": se}


def table(df, outcome, strata=None):
    """Return a,b,c,d (female case, female non-case, male case, male non-case)."""
    keys = (strata or []) + ["female", outcome]
    g = df.groupby(keys, observed=True)["n"].sum()
    if strata:
        g = g.unstack(["female", outcome], fill_value=0)
        def col(f, o):
            return g[(f, o)].to_numpy(float) if (f, o) in g.columns else np.zeros(len(g))
        return col(True, True), col(True, False), col(False, True), col(False, False), g.index
    def cell(f, o):
        return float(g.get((f, o), 0))
    return cell(True, True), cell(True, False), cell(False, True), cell(False, False), None


def crude(df, outcome):
    a, b, c, d, _ = table(df, outcome)
    y, v = lib.woolf(a, b, c, d)
    out = est(float(y), float(v))
    out.update({"female_cases": a, "female_n": a + b, "male_cases": c, "male_n": c + d})
    return out


def mh(df, outcome, strata):
    a, b, c, d, _ = table(df, outcome, strata)
    y, v, k = lib.mantel_haenszel(a, b, c, d)
    out = est(y, v)
    out["informative_strata"] = k
    return out


def ratio(e1, e2):
    """Ratio of two independent odds ratios (e1 over e2)."""
    return est(e1["log_or"] - e2["log_or"], e1["se"] ** 2 + e2["se"] ** 2)


def lrt_fixed_or(a, b, c, d, psi):
    """Likelihood-ratio statistic for H0: OR = psi in a 2x2 table (1 df)."""
    r1, r2, c1 = a + b, c + d, a + c
    if abs(psi - 1) < 1e-12:
        ea = r1 * c1 / (r1 + r2)
    else:
        A = 1 - psi
        B = r2 - c1 + psi * (r1 + c1)
        C = -psi * r1 * c1
        disc = math.sqrt(B * B - 4 * A * C)
        roots = [(-B + disc) / (2 * A), (-B - disc) / (2 * A)]
        lo, hi = max(0.0, c1 - r2), min(r1, c1)
        ea = [r for r in roots if lo - 1e-9 <= r <= hi + 1e-9][0]
    E = [ea, r1 - ea, c1 - ea, r2 - c1 + ea]
    O = [a, b, c, d]
    g2 = 2 * sum(o * math.log(o / e) for o, e in zip(O, E) if o > 0)
    return max(g2, 0.0)


def select(rc, pc, variant):
    case = "narrow"
    if variant != "with_overdose":
        rc, pc = rc[~rc["overdose"]], pc[~pc["overdose"]]
    if variant == "hp":
        rc, pc = rc[rc["reporter"] == "HP"], pc[pc["reporter"] == "HP"]
    elif variant == "us":
        rc, pc = rc[rc["region"] == "US"], pc[pc["region"] == "US"]
    elif variant == "broad":
        case = "broad"
    elif variant == "dedup2":
        rc, pc = rc[~rc["dup2"]], pc[~pc["dup2"]]
    elif variant == "ps_only":
        pc = pc[pc["is_ps"]]
    elif variant in ("le20", "le10", "le5"):
        drop = {"le20": ["21-50", "51+"], "le10": ["11-20", "21-50", "51+"],
                "le5": ["6-10", "11-20", "21-50", "51+"]}[variant]
        rc, pc = rc[~rc["nsusp_band"].isin(drop)], pc[~pc["nsusp_band"].isin(drop)]
    elif variant == "no_ca":
        rc, pc = rc[~rc["ca"]], pc[~pc["ca"]]
    elif variant == "le20_hp":
        keep = ~rc["nsusp_band"].isin(["21-50", "51+"]) & (rc["reporter"] == "HP")
        keepp = ~pc["nsusp_band"].isin(["21-50", "51+"]) & (pc["reporter"] == "HP")
        rc, pc = rc[keep], pc[keepp]
    elif variant == "le20_ps":
        rc, pc = rc[~rc["nsusp_band"].isin(["21-50", "51+"])], pc[~pc["nsusp_band"].isin(["21-50", "51+"])]
        pc = pc[pc["is_ps"]]
    elif variant in ("linked", "linked_ps"):
        rc, pc = rc[~rc["link_drop"]], pc[~pc["link_drop"]]
        if variant == "linked_ps":
            pc = pc[pc["is_ps"]]
    elif variant == "linked9":
        rc, pc = rc[~rc["link_drop9"]], pc[~pc["link_drop9"]]
    elif variant == "linked_first":
        rc, pc = rc[~rc["link_dropf"]], pc[~pc["link_dropf"]]
    elif variant == "no_top":
        top = rc[rc[case]].groupby("ps_drug")["n"].sum().idxmax()
        rc = rc[rc["ps_drug"] != top]
        pc = pc[pc["ingredient"].map(lambda s: s not in top.split("\\"))]
    return rc.copy(), pc.copy(), case


# ---------------------------------------------------------------- N1
def n1(rc, pc, case, out):
    res = {}
    res["sex_set"] = {"crude": crude(rc, case), "mh_ps_drug": mh(rc, case, ["ps_drug"])}
    ad = rc[rc["age_grp"].isin(ADULT)]
    res["adult_set"] = {
        "crude": crude(ad, case),
        "mh_ps_drug": mh(ad, case, ["ps_drug"]),
        "mh_ps_drug_age": mh(ad, case, ["ps_drug", "age_grp"]),
        "mh_full": mh(ad, case, ["ps_drug"] + COVS),
    }
    res["by_age"] = {g: mh(ad[ad["age_grp"] == g], case, ["ps_drug", "period", "region", "reporter"])
                     for g in ADULT}
    # post hoc: does the fully stratified estimate differ by region, reporter type or period
    em = {}
    for col, levels in (("region", ["US", "nonUS"]), ("reporter", ["HP", "consumer"]),
                        ("period", ["2014-2017", "2018-2021", "2022+"])):
        for lv in levels:
            sub = ad[ad[col] == lv]
            em[f"{col}={lv}"] = {"full": mh(sub, case, ["ps_drug"] + COVS), "crude": crude(sub, case),
                                 "n_cases": int(sub.loc[sub[case], "n"].sum())}
    res["posthoc_effect_modification"] = em
    young = mh(ad[ad["age_grp"] == "18-49"], case, ["ps_drug", "period", "region", "reporter"])
    old = mh(ad[ad["age_grp"] != "18-49"], case, ["ps_drug", "age_grp", "period", "region", "reporter"])
    res["age50"] = {"under50": young, "50plus": old, "ratio_under50_over_50plus": ratio(young, old)}

    # per ingredient
    a, b, c, d, idx = table(pc, case, ["ingredient"])
    t = pd.DataFrame({"ingredient": idx, "f_case": a, "f_non": b, "m_case": c, "m_non": d})
    t = t[(t.f_case >= MIN_CASES) & (t.m_case >= MIN_CASES)].reset_index(drop=True)
    y, v = lib.woolf(t.f_case, t.f_non, t.m_case, t.m_non)
    t["log_or"], t["var"] = y, v
    het = lib.reml_tau2(y, v)
    psi = math.exp(het["mu"])
    t["lrt"] = [lrt_fixed_or(*r, psi) for r in t[["f_case", "f_non", "m_case", "m_non"]].to_numpy(float)]
    t["p_lrt"] = chi2.sf(t["lrt"], 1)
    t["q_lrt"] = lib.bh_fdr(t["p_lrt"])
    t["p_wald"] = lib.norm_p((t["log_or"] - het["mu"]) / np.sqrt(t["var"]))
    t["q_wald"] = lib.bh_fdr(t["p_wald"])
    # Test against OR = 1 as in Lu et al. 2023, to show how the null chosen changes the picture.
    t["lrt_one"] = [lrt_fixed_or(*r, 1.0) for r in t[["f_case", "f_non", "m_case", "m_non"]].to_numpy(float)]
    t["p_one"] = chi2.sf(t["lrt_one"], 1)
    t["q_one"] = lib.bh_fdr(t["p_one"])
    eb, ebv = lib.eb_shrink(y, v, het["mu"], het["tau2"])
    t["eb_log_or"], t["eb_var"] = eb, ebv
    t["direction"] = np.where(t["log_or"] > het["mu"], "female-shifted", "male-shifted")
    # adjusted (adult) per ingredient
    pad = pc[pc["age_grp"].isin(ADULT) & pc["ingredient"].isin(set(t["ingredient"]))]
    adj = {}
    for ing, g in pad.groupby("ingredient", observed=True):
        aa, bb, cc, dd, _ = table(g, case, COVS)
        adj[ing] = lib.mantel_haenszel(aa, bb, cc, dd)[:2]
    t["adj_log_or"] = t["ingredient"].map(lambda i: adj.get(i, (np.nan, np.nan))[0])
    t["adj_var"] = t["ingredient"].map(lambda i: adj.get(i, (np.nan, np.nan))[1])
    # age interaction per ingredient
    pa = pad.assign(young=pad["age_grp"] == "18-49")
    rows = []
    for ing, g in pa.groupby("ingredient", observed=True):
        cells = {}
        for yg in (True, False):
            aa, bb, cc, dd, _ = table(g[g["young"] == yg], case)
            cells[yg] = (aa, bb, cc, dd)
        if min(cells[True][0], cells[True][2], cells[False][0], cells[False][2]) < MIN_CELL_AGE:
            continue
        (y1, v1), (y0, v0) = lib.woolf(*cells[True]), lib.woolf(*cells[False])
        rows.append({"ingredient": ing, "log_or_under50": float(y1), "var_under50": float(v1),
                     "log_or_50plus": float(y0), "var_50plus": float(v0),
                     "f_case_u50": cells[True][0], "m_case_u50": cells[True][2],
                     "f_case_50p": cells[False][0], "m_case_50p": cells[False][2]})
    ta = pd.DataFrame(rows)
    if len(ta):
        ta["diff"] = ta["log_or_under50"] - ta["log_or_50plus"]
        ta["var_diff"] = ta["var_under50"] + ta["var_50plus"]
        hetd = lib.reml_tau2(ta["diff"], ta["var_diff"])
        ta["p"] = lib.norm_p((ta["diff"] - hetd["mu"]) / np.sqrt(ta["var_diff"]))
        ta["q"] = lib.bh_fdr(ta["p"])
        ta["p_zero"] = lib.norm_p(ta["diff"] / np.sqrt(ta["var_diff"]))
        ta["q_zero"] = lib.bh_fdr(ta["p_zero"])
        res["age_interaction_ingredients"] = {
            "k": int(len(ta)), **hetd,
            "n_q05_vs_pooled": int((ta["q"] < 0.05).sum()),
            "n_q05_vs_zero": int((ta["q_zero"] < 0.05).sum())}
        ta.sort_values("p_zero").to_csv(os.path.join(out, "n1_age_interaction.csv"), index=False)
    res["ingredients"] = {
        "k": int(len(t)), **het,
        "pooled_or": math.exp(het["mu"]),
        "pooled_lo": math.exp(het["mu"] - Z * het["se_mu"]),
        "pooled_hi": math.exp(het["mu"] + Z * het["se_mu"]),
        "pred_lo": math.exp(het["mu"] - Z * math.sqrt(het["tau2"] + het["se_mu"] ** 2)),
        "pred_hi": math.exp(het["mu"] + Z * math.sqrt(het["tau2"] + het["se_mu"] ** 2)),
        "n_q05_lrt": int((t["q_lrt"] < 0.05).sum()),
        "n_q05_lrt_female_shifted": int(((t["q_lrt"] < 0.05) & (t["direction"] == "female-shifted")).sum()),
        "n_q05_lrt_male_shifted": int(((t["q_lrt"] < 0.05) & (t["direction"] == "male-shifted")).sum()),
        "n_q05_wald": int((t["q_wald"] < 0.05).sum()),
        "n_q05_vs_one": int((t["q_one"] < 0.05).sum()),
        "n_q05_vs_one_or_below_1": int(((t["q_one"] < 0.05) & (t["log_or"] < 0)).sum()),
        "n_q05_vs_one_or_above_1": int(((t["q_one"] < 0.05) & (t["log_or"] > 0)).sum()),
        "n_or_below_1": int((t["log_or"] < 0).sum()),
        "n_q05_eb_dev_ge_1p5": int(((t["q_lrt"] < 0.05) & ((t["eb_log_or"] - het["mu"]).abs() >= math.log(1.5))).sum()),
        "n_q05_eb_dev_ge_2": int(((t["q_lrt"] < 0.05) & ((t["eb_log_or"] - het["mu"]).abs() >= math.log(2))).sum()),
    }
    t.sort_values("p_lrt").to_csv(os.path.join(out, "n1_ingredients.csv"), index=False)
    return res


# ---------------------------------------------------------------- N2
def boot_share(cells, B=2000, seed=20261123, chunk=100):
    """Bootstrap the share of the crude log OR removed by stratification.

    cells: array (S, 4) of counts a,b,c,d per stratum. Reports are resampled
    with replacement, which is a multinomial draw over the cells.
    """
    rng = np.random.default_rng(seed)
    N = int(cells.sum())
    p = (cells / N).ravel()
    shares, crude_l, mh_l = [], [], []
    for _ in range(B // chunk):
        draw = rng.multinomial(N, p, size=chunk).reshape(chunk, -1, 4).astype(float)
        a, b, c, d = (draw[:, :, i] for i in range(4))
        tot = draw.sum(axis=1)
        cl = np.log(tot[:, 0] * tot[:, 3] / (tot[:, 1] * tot[:, 2]))
        n = a + b + c + d
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(n > 0, a * d / n, 0).sum(axis=1)
            S = np.where(n > 0, b * c / n, 0).sum(axis=1)
        ml = np.log(R / S)
        shares.extend(1 - ml / cl)
        crude_l.extend(cl)
        mh_l.extend(ml)
    q = np.percentile(shares, [2.5, 97.5])
    return {"share_lo": float(q[0]), "share_hi": float(q[1]), "B": B,
            "crude_log_or_boot_sd": float(np.std(crude_l)), "mh_log_or_boot_sd": float(np.std(mh_l))}


def n2(rc, pc, case, out):
    res = {}
    hep = rc[rc[case]]
    ph = hep[hep["pheno"].isin(["HC", "CH"])].assign(ch=lambda x: x["pheno"] == "CH")
    counts = hep.groupby(["female", "pheno"], observed=True)["n"].sum().unstack(fill_value=0)
    res["pheno_counts_by_sex"] = {("female" if f else "male"): {k: int(v) for k, v in row.items()}
                                  for f, row in counts.iterrows()}
    hep2 = hep.assign(coded=hep["pheno"] != "")
    res["phenotype_coded_among_cases"] = {"crude": crude(hep2, "coded"),
                                          "mh_ps_drug": mh(hep2, "coded", ["ps_drug"])}
    res["sex_set"] = {"crude": crude(ph, "ch"), "mh_ps_drug": mh(ph, "ch", ["ps_drug"])}
    c_, m_ = res["sex_set"]["crude"], res["sex_set"]["mh_ps_drug"]
    a, b, c, d, _ = table(ph, "ch", ["ps_drug"])
    res["sex_set"]["share_removed_by_drug"] = 1 - m_["log_or"] / c_["log_or"]
    res["sex_set"]["share_boot"] = boot_share(np.column_stack([a, b, c, d]))
    ad = ph[ph["age_grp"].isin(ADULT)]
    res["adult_set"] = {
        "crude": crude(ad, "ch"),
        "mh_age": mh(ad, "ch", ["age_grp"]),
        "mh_ps_drug": mh(ad, "ch", ["ps_drug"]),
        "mh_ps_drug_age": mh(ad, "ch", ["ps_drug", "age_grp"]),
        "mh_full": mh(ad, "ch", ["ps_drug"] + COVS),
    }
    ca, ma = res["adult_set"]["crude"], res["adult_set"]["mh_ps_drug_age"]
    a, b, c, d, _ = table(ad, "ch", ["ps_drug", "age_grp"])
    res["adult_set"]["share_removed_by_drug_age"] = 1 - ma["log_or"] / ca["log_or"]
    res["adult_set"]["share_boot"] = boot_share(np.column_stack([a, b, c, d]))
    res["adult_set"]["share_removed_by_drug_only"] = 1 - res["adult_set"]["mh_ps_drug"]["log_or"] / ca["log_or"]
    res["adult_set"]["share_removed_by_age_only"] = 1 - res["adult_set"]["mh_age"]["log_or"] / ca["log_or"]
    # post hoc: sex odds ratio within each age band
    res["posthoc_by_age"] = {g: {"crude": crude(ad[ad["age_grp"] == g], "ch"),
                                 "mh_ps_drug": mh(ad[ad["age_grp"] == g], "ch", ["ps_drug"])}
                             for g in ADULT}
    for g in ADULT:
        res["posthoc_by_age"][g]["mh_full"] = mh(ad[ad["age_grp"] == g], "ch",
                                                 ["ps_drug", "period", "region", "reporter"])
    reg = ph.groupby("region", observed=True)["n"].sum()
    res["pheno_reports_by_region"] = {str(k): int(v) for k, v in reg.items()}
    yf = res["posthoc_by_age"]["18-49"]["mh_full"]
    of = mh(ad[ad["age_grp"] != "18-49"], "ch", ["ps_drug", "age_grp", "period", "region", "reporter"])
    res["posthoc_age50_full"] = {"under50": yf, "50plus": of, "ratio_50plus_over_under50": ratio(of, yf)}
    y_, o_ = res["posthoc_by_age"]["18-49"]["mh_ps_drug"], mh(ad[ad["age_grp"] != "18-49"], "ch", ["ps_drug", "age_grp"])
    res["posthoc_age50"] = {"under50": y_, "50plus": o_, "ratio_50plus_over_under50": ratio(o_, y_)}
    # post hoc: the same decomposition on the collapsible risk-difference scale
    def rdiff(df, strata):
        if strata:
            a, b, c, d, _ = table(df, "ch", strata)
        else:
            a, b, c, d, _ = table(df, "ch")
            a, b, c, d = (np.array([x]) for x in (a, b, c, d))
        r, v = lib.mh_risk_difference(a, b, c, d)
        return {"rd": r, "se": math.sqrt(v), "lo": r - Z * math.sqrt(v), "hi": r + Z * math.sqrt(v)}
    def standardized(df, strata, B=2000, seed=20261124):
        """Direct standardization of the cholestatic share to the combined (women plus men)
        distribution of the strata in which both sexes occur. Marginal, collapsible contrasts.
        The share removed is taken against the crude difference among ALL reports of the set,
        the same denominator as the Mantel-Haenszel shares."""
        a, b, c, d, _ = table(df, "ch", strata)
        cells = np.column_stack([a, b, c, d])

        def calc(a, b, c, d):
            nf, nm = a + b, c + d
            both = (nf > 0) & (nm > 0)
            w = np.where(both, nf + nm, 0.0)
            w = w / w.sum(axis=-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                pw = (w * np.where(both, a / np.where(nf > 0, nf, 1), 0)).sum(axis=-1)
                pm = (w * np.where(both, c / np.where(nm > 0, nm, 1), 0)).sum(axis=-1)
            cw, cm = a.sum(axis=-1) / nf.sum(axis=-1), c.sum(axis=-1) / nm.sum(axis=-1)
            return pw, pm, cw, cm, 1 - (pw - pm) / (cw - cm)
        pw, pm, cw, cm, srd = calc(*(cells[:, i] for i in range(4)))
        rng = np.random.default_rng(seed)
        N = int(cells.sum())
        pr = (cells / N).ravel()
        bs_rd, bs_d = [], []
        for _ in range(B // 100):
            dr = rng.multinomial(N, pr, size=100).reshape(100, -1, 4).astype(float)
            r = calc(*(dr[:, :, i] for i in range(4)))
            bs_rd.extend(r[4]); bs_d.extend(r[0] - r[1])
        q = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
        both = ((a + b) > 0) & ((c + d) > 0)
        return {"p_women_std": float(pw), "p_men_std": float(pm), "p_women_crude": float(cw),
                "p_men_crude": float(cm), "rd_std": float(pw - pm), "rd_std_ci": q(bs_d),
                "rd_crude": float(cw - cm),
                "or_std": float((pw / (1 - pw)) / (pm / (1 - pm))), "or_crude": float((cw / (1 - cw)) / (cm / (1 - cm))),
                "share_rd": float(srd), "share_rd_ci": q(bs_rd),
                "strata_used": int(both.sum()), "reports_used": int(cells[both].sum()), "reports_total": N}
    res["posthoc_standardized"] = {"sex_by_drug": standardized(ph, ["ps_drug"]),
                                   "adult_by_drug_age": standardized(ad, ["ps_drug", "age_grp"])}
    rdc, rdd = rdiff(ph, None), rdiff(ph, ["ps_drug"])
    rac, rad, rada = rdiff(ad, None), rdiff(ad, ["ps_drug"]), rdiff(ad, ["ps_drug", "age_grp"])
    res["posthoc_rd"] = {"sex_crude": rdc, "sex_by_drug": rdd, "adult_crude": rac, "adult_by_drug": rad,
                         "adult_by_drug_age": rada,
                         "share_removed_by_drug": 1 - rdd["rd"] / rdc["rd"],
                         "share_removed_by_drug_age_adult": 1 - rada["rd"] / rac["rd"]}
    # mixed pattern
    pm = hep[hep["pheno"].isin(["HC", "CH", "MX"])].assign(mx=lambda x: x["pheno"] == "MX")
    res["mixed"] = {"crude": crude(pm, "mx"), "mh_ps_drug": mh(pm, "mx", ["ps_drug"])}
    # validity check (i): cholestatic share by age band
    byage = ad.groupby(["age_grp", "ch"], observed=True)["n"].sum().unstack(fill_value=0)
    res["validity_age"] = {g: {"ch": int(byage.loc[g, True]), "hc": int(byage.loc[g, False]),
                               "ch_share": float(byage.loc[g, True] / byage.loc[g].sum())}
                           for g in ADULT if g in byage.index}
    bas = ad.groupby(["female", "age_grp", "ch"], observed=True)["n"].sum().unstack(fill_value=0)
    res["ch_share_by_age_sex"] = {
        ("female" if f else "male"): {g: float(bas.loc[(f, g), True] / bas.loc[(f, g)].sum())
                                      for g in ADULT if (f, g) in bas.index}
        for f in (True, False)}
    old_vs_young = ad[ad["age_grp"].isin(["18-49", "65+"])].assign(
        old=lambda x: x["age_grp"] == "65+")
    g = old_vs_young.rename(columns={"female": "sexf", "old": "female"})
    res["validity_age"]["or_65plus_vs_18_49_mh_ps_drug_sex"] = mh(g, "ch", ["ps_drug", "sexf"])
    # validity check (ii): anchor drugs
    pph = pc[pc[case] & pc["pheno"].isin(["HC", "CH"])].assign(ch=lambda x: x["pheno"] == "CH")
    overall = float(pph.loc[pph["ch"], "n"].sum() / pph["n"].sum())
    anch = {}
    for ing in ANCHORS_CH + ANCHORS_HC:
        g = pph[pph["ingredient"] == ing]
        nch, ntot = float(g.loc[g["ch"], "n"].sum()), float(g["n"].sum())
        if ntot > 0:
            ph_ = nch / ntot
            den = 1 + Z ** 2 / ntot
            centre = (ph_ + Z ** 2 / (2 * ntot)) / den
            half = Z * math.sqrt(ph_ * (1 - ph_) / ntot + Z ** 2 / (4 * ntot ** 2)) / den
            anch[ing] = {"expected": "CH" if ing in ANCHORS_CH else "HC", "ch": nch, "n": ntot,
                         "ch_share": ph_, "lo": centre - half, "hi": centre + half}
        else:
            anch[ing] = {"expected": "CH" if ing in ANCHORS_CH else "HC", "n": 0}
    res["validity_anchors"] = {"overall_ch_share_pairs": overall, "drugs": anch}
    pps = pph[pph["is_ps"]]
    anch_ps = {}
    for ing in ANCHORS_CH + ANCHORS_HC:
        g = pps[pps["ingredient"] == ing]
        nch, ntot = float(g.loc[g["ch"], "n"].sum()), float(g["n"].sum())
        anch_ps[ing] = {"expected": "CH" if ing in ANCHORS_CH else "HC", "ch": nch, "n": ntot,
                        "ch_share": (nch / ntot) if ntot else None}
    # Comparator: cholestatic share among all hepatocellular or cholestatic reports, counted
    # once per report. The first version used the share among primary-suspect ingredient
    # pairs, which counts a combination product once per ingredient (corrected 2026-11-24).
    res["validity_anchors_ps"] = {"overall_ch_share": float(ph.loc[ph["ch"], "n"].sum() / ph["n"].sum()),
                                  "overall_ch_share_ps_pairs": float(pps.loc[pps["ch"], "n"].sum() / pps["n"].sum()),
                                  "drugs": anch_ps}
    # per ingredient
    a, b, c, d, idx = table(pph, "ch", ["ingredient"])
    t = pd.DataFrame({"ingredient": idx, "f_ch": a, "f_hc": b, "m_ch": c, "m_hc": d})
    t = t[((t.f_ch + t.f_hc) >= MIN_CASES) & ((t.m_ch + t.m_hc) >= MIN_CASES)].reset_index(drop=True)
    y, v = lib.woolf(t.f_ch, t.f_hc, t.m_ch, t.m_hc)
    t["log_or"], t["var"] = y, v
    het = lib.reml_tau2(y, v)
    t["p_vs_pooled"] = lib.norm_p((t["log_or"] - het["mu"]) / np.sqrt(t["var"]))
    t["q_vs_pooled"] = lib.bh_fdr(t["p_vs_pooled"])
    t["ch_share"] = (t.f_ch + t.m_ch) / (t.f_ch + t.f_hc + t.m_ch + t.m_hc)
    t.sort_values("p_vs_pooled").to_csv(os.path.join(out, "n2_ingredients.csv"), index=False)
    res["ingredients"] = {"k": int(len(t)), **het, "pooled_or": math.exp(het["mu"]),
                          "pooled_lo": math.exp(het["mu"] - Z * het["se_mu"]),
                          "pooled_hi": math.exp(het["mu"] + Z * het["se_mu"]),
                          "n_q05_vs_pooled": int((t["q_vs_pooled"] < 0.05).sum())}
    return res


# ---------------------------------------------------------------- N3
def n3(rc, case, out):
    res = {}
    hep = rc[rc[case]]
    non = rc[~rc["broad"] & ~rc["narrow"]]
    for name, sub_h, sub_n, strata in [
        ("sex_set_ps_drug", hep, non, ["ps_drug"]),
        ("adult_full", hep[hep["age_grp"].isin(ADULT)], non[non["age_grp"].isin(ADULT)],
         ["ps_drug"] + COVS),
    ]:
        common = set(sub_h["ps_drug"].unique())
        sub_n = sub_n[sub_n["ps_drug"].isin(common)]
        h_c, n_c = crude(sub_h, "death"), crude(sub_n, "death")
        h_m, n_m = mh(sub_h, "death", strata), mh(sub_n, "death", strata)
        res[name] = {"hepatic_crude": h_c, "nonhepatic_crude": n_c,
                     "hepatic_mh": h_m, "nonhepatic_mh": n_m,
                     "calibrated_crude": ratio(h_c, n_c), "calibrated_mh": ratio(h_m, n_m)}
    def rd(df, strata):
        a, b, c, d, _ = table(df, "death", strata)
        r, v = lib.mh_risk_difference(a, b, c, d)
        return {"rd": r, "se": math.sqrt(v), "lo": r - Z * math.sqrt(v), "hi": r + Z * math.sqrt(v)}

    def prop(df):
        g = df.groupby(["female", "death"], observed=True)["n"].sum()
        return {("female" if f else "male"): float(g.get((f, True), 0) / (g.get((f, True), 0) + g.get((f, False), 0)))
                for f in (True, False)}
    common = set(hep["ps_drug"].unique())
    non_c = non[non["ps_drug"].isin(common)]
    rh, rn = rd(hep, ["ps_drug"]), rd(non_c, ["ps_drug"])
    dd, dse = rh["rd"] - rn["rd"], math.sqrt(rh["se"] ** 2 + rn["se"] ** 2)
    res["posthoc_risk_difference"] = {
        "fatal_share_hepatic": prop(hep), "fatal_share_other": prop(non_c),
        "rd_hepatic_mh_ps_drug": rh, "rd_other_mh_ps_drug": rn,
        "difference_of_rd": {"rd": dd, "se": dse, "lo": dd - Z * dse, "hi": dd + Z * dse}}
    def std_did(sub_h, sub_n, strata, standard):
        a, b, c, d, i1 = table(sub_h, "death", strata)
        H = pd.DataFrame({"a": a, "b": b, "c": c, "d": d}, index=i1)
        a, b, c, d, i2 = table(sub_n, "death", strata)
        O = pd.DataFrame({"a2": a, "b2": b, "c2": c, "d2": d}, index=i2)
        J = H.join(O, how="inner")
        J = J[((J.a + J.b) > 0) & ((J.c + J.d) > 0) & ((J.a2 + J.b2) > 0) & ((J.c2 + J.d2) > 0)]
        nfh, nmh, nfo, nmo = J.a + J.b, J.c + J.d, J.a2 + J.b2, J.c2 + J.d2
        w = (nfh + nmh) if standard == "hepatic" else (nfo + nmo)
        w = w / w.sum()
        pfh, pmh, pfo, pmo = J.a / nfh, J.c / nmh, J.a2 / nfo, J.c2 / nmo
        rdh, rdo = float((w * (pfh - pmh)).sum()), float((w * (pfo - pmo)).sum())
        var = float((w ** 2 * (pfh * (1 - pfh) / nfh + pmh * (1 - pmh) / nmh
                               + pfo * (1 - pfo) / nfo + pmo * (1 - pmo) / nmo)).sum())
        did = rdh - rdo
        return {"rd_hepatic": rdh, "rd_other": rdo,
                "did": {"rd": did, "se": math.sqrt(var), "lo": did - Z * math.sqrt(var), "hi": did + Z * math.sqrt(var)},
                "strata": int(len(J)), "hepatic_reports_covered": float((nfh + nmh).sum() / sub_h["n"].sum())}
    res["posthoc_common_standard"] = {
        "hepatic_standard": std_did(hep, non_c, ["ps_drug"], "hepatic"),
        "other_standard": std_did(hep, non_c, ["ps_drug"], "other")}
    res["severe_term"] = {"crude": crude(hep, "severe"), "mh_ps_drug": mh(hep, "severe", ["ps_drug"]),
                          "adult_mh_full": mh(hep[hep["age_grp"].isin(ADULT)], "severe",
                                              ["ps_drug"] + COVS)}
    bp = hep[hep["pheno"] != ""].groupby(["pheno", "female", "death"], observed=True)["n"].sum()
    res["death_by_pheno_sex"] = {f"{p}|{'F' if f else 'M'}|{'death' if dth else 'alive'}": int(n)
                                 for (p, f, dth), n in bp.items()}
    for p in ["HC", "CH", "MX"]:
        sub = hep[hep["pheno"] == p]
        if sub["n"].sum() > 0:
            res[f"death_sex_or_{p}"] = {"crude": crude(sub, "death"),
                                        "mh_ps_drug": mh(sub, "death", ["ps_drug"])}
    return res


# ---------------------------------------------------------------- N4
def n4(rc, case, out):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    ad = rc[rc["age_grp"].isin(ADULT)].copy()
    n_all_adults = int(ad["n"].sum())
    ad = ad[ad["region"] != "missing"]  # complete cases for region (0.1% of adult reports)
    dims = ["female", "age_grp", "ning_band", "dilirank", "reporter", "region", "period",
            "alcohol_proxy", "liverdx_proxy", "wt_cat"]
    g = ad.groupby(dims + [case], observed=True)["n"].sum().unstack(case, fill_value=0)
    g = g.rename(columns={True: "cases", False: "noncases"}).reset_index()
    for c in ("cases", "noncases"):
        if c not in g:
            g[c] = 0
    for c in ["alcohol_proxy", "liverdx_proxy", "female"]:
        g[c] = g[c].astype(int)
    base = ("C(age_grp, Treatment('18-49')) + C(ning_band, Treatment('1')) + "
            "C(dilirank, Treatment('no')) + C(reporter, Treatment('HP')) + "
            "C(region, Treatment('US')) + C(period, Treatment('2014-2017')) + "
            "alcohol_proxy + liverdx_proxy")
    res = {"n_adult_reports_before_region_exclusion": n_all_adults,
           "n_adult_reports": int(g[["cases", "noncases"]].to_numpy().sum()),
           "n_cases": int(g["cases"].sum())}
    # post hoc: sex odds ratio within drug, separately by DILIrank class of the primary suspect
    res["posthoc_sex_or_by_dilirank"] = {
        cl: mh(ad[ad["dilirank"] == cl], case, ["ps_drug"] + COVS)
        for cl in ["most", "less", "ambiguous", "no", "unlisted"]}

    def fit(data, rhs):
        gg = data.groupby([c for c in dims if c in rhs or c == "female"], observed=True)[
            ["cases", "noncases"]].sum().reset_index()
        m = smf.glm(f"cases + noncases ~ {rhs}", data=gg, family=sm.families.Binomial()).fit()
        return m

    tabs = []
    for label, sub in [("women", g[g.female == 1]), ("men", g[g.female == 0])]:
        m = fit(sub, base)
        ci = m.conf_int()
        for k in m.params.index:
            tabs.append({"model": label, "term": k, "or": math.exp(m.params[k]),
                         "lo": math.exp(ci.loc[k, 0]), "hi": math.exp(ci.loc[k, 1]), "p": m.pvalues[k]})
    mi = fit(g, f"female * ({base})")
    ci = mi.conf_int()
    for k in mi.params.index:
        if k.startswith("female:"):
            tabs.append({"model": "interaction", "term": k[7:], "or": math.exp(mi.params[k]),
                         "lo": math.exp(ci.loc[k, 0]), "hi": math.exp(ci.loc[k, 1]), "p": mi.pvalues[k]})
    # weight sub-model, complete cases
    gw = g[g["wt_cat"] != "missing"]
    res["n_weight_known"] = int(gw[["cases", "noncases"]].to_numpy().sum())
    basew = base + " + C(wt_cat, Treatment('T2'))"
    for label, sub in [("women_weight", gw[gw.female == 1]), ("men_weight", gw[gw.female == 0])]:
        m = fit(sub, basew)
        ci = m.conf_int()
        for k in m.params.index:
            if "wt_cat" in k:
                tabs.append({"model": label, "term": k, "or": math.exp(m.params[k]),
                             "lo": math.exp(ci.loc[k, 0]), "hi": math.exp(ci.loc[k, 1]), "p": m.pvalues[k]})
    mw = fit(gw, f"female * ({basew})")
    ci = mw.conf_int()
    for k in mw.params.index:
        if k.startswith("female:") and "wt_cat" in k:
            tabs.append({"model": "interaction_weight", "term": k[7:], "or": math.exp(mw.params[k]),
                         "lo": math.exp(ci.loc[k, 0]), "hi": math.exp(ci.loc[k, 1]), "p": mw.pvalues[k]})
    t = pd.DataFrame(tabs)
    inter = t["model"].isin(["interaction", "interaction_weight"])
    t.loc[inter, "q"] = lib.bh_fdr(t.loc[inter, "p"].to_numpy())
    t.to_csv(os.path.join(out, "n4_models.csv"), index=False)
    res["n_interactions_q05"] = int((t.loc[inter, "q"] < 0.05).sum())
    res["n_interactions_tested"] = int(inter.sum())
    return res


def descriptives(rc, case):
    d = {}
    tot = rc.groupby("female", observed=True)["n"].sum()
    d["reports_by_sex"] = {"female": int(tot.get(True, 0)), "male": int(tot.get(False, 0))}
    for col in ["age_grp", "period", "region", "reporter", "ning_band", "dilirank"]:
        for lab, sub in [("all", rc), ("cases", rc[rc[case]])]:
            g = sub.groupby(["female", col], observed=True)["n"].sum().unstack(fill_value=0)
            d[f"{col}|{lab}"] = {("female" if f else "male"): {str(k): int(v) for k, v in row.items()}
                                 for f, row in g.iterrows()}
    for col in [case, "death", "severe", "alcohol_proxy", "liverdx_proxy"]:
        for lab, sub in [("all", rc), ("cases", rc[rc[case]])]:
            g = sub[sub[col]].groupby("female", observed=True)["n"].sum()
            d[f"{col}|{lab}"] = {"female": int(g.get(True, 0)), "male": int(g.get(False, 0))}
    for lab, sub in [("all", rc), ("cases", rc[rc[case]])]:
        g = sub.groupby(["nsusp_band", "female"], observed=True)["n"].sum().unstack(fill_value=0)
        d[f"nsusp_band|{lab}"] = {str(k): {"female": int(row.get(True, 0)), "male": int(row.get(False, 0))}
                                 for k, row in g.iterrows()}
        g = sub.groupby(["ca", "female"], observed=True)["n"].sum().unstack(fill_value=0)
        d[f"canada|{lab}"] = {("CA" if k else "other"): {"female": int(row.get(True, 0)), "male": int(row.get(False, 0))}
                             for k, row in g.iterrows()}
    hi = rc[rc["nsusp_band"].isin(["21-50", "51+"])]
    d["over20_suspects"] = {"reports": int(hi["n"].sum()), "cases": int(hi.loc[hi[case], "n"].sum()),
                            "cases_female": int(hi.loc[hi[case] & hi["female"], "n"].sum()),
                            "cases_canada": int(hi.loc[hi[case] & hi["ca"], "n"].sum()),
                            "cases_fatal": int(hi.loc[hi[case] & hi["death"], "n"].sum())}
    top = rc[rc[case]].groupby("ps_drug", observed=True)["n"].sum().sort_values(ascending=False)
    d["top_ps_drugs_cases"] = {k: int(v) for k, v in top.head(30).items()}
    d["n_ps_strata_with_cases"] = int(len(top))
    return d


def main(cdir, rdir, variant="primary"):
    out = os.path.join(rdir, variant)
    os.makedirs(out, exist_ok=True)
    rc = pd.read_parquet(os.path.join(cdir, "report_cube.parquet"))
    pc = pd.read_parquet(os.path.join(cdir, "pair_cube.parquet"))
    rc, pc, case = select(rc, pc, variant)
    res = {"variant": variant, "case_definition": case,
           "n_reports": int(rc["n"].sum()), "n_cases": int(rc.loc[rc[case], "n"].sum())}
    res["descriptives"] = descriptives(rc, case)
    res["N1"] = n1(rc, pc, case, out)
    res["N2"] = n2(rc, pc, case, out)
    res["N3"] = n3(rc, case, out)
    if variant in ("primary", "le20", "hp", "le20_hp"):
        res["N4"] = n4(rc, case, out)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(variant, "done:", res["n_reports"], "reports,", res["n_cases"], "cases")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "primary")
