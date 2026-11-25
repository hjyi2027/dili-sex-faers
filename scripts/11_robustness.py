"""Robustness analyses added after an external review (PROTOCOL change 18). All post hoc.

Usage: python3 11_robustness.py <analysis_dir> <merged_dir> <results_dir>

Writes <results_dir>/robustness.json and <results_dir>/robustness_countries.csv.
Every analysis uses the restricted set: sex recorded, no overdose-context term, at most
20 distinct suspect ingredients. The reference rows repeat the main estimates so that this
script can be checked against 05_analyze.py (variant le20).
"""
import json
import math
import os
import sys
from importlib import import_module

import numpy as np
from scipy import stats
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

A = import_module("05_analyze")
crude, mh, ratio, table, ADULT, COVS, Z = A.crude, A.mh, A.ratio, A.table, A.ADULT, A.COVS, A.Z

# Cholestatic and hepatocellular terms that name a liver injury. The strict definition
# leaves out the bare terms "cholestasis", "jaundice cholestatic" and "jaundice hepatocellular".
STRICT_CH = {"HEPATITIS CHOLESTATIC", "CHOLESTATIC LIVER INJURY", "DRUG-INDUCED CHOLESTASIS",
             "IMMUNE-MEDIATED CHOLESTASIS", "VANISHING BILE DUCT SYNDROME"}
STRICT_HC = {"HEPATOCELLULAR INJURY", "HEPATIC CYTOLYSIS"}
TOP_COUNTRIES = 5


def std_did(sub_h, sub_n, strata, standard):
    """Difference in differences of fatal share on one common standard (as in 05_analyze.n3)."""
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
    did, se = rdh - rdo, math.sqrt(var)
    return {"rd_hepatic": rdh, "rd_other": rdo, "did": did, "se": se,
            "lo": did - Z * se, "hi": did + Z * se, "strata": int(len(J)),
            "hepatic_reports_covered": float((nfh + nmh).sum() / sub_h["n"].sum())}


def informative(df, outcome, strata):
    """Share of reports, and of reports with the outcome, in strata that inform the
    Mantel-Haenszel estimate (both sexes and both outcome levels present)."""
    a, b, c, d, _ = table(df, outcome, strata)
    keep = ((a + c) > 0) & ((b + d) > 0) & ((a + b) > 0) & ((c + d) > 0)
    tot, pos = (a + b + c + d).sum(), (a + c).sum()
    return {"strata": int(len(a)), "informative_strata": int(keep.sum()),
            "share_reports": float((a + b + c + d)[keep].sum() / tot),
            "share_outcome_reports": float((a + c)[keep].sum() / pos)}


def pheno_block(ph, extra=None):
    """Sex odds ratio of a cholestatic term among hepatocellular or cholestatic reports."""
    extra = extra or []
    ad = ph[ph["age_grp"].isin(ADULT)]
    young = ad[ad["age_grp"] == "18-49"]
    old = ad[ad["age_grp"] != "18-49"]
    y_ = mh(young, "ch", ["ps_drug"] + extra)
    o_ = mh(old, "ch", ["ps_drug", "age_grp"] + extra)
    f, m = ph[ph["female"]], ph[~ph["female"]]
    out = {
        "n": int(ph["n"].sum()), "n_female": int(f["n"].sum()), "n_adult": int(ad["n"].sum()),
        "ch_share_female": float(f.loc[f["ch"], "n"].sum() / max(f["n"].sum(), 1)),
        "ch_share_male": float(m.loc[m["ch"], "n"].sum() / max(m["n"].sum(), 1)),
        "fatal_share_ch": float(ph.loc[ph["ch"] & ph["death"], "n"].sum() / max(ph.loc[ph["ch"], "n"].sum(), 1)),
        "fatal_share_hc": float(ph.loc[~ph["ch"] & ph["death"], "n"].sum() / max(ph.loc[~ph["ch"], "n"].sum(), 1)),
        "crude": crude(ph, "ch"),
        "mh_country_only": mh(ph, "ch", ["country"]),
        "mh_ps_drug": mh(ph, "ch", ["ps_drug"] + extra),
        "adult_crude": crude(ad, "ch"),
        "adult_mh_ps_drug_age": mh(ad, "ch", ["ps_drug", "age_grp"] + extra),
        "under50": y_, "50plus": o_,
    }
    if y_["log_or"] is not None and o_["log_or"] is not None:
        out["ratio_50plus_over_under50"] = ratio(o_, y_)
    return out


def report_block(rc, extra=None):
    """Sex odds ratio of a hepatic injury report."""
    extra = extra or []
    ad = rc[rc["age_grp"].isin(ADULT)]
    return {"n": int(rc["n"].sum()), "cases": int(rc.loc[rc["narrow"], "n"].sum()),
            "crude": crude(rc, "narrow"),
            "mh_ps_drug": mh(rc, "narrow", ["ps_drug"] + extra),
            "adult_mh_full": mh(ad, "narrow", ["ps_drug"] + COVS + extra)}


def fatal_block(rc, extra=None):
    """Fatal outcome among hepatic reports against other reports of the same drugs."""
    extra = extra or []
    hep = rc[rc["narrow"]]
    non = rc[~rc["broad"] & ~rc["narrow"]]
    non = non[non["ps_drug"].isin(set(hep["ps_drug"].unique()))]
    h_m, n_m = mh(hep, "death", ["ps_drug"] + extra), mh(non, "death", ["ps_drug"] + extra)
    hf, hm = hep[hep["female"]], hep[~hep["female"]]
    out = {"hepatic_reports": int(hep["n"].sum()),
           "fatal_share_hepatic_female": float(hf.loc[hf["death"], "n"].sum() / hf["n"].sum()),
           "fatal_share_hepatic_male": float(hm.loc[hm["death"], "n"].sum() / hm["n"].sum()),
           "hepatic_mh": h_m, "other_mh": n_m, "ratio": ratio(h_m, n_m),
           "did_hepatic_standard": std_did(hep, non, ["ps_drug"] + extra, "hepatic"),
           "did_other_standard": std_did(hep, non, ["ps_drug"] + extra, "other")}
    ha, na = hep[hep["age_grp"].isin(ADULT)], non[non["age_grp"].isin(ADULT)]
    na = na[na["ps_drug"].isin(set(ha["ps_drug"].unique()))]
    out["adult_full_ratio"] = ratio(mh(ha, "death", ["ps_drug"] + COVS + extra),
                                    mh(na, "death", ["ps_drug"] + COVS + extra))
    return out


def fatal_ratio_simple(rc):
    """Fatality odds ratio ratio by drug only (no adult restriction)."""
    hep = rc[rc["narrow"]]
    non = rc[~rc["broad"] & ~rc["narrow"]]
    non = non[non["ps_drug"].isin(set(hep["ps_drug"].unique()))]
    h_m, n_m = mh(hep, "death", ["ps_drug"]), mh(non, "death", ["ps_drug"])
    out = ratio(h_m, n_m)
    out["hepatic_reports"] = int(hep["n"].sum())
    return out


def main(adir, mdir, rdir):
    cols = ["primaryid", "sex", "female", "age_yr", "period", "region", "country", "reporter",
            "ps_drug", "n_ing_all", "death", "narrow", "broad", "overdose", "pheno",
            "alcohol_proxy", "liverdx_proxy", "dilirank"]
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"), columns=cols)
    rep = rep[rep["sex"].notna() & ~rep["overdose"]].drop(columns=["sex", "overdose"])
    rep = rep.merge(pd.read_parquet(os.path.join(adir, "ps_indi.parquet")), on="primaryid", how="left")
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    ns = pr.groupby("primaryid", observed=True, sort=False)["ingredient"].nunique()
    del pr
    rep = rep[rep["primaryid"].map(ns).fillna(0) <= 20].copy()
    del ns
    age = rep["age_yr"]
    rep["age_grp"] = np.select(
        [age.isna(), age < 18, age < 50, age < 65], ["missing", "under18", "18-49", "50-64"],
        default="65+")
    rep["single"] = rep["n_ing_all"] == 1
    rep["has_compete"] = rep["compete"].astype(str) != ""
    rep["indi_known"] = rep["ps_indi"].astype(str) != ""
    for c in ["ps_drug", "ps_indi", "country", "period", "region", "reporter", "pheno", "dilirank"]:
        rep[c] = rep[c].astype(str)
    res = {"set": "restricted (at most 20 suspect ingredients)", "n_reports": int(len(rep)),
           "n_hepatic": int(rep["narrow"].sum())}
    res["indication"] = {
        "share_reports_with_ps_indication": float(rep["indi_known"].mean()),
        "share_hepatic_with_ps_indication": float(rep.loc[rep["narrow"], "indi_known"].mean()),
        "distinct_indications": int(rep.loc[rep["indi_known"], "ps_indi"].nunique())}
    res["single_drug"] = {"share_reports": float(rep["single"].mean()),
                          "share_hepatic": float(rep.loc[rep["narrow"], "single"].mean())}
    res["competing_cause"] = {
        "share_hepatic": float(rep.loc[rep["narrow"], "has_compete"].mean()),
        "share_other": float(rep.loc[~rep["narrow"], "has_compete"].mean()),
        "hepatic_by_group": {k: int(v) for k, v in
                             rep.loc[rep["narrow"] & rep["has_compete"], "compete"].astype(str)
                             .str.split("+").explode().value_counts().items()}}

    # ---------------- reporting and fatal outcome: count cube over all reports
    dims = ["ps_drug", "ps_indi", "age_grp", "period", "region", "reporter", "female", "narrow",
            "broad", "death", "single", "has_compete"]
    rc = rep.groupby(dims, observed=True, sort=False).size().rename("n").reset_index()
    dmap = rep.drop_duplicates("ps_drug").set_index("ps_drug")["dilirank"].astype(str)
    rc["dilirank"] = rc["ps_drug"].map(dmap).astype(str)
    res["reporting"] = {
        "reference": report_block(rc),
        "by_indication": report_block(rc, ["ps_indi"]),
        "indication_known_only": report_block(rc[rc["ps_indi"] != ""], ["ps_indi"]),
        "single_drug": report_block(rc[rc["single"]]),
        "single_drug_by_indication": report_block(rc[rc["single"]], ["ps_indi"]),
        "no_competing_cause": report_block(rc[~rc["has_compete"]]),
    }
    res["fatal"] = {
        "reference": fatal_block(rc),
        "by_indication": fatal_block(rc, ["ps_indi"]),
        "single_drug": fatal_block(rc[rc["single"]]),
        "no_competing_cause": fatal_block(rc[~rc["has_compete"]]),
    }
    # Heterogeneity across drugs of the ratio (OR of death by sex among hepatic reports over the
    # same OR among other reports), drugs with no empty cell in either table.
    hep, non = rc[rc["narrow"]], rc[~rc["broad"] & ~rc["narrow"]]
    a, b, c, d, i1 = table(hep, "death", ["ps_drug"])
    H = pd.DataFrame({"a": a, "b": b, "c": c, "d": d}, index=i1)
    a, b, c, d, i2 = table(non, "death", ["ps_drug"])
    O = pd.DataFrame({"a2": a, "b2": b, "c2": c, "d2": d}, index=i2)
    J = H.join(O, how="inner")
    J = J[(J > 0).all(axis=1)]
    y = np.log(J.a * J.d / (J.b * J.c)) - np.log(J.a2 * J.d2 / (J.b2 * J.c2))
    v = 1 / J.a + 1 / J.b + 1 / J.c + 1 / J.d + 1 / J.a2 + 1 / J.b2 + 1 / J.c2 + 1 / J.d2
    w = 1 / v
    mu = float((w * y).sum() / w.sum())
    q = float((w * (y - mu) ** 2).sum())
    se = float(math.sqrt(1 / w.sum()))
    het = lib.reml_tau2(y.to_numpy(), v.to_numpy())
    tau = math.sqrt(het["tau2"])
    pse = math.sqrt(het["tau2"] + het["se_mu"] ** 2)
    res["fatal_ratio_by_drug"] = {"drugs": int(len(J)), "pooled": {"or": math.exp(mu), "lo": math.exp(mu - Z * se),
                                  "hi": math.exp(mu + Z * se), "log_or": mu, "se": se},
                                  "q": q, "df": int(len(J) - 1),
                                  "i2": float(max(0.0, (q - (len(J) - 1)) / q)),
                                  "random_effects": {"or": math.exp(het["mu"]), "lo": math.exp(het["mu"] - Z * het["se_mu"]),
                                                     "hi": math.exp(het["mu"] + Z * het["se_mu"]),
                                                     "log_or": het["mu"], "se": het["se_mu"]},
                                  "tau": tau,
                                  "prediction_lo": math.exp(het["mu"] - Z * pse),
                                  "prediction_hi": math.exp(het["mu"] + Z * pse),
                                  "hepatic_reports_in_these_drugs": float((J.a + J.b + J.c + J.d).sum() / hep["n"].sum())}
    # Absolute contrast standardized to a common distribution of drug, age band and region (adults)
    ha, na = hep[hep["age_grp"].isin(ADULT)], non[non["age_grp"].isin(ADULT)]
    res["fatal_did_drug_age_region"] = {
        std: std_did(ha, na, ["ps_drug", "age_grp", "region"], std) for std in ("hepatic", "other")}
    res["fatal_did_drug_adults"] = {
        std: std_did(ha, na, ["ps_drug"], std) for std in ("hepatic", "other")}
    # Missing age
    am = rc["age_grp"] == "missing"
    def share(mask_num, mask_den):
        return float(rc.loc[mask_num & mask_den, "n"].sum() / rc.loc[mask_den, "n"].sum())
    allr = pd.Series(True, index=rc.index)
    res["age_missing"] = {
        "share_women": share(am, rc["female"]), "share_men": share(am, ~rc["female"]),
        "share_hepatic_women": share(am, rc["female"] & rc["narrow"]),
        "share_hepatic_men": share(am, ~rc["female"] & rc["narrow"]),
        "share_us": share(am, rc["region"] == "US"), "share_non_us": share(am, rc["region"] == "nonUS"),
        "share_health_professional": share(am, rc["reporter"] == "HP"),
        "share_consumer": share(am, rc["reporter"] == "consumer"),
        "share_all": share(am, allr),
        "reporting_by_drug_age_recorded": mh(rc[~am], "narrow", ["ps_drug"]),
        "reporting_by_drug_age_missing": mh(rc[am], "narrow", ["ps_drug"]),
        "reporting_full_with_missing_age_stratum": mh(rc, "narrow", ["ps_drug"] + COVS),
        "fatal_ratio_age_recorded": fatal_ratio_simple(rc[~am]),
        "fatal_ratio_age_missing": fatal_ratio_simple(rc[am]),
    }
    res["fatal_ratio_by_period"] = {per: fatal_ratio_simple(rc[rc["period"] == per])
                                    for per in ("2014-2017", "2018-2021", "2022+")}
    res["fatal_ratio_by_dilirank"] = {"most": fatal_ratio_simple(rc[rc["dilirank"] == "most"]),
                                      "not_most": fatal_ratio_simple(rc[rc["dilirank"] != "most"])}
    # Share of reports in informative strata
    ad_ = rc[rc["age_grp"].isin(ADULT)]
    res["informative"] = {
        "reporting_by_drug": informative(rc, "narrow", ["ps_drug"]),
        "reporting_full": informative(ad_, "narrow", ["ps_drug"] + COVS),
        "reporting_full_indication": informative(ad_, "narrow", ["ps_drug"] + COVS + ["ps_indi"]),
        "fatal_hepatic_by_drug": informative(hep, "death", ["ps_drug"]),
        "fatal_hepatic_full": informative(ha, "death", ["ps_drug"] + COVS),
    }
    del rc, hep, non, ad_, ha, na

    # ---------------- phenotype: report level
    ph = rep[rep["narrow"] & rep["pheno"].isin(["HC", "CH"])].copy()
    ph["ch"] = ph["pheno"] == "CH"
    ph["n"] = 1
    pt = pd.read_parquet(os.path.join(mdir, "ptrows.parquet"))
    pt = pt[pt["primaryid"].isin(ph["primaryid"])]
    pt["pt"] = pt["pt"].astype(str)
    s_ch = set(pt.loc[pt["pt"].isin(STRICT_CH), "primaryid"])
    s_hc = set(pt.loc[pt["pt"].isin(STRICT_HC), "primaryid"])
    ph["strict"] = np.where(ph["ch"], ph["primaryid"].isin(s_ch), ph["primaryid"].isin(s_hc))
    # Hepatocellular reports whose only hepatocellular term is "hepatic cytolysis". In French
    # usage the term often records a transaminase rise (third review round, referee point M1).
    has_inj = set(pt.loc[pt["pt"].isin({"HEPATOCELLULAR INJURY", "JAUNDICE HEPATOCELLULAR"}), "primaryid"])
    has_cyt = set(pt.loc[pt["pt"] == "HEPATIC CYTOLYSIS", "primaryid"])
    ph["cyto_only"] = ~ph["ch"] & ph["primaryid"].isin(has_cyt) & ~ph["primaryid"].isin(has_inj)
    ph["hc_injury_term"] = ph["primaryid"].isin(set(pt.loc[pt["pt"] == "HEPATOCELLULAR INJURY", "primaryid"]))
    ph["proxy"] = ph["alcohol_proxy"] | ph["liverdx_proxy"]
    P = {}
    P["reference"] = pheno_block(ph)
    P["strict_terms"] = pheno_block(ph[ph["strict"]])
    P["bare_cholestasis_terms"] = pheno_block(ph[~ph["ch"] | ~ph["strict"]])
    P["nonfatal"] = pheno_block(ph[~ph["death"]])
    P["no_competing_cause"] = pheno_block(ph[~ph["has_compete"]])
    P["strict_nonfatal_no_competing_cause"] = pheno_block(
        ph[ph["strict"] & ~ph["death"] & ~ph["has_compete"]])
    P["single_drug"] = pheno_block(ph[ph["single"]])
    P["health_professional"] = pheno_block(ph[ph["reporter"] == "HP"])
    P["by_indication"] = pheno_block(ph, ["ps_indi"])
    P["indication_known_only"] = pheno_block(ph[ph["ps_indi"] != ""], ["ps_indi"])
    P["by_country"] = pheno_block(ph, ["country"])
    nc = ph[~ph["cyto_only"]]
    P["no_cytolysis_only"] = pheno_block(nc)
    P["no_cytolysis_only_by_country"] = pheno_block(nc, ["country"])
    P["no_cytolysis_only_top_country"] = pheno_block(nc[nc["country"] == "FR"])
    P["no_cytolysis_only_other_countries"] = pheno_block(nc[nc["country"] != "FR"])
    P["hc_injury_term_only"] = pheno_block(ph[ph["ch"] | ph["hc_injury_term"]])
    P["no_alcohol_or_liver_disease_proxy"] = pheno_block(ph[~ph["proxy"]])
    P["dilirank_most_concern"] = pheno_block(ph[ph["dilirank"] == "most"])
    P["dilirank_most_or_less_concern"] = pheno_block(ph[ph["dilirank"].isin(["most", "less"])])
    P["dilirank_less_concern"] = pheno_block(ph[ph["dilirank"] == "less"])
    P["dilirank_not_most_concern"] = pheno_block(ph[ph["dilirank"] != "most"])
    for per in ("2014-2017", "2018-2021", "2022+"):
        P["period_" + per] = pheno_block(ph[ph["period"] == per])

    def het(ests):
        """Cochran Q across independent estimates."""
        y = np.array([e["log_or"] for e in ests]); w = 1 / np.array([e["se"] ** 2 for e in ests])
        q = float((w * (y - (w * y).sum() / w.sum()) ** 2).sum())
        return {"q": q, "df": len(ests) - 1, "p": float(stats.chi2.sf(q, len(ests) - 1))}
    res["phenotype_heterogeneity"] = {
        "period_by_drug": het([P["period_" + per]["mh_ps_drug"] for per in ("2014-2017", "2018-2021", "2022+")]),
        "period_adult_drug_age": het([P["period_" + per]["adult_mh_ps_drug_age"]
                                      for per in ("2014-2017", "2018-2021", "2022+")]),
        "dilirank_most_vs_rest_by_drug": het([P["dilirank_most_concern"]["mh_ps_drug"],
                                              P["dilirank_not_most_concern"]["mh_ps_drug"]]),
        "dilirank_most_vs_rest_adult_drug_age": het([P["dilirank_most_concern"]["adult_mh_ps_drug_age"],
                                                     P["dilirank_not_most_concern"]["adult_mh_ps_drug_age"]])}
    res["age_missing"].update({
        "phenotype_reports_age_missing": int(ph.loc[ph["age_grp"] == "missing", "n"].sum()),
        "phenotype_by_drug_age_missing": mh(ph[ph["age_grp"] == "missing"], "ch", ["ps_drug"]),
        "phenotype_by_drug_age_recorded": mh(ph[ph["age_grp"] != "missing"], "ch", ["ps_drug"])})
    pad = ph[ph["age_grp"].isin(ADULT)]
    res["informative"].update({
        "phenotype_by_drug": informative(ph, "ch", ["ps_drug"]),
        "phenotype_drug_age": informative(pad, "ch", ["ps_drug", "age_grp"]),
        "phenotype_drug_age_country": informative(pad, "ch", ["ps_drug", "age_grp", "country"]),
        "phenotype_drug_age_indication": informative(pad, "ch", ["ps_drug", "age_grp", "ps_indi"]),
    })
    hc = ph[~ph["ch"]]
    hca = hc[hc["age_grp"].isin(ADULT)]
    res["cytolysis_only"] = {
        "hc_reports": int(len(hc)), "cyto_only_reports": int(hc["cyto_only"].sum()),
        "share_of_hc": float(hc["cyto_only"].mean()),
        "share_of_hc_adults": float(hca["cyto_only"].mean()),
        "share_from_top_country": float((hc.loc[hc["cyto_only"], "country"] == "FR").mean()),
        "share_of_hc_female": float(hc.loc[hc["female"], "cyto_only"].mean()),
        "share_of_hc_male": float(hc.loc[~hc["female"], "cyto_only"].mean()),
        "sex_or_crude": crude(hc, "cyto_only"),
        "sex_or_adult_mh_ps_drug_age": mh(hca, "cyto_only", ["ps_drug", "age_grp"]),
        "sex_or_adult_mh_ps_drug_age_country": mh(hca, "cyto_only", ["ps_drug", "age_grp", "country"]),
    }
    ct = ph.groupby("country").agg(n=("n", "sum"), ch_share=("ch", "mean"),
                                   female_share=("female", "mean"),
                                   fatal_share=("death", "mean")).sort_values("n", ascending=False)
    ct["share_of_reports"] = ct["n"] / ct["n"].sum()
    ct.to_csv(os.path.join(rdir, "robustness_countries.csv"))
    top = [c for c in ct.index if c != ""][:TOP_COUNTRIES]
    res["phenotype_countries"] = {
        "n_countries": int((ct.index != "").sum()),
        "top": [{"country": c, "n": int(ct.loc[c, "n"]), "share": float(ct.loc[c, "share_of_reports"]),
                 "ch_share": float(ct.loc[c, "ch_share"]), "female_share": float(ct.loc[c, "female_share"])}
                for c in top]}
    P["leave_one_country_out"] = {c: pheno_block(ph[ph["country"] != c]) for c in top}
    P["within_country"] = {c: pheno_block(ph[ph["country"] == c]) for c in top}
    P["outside_top_country"] = pheno_block(ph[ph["country"] != top[0]])
    res["phenotype"] = P
    # Fatal share of cholestatic against hepatocellular reports. The exposure column of the
    # estimator functions is named "female", so the phenotype is passed under that name here.
    g = ph.rename(columns={"female": "sexf"}).assign(female=ph["ch"])
    t0 = top[0]
    res["fatal_by_phenotype"] = {
        "crude": crude(g, "death"),
        "mh_country": mh(g, "death", ["country"]),
        "mh_country_drug": mh(g, "death", ["country", "ps_drug"]),
        "mh_country_drug_no_competing_cause": mh(g[~g["has_compete"]], "death", ["country", "ps_drug"]),
        "adult_mh_country_drug_age": mh(g[g["age_grp"].isin(ADULT)], "death",
                                        ["country", "ps_drug", "age_grp"]),
        "top_country": t0,
        "top_country_fatal_share_ch": float(ph.loc[(ph["country"] == t0) & ph["ch"], "death"].mean()),
        "top_country_fatal_share_hc": float(ph.loc[(ph["country"] == t0) & ~ph["ch"], "death"].mean()),
        "other_countries_fatal_share_ch": float(ph.loc[(ph["country"] != t0) & ph["ch"], "death"].mean()),
        "other_countries_fatal_share_hc": float(ph.loc[(ph["country"] != t0) & ~ph["ch"], "death"].mean()),
        "hc_share_from_top_country": float((ph.loc[~ph["ch"], "country"] == t0).mean()),
        "ch_share_from_top_country": float((ph.loc[ph["ch"], "country"] == t0).mean()),
    }
    res["phenotype_term_counts"] = {
        "ch_strict": int((ph["ch"] & ph["strict"]).sum()), "ch_bare_only": int((ph["ch"] & ~ph["strict"]).sum()),
        "hc_strict": int((~ph["ch"] & ph["strict"]).sum()), "hc_bare_only": int((~ph["ch"] & ~ph["strict"]).sum())}
    # Anchor drugs against all other drugs, with reporter country as a stratum. The cholestatic
    # share differs several-fold between countries, so the unstratified comparison of an anchor
    # with the overall share mixes drug and country.
    ctry_share = ph.groupby("country")["ch"].mean()
    anch = {}
    for ing in A.ANCHORS_CH + A.ANCHORS_HC:
        is_a = ph["ps_drug"].map(lambda k: ing in k.split("\\"))
        sub = ph[is_a]
        d = {"expected": "CH" if ing in A.ANCHORS_CH else "HC", "n": int(len(sub))}
        if len(sub) >= 10:
            others = ph[~is_a].groupby("country")["ch"].mean()
            exp = sub["country"].map(others)
            exp = exp.where(exp.notna(), sub["country"].map(ctry_share))
            d["ch_share"] = float(sub["ch"].mean())
            d["expected_ch_share_given_country"] = float(exp.mean())
            ga = ph.rename(columns={"female": "sexf"}).assign(female=is_a)
            d["or_vs_other_drugs_crude"] = crude(ga, "ch")
            d["or_vs_other_drugs_mh_country"] = mh(ga, "ch", ["country"])
        anch[ing] = d
    res["anchors_by_country"] = anch
    with open(os.path.join(rdir, "robustness.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    ref = P["reference"]
    print("robustness written | phenotype reference by drug and age",
          round(ref["adult_mh_ps_drug_age"]["or"], 4), "| n", ref["n"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
