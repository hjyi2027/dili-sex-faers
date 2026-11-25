"""Hepatic definition of the paper against the FDA Office of New Drugs Custom Medical Queries (post hoc, change 31).

Usage: python3 20_ocmq_definition.py <data_dir> <analysis_dir> <ocmq_csv> <results_dir>

<ocmq_csv> is the sheet "Consolidated List" of the FDA workbook (version 4.1) with columns ocmq, term,
classification. The comparison definition of a hepatic report is any narrow term of the Hepatic Injury or
the Hepatic Failure query, as published, matched to the FAERS preferred term by name. Non-cases for the
fatality baseline carry no narrow or broad term of either query. The rule was fixed before any count by sex.

Writes <results_dir>/ocmq_definition.json and <results_dir>/ocmq_term_comparison.csv (no counts by sex).
"""
import glob
import json
import math
import os
import sys
import zipfile
from collections import Counter
from importlib import import_module

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

R = import_module("01_reduce")
D = import_module("14_dependence")
Z = 1.959963984540054
QUERIES = ("Hepatic Injury", "Hepatic Failure")
# Classes of narrow query terms outside the narrow list of the paper (change 32), by term name, in this order:
# laboratory and nonspecific terms of the broad list of the paper, competing causes by the rule of lib.py,
# chronic liver disease with portal hypertension, alcoholic and fatty liver disease, and the rest.
CHRONIC_RULE = (r"CIRRHOS|PORTAL HYPERTENS|PORTAL VEIN|PORTAL SHUNT|PORTAL FIBROSIS|PORTOPULMONARY|VARIC|FIBROSIS|STEATO|"
                r"FATTY LIVER|ALCOHOL|CHRONIC HEPAT|ACUTE ON CHRONIC|HEPATITIS CHRONIC|HEPATORENAL|HEPATOPULMONARY|"
                r"HEPATIC HYDROTHORAX|CHILD-PUGH|NODULAR REGENERATIVE|SHUNT|BILIARY ASCITES")
CLASSES = ("shared", "lab", "competing", "chronic", "other")


def term_class(t, my_narrow, my_broad):
    import re
    if t in my_narrow:
        return "shared"
    if t in my_broad:
        return "lab"
    if lib.competing_cause_group(t):
        return "competing"
    if re.search(CHRONIC_RULE, t):
        return "chronic"
    return "other"


def est(y, v):
    return {"or": float(np.exp(y)), "lo": float(np.exp(y - Z * np.sqrt(v))), "hi": float(np.exp(y + Z * np.sqrt(v)))}


def scan_reac(ddir, pid, sets, count_terms, rules=None):
    """One pass over the REAC files. sets: name -> set of terms. Returns flags per report and report counts per term."""
    n = len(pid)
    rules = rules or {}
    flags = {k: np.zeros(n, dtype=bool) for k in list(sets) + list(rules)}
    counts = Counter()
    for z in sorted(glob.glob(os.path.join(ddir, "faers_ascii_*.zip"))):
        zf = zipfile.ZipFile(z)
        reac, _ = R.read_table(zf, R.find_member(zf, "REAC"), ["primaryid", "pt"])
        reac["primaryid"] = R.to_int(reac["primaryid"])
        reac = reac.dropna(subset=["primaryid"])
        p = reac["primaryid"].astype("int64").to_numpy()
        idx = np.searchsorted(pid, p)
        idx[idx == n] = 0
        ok = pid[idx] == p
        idx = idx[ok]
        pt = pd.Series(reac["pt"].to_numpy()[ok]).str.upper().str.strip()
        for k, terms in sets.items():
            flags[k][idx[pt.isin(terms).to_numpy()]] = True
        u = pd.unique(pt)
        for k, fn in rules.items():
            flags[k][idx[pt.isin({t for t in u if fn(t)}).to_numpy()]] = True
        m = pt.isin(count_terms).to_numpy()
        pairs = pd.DataFrame({"i": idx[m], "pt": pt[m].to_numpy()}).drop_duplicates()
        counts.update(pairs["pt"].value_counts().to_dict())
    return flags, counts


def main(ddir, adir, ocmq_csv, rdir):
    o = pd.read_csv(ocmq_csv)
    o = o[o["ocmq"].isin(QUERIES)].copy()
    o["pt"] = o["term"].astype(str).str.upper().str.strip()
    inj_n = set(o.loc[(o["ocmq"] == "Hepatic Injury") & (o["classification"] == "Narrow"), "pt"])
    fail_n = set(o.loc[(o["ocmq"] == "Hepatic Failure") & (o["classification"] == "Narrow"), "pt"])
    any_t = set(o["pt"])
    terms = pd.read_csv(os.path.join(adir, "terms.csv"))
    my_narrow = set(terms.loc[terms["narrow"].astype(str) == "True", "pt"].str.upper())
    my_broad = set(terms.loc[terms["broad"].astype(str) == "True", "pt"].str.upper())

    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                          columns=["primaryid", "sex", "female", "overdose", "narrow", "broad", "death", "ps_drug",
                                   "age_band", "adult", "period", "region", "reporter", "pheno"])
    rep = rep[rep["sex"].notna() & ~rep["overdose"]].sort_values("primaryid").reset_index(drop=True)
    pid = rep["primaryid"].to_numpy()
    n = len(rep)
    allnar = inj_n | fail_n
    cls_terms = {c: {t for t in allnar if term_class(t, my_narrow, my_broad) == c} for c in CLASSES}
    sets = {"o_narrow": allnar, "inj_narrow": inj_n, "o_any": any_t}
    sets.update({"c_" + c: cls_terms[c] for c in CLASSES})
    flags, counts = scan_reac(ddir, pid, sets, any_t | my_narrow | my_broad, rules={"cc": lambda t: bool(lib.competing_cause_group(t))})
    o_narrow, inj_narrow, o_any = flags["o_narrow"], flags["inj_narrow"], flags["o_any"]

    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    pr = pr[pr["primaryid"].isin(pid)].drop_duplicates()
    nsusp = np.bincount(np.searchsorted(pid, pr["primaryid"].to_numpy()), minlength=n)
    del pr
    le20 = nsusp <= 20
    female = rep["female"].to_numpy().astype(np.int64)
    narrow = rep["narrow"].to_numpy().astype(bool)
    broad = rep["broad"].to_numpy().astype(bool)
    death = rep["death"].to_numpy().astype(np.int64)
    adult = rep["adult"].to_numpy().astype(bool)
    ph = rep["pheno"].astype(str).to_numpy()
    coded = np.isin(ph, ["HC", "CH"])
    chol = (ph == "CH").astype(np.int64)
    cat = rep["ps_drug"].astype(str).astype("category")
    dcode = cat.cat.codes.to_numpy().astype(np.int64)
    dnames = np.asarray(cat.cat.categories.astype(str))
    nd = len(dnames)
    ac, au = pd.factorize(rep["age_band"].astype(str))
    dacode = dcode * len(au) + ac
    fraw = dcode.copy()
    for col in ("age_band", "period", "region", "reporter"):
        c, u = pd.factorize(rep[col].astype(str))
        fraw = fraw * len(u) + c
    fcode = np.unique(fraw, return_inverse=True)[1].astype(np.int64)
    del fraw

    def mh(code, f, y):
        code = np.unique(code, return_inverse=True)[1]
        k = int(code.max()) + 1 if len(code) else 1
        cnt = np.bincount(code * 4 + f * 2 + y, minlength=k * 4).reshape(-1, 4).astype(float)
        return lib.mantel_haenszel(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])[:2]

    def contrasts(case, allm, other):
        y = case.astype(np.int64)
        f, c = female[allm], y[allm]
        out = {"reports": int(allm.sum()), "hepatic_reports": int((allm & case).sum()),
               "female_pct_of_hepatic": float(100 * female[allm & case].mean())}
        out["reporting_crude"] = est(*[float(v) for v in lib.woolf(((f == 1) & (c == 1)).sum(), ((f == 1) & (c == 0)).sum(),
                                                                    ((f == 0) & (c == 1)).sum(), ((f == 0) & (c == 0)).sum())])
        out["reporting_by_drug"] = est(*mh(dcode[allm], f, c))
        ma = allm & adult
        out["reporting_adult_full"] = est(*mh(fcode[ma], female[ma], y[ma]))
        mh_ = allm & case
        mo = other & np.isin(dcode, np.unique(dcode[mh_]))
        yh, vh = mh(dcode[mh_], female[mh_], death[mh_])
        yo, vo = mh(dcode[mo], female[mo], death[mo])
        out["fatality_hepatic_by_drug"] = est(yh, vh)
        out["fatality_other_by_drug"] = est(yo, vo)
        out["fatality_ratio_by_drug"] = est(yh - yo, vh + vo)
        mp = allm & case & coded
        out["phenotype_reports"] = int(mp.sum())
        out["phenotype_by_drug"] = est(*mh(dcode[mp], female[mp], chol[mp]))
        mpa = mp & adult
        out["phenotype_adult_drug_age"] = est(*mh(dacode[mpa], female[mpa], chol[mpa]))
        return out

    res = {"ocmq_version": "4.1", "queries": list(QUERIES),
           "terms": {"hepatic_injury_narrow": len(inj_n), "hepatic_failure_narrow": len(fail_n),
                     "narrow_either": len(inj_n | fail_n), "narrow_or_broad_either": len(any_t),
                     "narrow_either_found_in_faers": int(sum(1 for t in inj_n | fail_n if counts.get(t, 0) > 0)),
                     "paper_narrow": len(my_narrow), "paper_narrow_in_ocmq_narrow": len(my_narrow & (inj_n | fail_n)),
                     "paper_narrow_not_in_ocmq_narrow": sorted(my_narrow - (inj_n | fail_n)),
                     "paper_narrow_in_ocmq_broad_only": sorted((my_narrow & any_t) - (inj_n | fail_n)),
                     "paper_narrow_in_no_ocmq_list": sorted(my_narrow - any_t)}}
    for name, base in (("full", np.ones(n, dtype=bool)), ("restricted", le20)):
        both = base & narrow & o_narrow
        x = {"paper_cases": int((base & narrow).sum()), "ocmq_cases": int((base & o_narrow).sum()),
             "cases_under_both": int(both.sum()), "only_paper": int((base & narrow & ~o_narrow).sum()),
             "only_ocmq": int((base & o_narrow & ~narrow).sum()),
             "share_of_paper_cases_that_are_ocmq_cases": float(both.sum() / (base & narrow).sum()),
             "share_of_ocmq_cases_that_are_paper_cases": float(both.sum() / (base & o_narrow).sum()),
             "only_ocmq_that_meet_the_broad_definition_of_the_paper": int((base & o_narrow & ~narrow & broad).sum()),
             "phenotype_coded_paper_cases": int((base & narrow & coded).sum()),
             "phenotype_coded_not_ocmq_cases": int((base & narrow & coded & ~o_narrow).sum())}
        x["reference"] = contrasts(narrow, base, base & ~broad & ~narrow)
        x["ocmq_narrow"] = contrasts(o_narrow, base, base & ~o_any)
        x["ocmq_injury_narrow"] = contrasts(inj_narrow, base, base & ~o_any)
        # what the wider definition adds (post hoc, change 32)
        cf = {c: flags["c_" + c] for c in CLASSES}
        cc = flags["cc"]
        anyb = narrow | broad
        dec = {"paper_broad": contrasts(anyb, base, base & ~anyb),
               "reference_no_competing_cause_reports": contrasts(narrow, base & ~cc, base & ~anyb & ~cc),
               "ocmq_no_competing_cause_reports": contrasts(o_narrow, base & ~cc, base & ~o_any & ~cc),
               "competing_cause_reports": int((base & cc).sum()),
               "ocmq_cases_with_competing_cause_term": int((base & cc & o_narrow).sum()), "classes": {}, "cumulative": {}}
        for c in CLASSES:
            rest = np.zeros(n, dtype=bool)
            for d_ in CLASSES:
                if d_ != c:
                    rest |= cf[d_]
            only = base & cf[c] & ~rest
            dec["classes"][c] = {"terms": len(cls_terms[c]), "reports_with_a_term": int((base & cf[c]).sum()),
                                 "cases_only_through_this_class": int(only.sum()),
                                 "female_pct_of_those": float(100 * female[only].mean()) if only.any() else None,
                                 "left_out": contrasts(rest, base, base & ~o_any)}
        acc = np.zeros(n, dtype=bool)
        for c in ("shared", "lab", "other", "chronic", "competing"):
            acc = acc | cf[c]
            dec["cumulative"][c] = contrasts(acc, base, base & ~o_any)
        both_out = cf["shared"] | cf["lab"] | cf["other"]
        dec["without_chronic_and_competing_terms"] = contrasts(both_out, base, base & ~o_any)
        # Order-free attribution of the distance between the shared terms and all narrow terms (change 33).
        # The fully stratified log OR is computed for every subset of the four added classes. A class is credited
        # with its step when added first, when added last, and on average over all 24 orders.
        added = ("lab", "competing", "chronic", "other")
        ma_ = base & adult
        fc_, ff_ = fcode[ma_], female[ma_]
        val = {}
        for mask in range(16):
            case = cf["shared"].copy()
            for i, c in enumerate(added):
                if mask >> i & 1:
                    case |= cf[c]
            val[mask] = float(mh(fc_, ff_, case[ma_].astype(np.int64))[0])
        total = val[15] - val[0]
        of = {"log_or_shared": val[0], "log_or_all": val[15], "classes": {}}
        for i, c in enumerate(added):
            sh = 0.0
            for mask in range(16):
                if mask >> i & 1:
                    continue
                k = bin(mask).count("1")
                sh += math.factorial(k) * math.factorial(3 - k) / math.factorial(4) * (val[mask | 1 << i] - val[mask])
            of["classes"][c] = {"added_first_pct": 100 * (val[1 << i] - val[0]) / total,
                                "added_last_pct": 100 * (val[15] - val[15 ^ (1 << i)]) / total,
                                "all_orders_pct": 100 * sh / total,
                                "or_added_alone_to_shared": float(np.exp(val[1 << i])),
                                "or_dropped_from_all": float(np.exp(val[15 ^ (1 << i)]))}
        blk = (1 << added.index("competing")) | (1 << added.index("chronic"))
        of["liver_disease_terms"] = {"added_first_pct": 100 * (val[blk] - val[0]) / total,
                                     "added_last_pct": 100 * (val[15] - val[15 ^ blk]) / total,
                                     "all_orders_pct": of["classes"]["competing"]["all_orders_pct"] + of["classes"]["chronic"]["all_orders_pct"],
                                     "or_added_alone_to_shared": float(np.exp(val[blk])),
                                     "or_dropped_from_all": float(np.exp(val[15 ^ blk]))}
        dec["order_free"] = of
        x["decomposition"] = dec
        res[name] = x

    # scan by primary suspect stratum under the comparison definition (restricted set)
    ref = pd.read_csv(os.path.join(rdir, "stratum_scan_restricted.csv"), keep_default_na=False)
    rflag = set(ref.loc[ref["q_lrt"] < 0.05, "stratum"].astype(str))
    cnt = np.bincount(dcode[le20] * 4 + female[le20] * 2 + o_narrow[le20].astype(np.int64), minlength=nd * 4).reshape(-1, 4).astype(float)
    sc = D.scan(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0], names=dnames)
    t = sc.pop("table")
    flag = set(t.loc[t["q_lrt"] < 0.05, "stratum"].astype(str))
    allx = set(t["stratum"].astype(str))
    sc.update({"reference_flags": len(rflag), "flagged_in_both": len(rflag & flag),
               "reference_flags_lost": len((rflag - flag) & allx), "reference_flags_untested": len(rflag - allx),
               "new_flags": len(flag - rflag)})
    res["stratum_scan_ocmq_narrow"] = sc

    # term table, no counts by sex
    rows = []
    for _, r_ in o.iterrows():
        rows.append({"term": r_["pt"], "ocmq_query": r_["ocmq"], "ocmq_class": r_["classification"],
                     "paper_narrow": r_["pt"] in my_narrow, "paper_broad": r_["pt"] in my_broad,
                     "reports_with_term": int(counts.get(r_["pt"], 0))})
    for t_ in sorted(my_narrow - any_t):
        rows.append({"term": t_, "ocmq_query": "", "ocmq_class": "", "paper_narrow": True, "paper_broad": True,
                     "reports_with_term": int(counts.get(t_, 0))})
    tt = pd.DataFrame(rows).sort_values(["ocmq_query", "ocmq_class", "reports_with_term"], ascending=[True, True, False])
    tt.to_csv(os.path.join(rdir, "ocmq_term_comparison.csv"), index=False)
    on = tt[(tt["ocmq_class"] == "Narrow") & ~tt["paper_narrow"]].drop_duplicates("term").sort_values("reports_with_term", ascending=False)
    res["largest_ocmq_narrow_terms_outside_the_paper_definition"] = [
        {"term": a, "reports": int(b), "paper_broad": bool(c)} for a, b, c in zip(on["term"].head(15), on["reports_with_term"].head(15), on["paper_broad"].head(15))]
    with open(os.path.join(rdir, "ocmq_definition.json"), "w") as f:
        json.dump(res, f, indent=1)
    f = lambda e: f"{e['or']:.3f} ({e['lo']:.3f}-{e['hi']:.3f})"
    for k in ("full", "restricted"):
        d = res[k]["decomposition"]
        print(k, "ref", f(res[k]["reference"]["reporting_adult_full"]), "| broad", f(d["paper_broad"]["reporting_adult_full"]), "| ocmq", f(res[k]["ocmq_narrow"]["reporting_adult_full"]),
              "| ocmq no cc reports", f(d["ocmq_no_competing_cause_reports"]["reporting_adult_full"]), "| ref no cc reports", f(d["reference_no_competing_cause_reports"]["reporting_adult_full"]),
              "| ocmq without chronic+competing terms", f(d["without_chronic_and_competing_terms"]["reporting_adult_full"]))
        for c in CLASSES:
            z = d["classes"][c]
            print("   ", c, z["terms"], z["reports_with_a_term"], z["cases_only_through_this_class"], None if z["female_pct_of_those"] is None else round(z["female_pct_of_those"], 1),
                  "left out: crude", f(z["left_out"]["reporting_crude"]), "drug", f(z["left_out"]["reporting_by_drug"]), "full", f(z["left_out"]["reporting_adult_full"]))
        o_ = d["order_free"]
        print("    order-free:", {c: [round(z["added_first_pct"]), round(z["added_last_pct"]), round(z["all_orders_pct"])] for c, z in o_["classes"].items()},
              "liver disease", [round(o_["liver_disease_terms"][q]) for q in ("added_first_pct", "added_last_pct", "all_orders_pct")],
              "alone", {c: round(z["or_added_alone_to_shared"], 3) for c, z in o_["classes"].items()}, round(o_["liver_disease_terms"]["or_added_alone_to_shared"], 3))
        for c, z in d["cumulative"].items():
            print("    cum +", c, z["hepatic_reports"], "crude", f(z["reporting_crude"]), "drug", f(z["reporting_by_drug"]), "full", f(z["reporting_adult_full"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
