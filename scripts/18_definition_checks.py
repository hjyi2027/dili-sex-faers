"""Checks of the case definition and of the comparison group (post hoc, change 27).

Usage: python3 scripts/18_definition_checks.py <data_dir> <analysis_dir> <results_dir>

1. Strict case definition. Four terms of the narrow hepatic definition can name something other than drug
   toxicity: hepatic encephalopathy and coma hepatic (complications of existing liver disease, and the indication
   of some drugs), liver transplant, and autoimmune hepatitis (a disease in its own right). Reports whose only
   narrow terms are among these four are left out, and the rest of the definition is unchanged.
2. Pregnancy, obstetric and sex-linked events. Reports with such a reaction term (rule SEX_RULE on the term
   name) are removed from the analysis, cases and non-cases alike. A second version removes them from the
   non-cases only and keeps the hepatic injury reports that carry such a term. That version is asymmetric and
   is reported for comparison.
Both are applied to the reporting odds ratio (crude, by drug, fully stratified in adults), to the fatality ratio
by drug and, for the comparison group, to the scan by primary suspect stratum.
Also written: counts for two drugs whose indication bears on the reading of a flag (rifaximin, lenvatinib).

Writes <analysis_dir>/reac_flags.parquet, <analysis_dir>/sex_specific_terms.csv and
<results_dir>/definition_checks.json.
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
QUESTIONABLE = ["HEPATIC ENCEPHALOPATHY", "COMA HEPATIC", "LIVER TRANSPLANT", "AUTOIMMUNE HEPATITIS"]
SEX_RULE = (r"PREGNAN|MATERNAL|PATERNAL|FOETAL|FOETUS|ABORTION|STILLBIRTH|\bLABOUR\b|PREMATURE DELIVERY|CAESAREAN|"
            r"POSTPARTUM|PUERPER|LACTATION|BREAST|MENSTRU|MENORRH|METRORRHAG|MENOPAUS|UTERINE|UTERUS|OVARIAN|OVARY|"
            r"VAGINAL|VULV|ENDOMETRI|CERVIX|PROSTAT|TESTIC|TESTIS|ERECTILE|EJACULAT|PENILE|PENIS|SCROTAL|"
            r"GYNAECOMASTIA|SPERM|EPIDIDYM")
# classes of the marked terms, by name: pregnancy and obstetric terms first, then tumour terms, then the rest
PREG_RULE = (r"PREGNAN|MATERNAL|PATERNAL|FOETAL|FOETUS|ABORTION|STILLBIRTH|\bLABOUR\b|PREMATURE DELIVERY|CAESAREAN|"
             r"POSTPARTUM|PUERPER|LACTATION|BREAST FEEDING|BREAST MILK")
CANCER_RULE = r"CANCER|CARCINOMA|NEOPLASM|TUMOUR|SARCOMA|METASTA|MALIGNAN|ADENOMA|LYMPHOMA|MELANOMA"
CLASSES = ("pregnancy", "cancer", "other")


def term_class(pt):
    import re
    if re.search(PREG_RULE, pt):
        return "pregnancy"
    if re.search(CANCER_RULE, pt):
        return "cancer"
    return "other"


def scan_reac(ddir, pid, narrow_terms):
    n = len(pid)
    core_terms = set(narrow_terms) - set(QUESTIONABLE)
    sexspec = np.zeros(n, dtype=bool)
    bycls = {c: np.zeros(n, dtype=bool) for c in CLASSES}
    core = np.zeros(n, dtype=bool)
    quest = {t: np.zeros(n, dtype=bool) for t in QUESTIONABLE}
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
        u = pd.Series(pt.unique())
        sx = set(u[u.str.contains(SEX_RULE)])
        is_sx = pt.isin(sx).to_numpy()
        sexspec[idx[is_sx]] = True
        for c in CLASSES:
            sc_ = {t for t in sx if term_class(t) == c}
            bycls[c][idx[pt.isin(sc_).to_numpy()]] = True
        pairs = pd.DataFrame({"i": idx[is_sx], "pt": pt[is_sx].to_numpy()}).drop_duplicates()
        counts.update(pairs["pt"].value_counts().to_dict())
        core[idx[pt.isin(core_terms).to_numpy()]] = True
        for t in QUESTIONABLE:
            quest[t][idx[(pt == t).to_numpy()]] = True
    return sexspec, core, quest, counts, bycls


def est(y, v):
    return {"or": math.exp(y), "lo": math.exp(y - Z * math.sqrt(v)), "hi": math.exp(y + Z * math.sqrt(v))}


def main(ddir, adir, rdir):
    rep = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                          columns=["primaryid", "sex", "female", "overdose", "narrow", "broad", "death", "ps_drug",
                                   "age_band", "adult", "period", "region", "reporter"])
    rep = rep[rep["sex"].notna() & ~rep["overdose"]].sort_values("primaryid").reset_index(drop=True)
    pid = rep["primaryid"].to_numpy()
    n = len(rep)
    terms = pd.read_csv(os.path.join(adir, "terms.csv"))
    narrow_terms = terms.loc[terms["narrow"].astype(str) == "True", "pt"].str.upper().tolist()
    sexspec, core, quest, counts, bycls = scan_reac(ddir, pid, narrow_terms)
    anyq = np.zeros(n, dtype=bool)
    for t in QUESTIONABLE:
        anyq |= quest[t]
    pd.DataFrame({"primaryid": pid, "sex_specific": sexspec, "narrow_core": core, "questionable": anyq}).to_parquet(
        os.path.join(adir, "reac_flags.parquet"), index=False)
    tt = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["pt", "reports"])
    tt["class"] = tt["pt"].map(term_class)
    tt.to_csv(os.path.join(adir, "sex_specific_terms.csv"), index=False)

    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    pr = pr[pr["primaryid"].isin(pid)].drop_duplicates()
    nsusp = np.bincount(np.searchsorted(pid, pr["primaryid"].to_numpy()), minlength=n)
    del pr
    le20 = nsusp <= 20
    female = rep["female"].to_numpy().astype(np.int64)
    narrow = rep["narrow"].to_numpy().astype(bool)
    death = rep["death"].to_numpy().astype(np.int64)
    notbroad = ~rep["broad"].to_numpy().astype(bool) & ~narrow
    adult = rep["adult"].to_numpy().astype(bool)
    cat = rep["ps_drug"].astype(str).astype("category")
    dcode = cat.cat.codes.to_numpy().astype(np.int64)
    dnames = np.asarray(cat.cat.categories.astype(str))
    nd = len(dnames)
    fraw = dcode.copy()
    for col in ("age_band", "period", "region", "reporter"):
        c, u = pd.factorize(rep[col].astype(str))
        fraw = fraw * len(u) + c
    fcode = np.unique(fraw, return_inverse=True)[1].astype(np.int64)
    del fraw

    def mh(code, f, y):
        k = int(code.max()) + 1 if len(code) else 1
        cnt = np.bincount(code * 4 + f * 2 + y, minlength=k * 4).reshape(-1, 4).astype(float)
        return lib.mantel_haenszel(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0])[:2]

    def contrasts(case, allm, other):
        y = case.astype(np.int64)
        f, c = female[allm], y[allm]
        out = {"reports": int(allm.sum()), "hepatic_reports": int((allm & case).sum())}
        out["reporting_crude"] = est(*[float(x) for x in lib.woolf(((f == 1) & (c == 1)).sum(), ((f == 1) & (c == 0)).sum(),
                                                                    ((f == 0) & (c == 1)).sum(), ((f == 0) & (c == 0)).sum())])
        out["reporting_by_drug"] = est(*mh(dcode[allm], f, c))
        ma = allm & adult
        out["reporting_adult_full"] = est(*mh(fcode[ma], female[ma], y[ma]))
        mh_ = allm & case
        mo = other & np.isin(dcode, np.unique(dcode[mh_]))
        yh, vh = mh(dcode[mh_], female[mh_], death[mh_])
        yo, vo = mh(dcode[mo], female[mo], death[mo])
        out["fatality_ratio_by_drug"] = est(yh - yo, vh + vo)
        return out

    qonly = narrow & ~core
    res = {"questionable_terms": QUESTIONABLE, "sex_rule": SEX_RULE, "sex_specific_terms_matched": len(counts),
           "narrow_flag_mismatch": int((narrow != (core | anyq)).sum()),
           "term_mentions_by_class_pct": {c: 100 * sum(v for k, v in counts.items() if term_class(k) == c) / sum(counts.values())
                                          for c in CLASSES}}
    for name, base in (("full", np.ones(n, dtype=bool)), ("restricted", le20)):
        hep = base & narrow
        x = {"hepatic_reports": int(hep.sum()),
             "questionable_only": int((base & qonly).sum()),
             "questionable_only_pct_of_hepatic": 100 * (base & qonly).sum() / hep.sum(),
             "questionable_only_female_pct": 100 * female[base & qonly].mean(),
             "other_hepatic_female_pct": 100 * female[base & core].mean(),
             "term_reports_among_hepatic": {t: int((hep & quest[t]).sum()) for t in QUESTIONABLE},
             "sex_specific_reports": int((base & sexspec).sum()),
             "sex_specific_pct_of_reports": 100 * (base & sexspec).sum() / base.sum(),
             "sex_specific_pct_of_women_noncases": 100 * (base & sexspec & ~narrow & (female == 1)).sum() / (base & ~narrow & (female == 1)).sum(),
             "sex_specific_pct_of_men_noncases": 100 * (base & sexspec & ~narrow & (female == 0)).sum() / (base & ~narrow & (female == 0)).sum(),
             "hepatic_reports_with_sex_specific_term": int((hep & sexspec).sum())}
        keep_cmp = base & (narrow | ~sexspec)
        x["reference"] = contrasts(narrow, base, base & notbroad)
        x["strict_definition"] = contrasts(core, base & ~qonly, base & notbroad)
        x["comparison_restricted"] = contrasts(narrow, keep_cmp, base & notbroad & ~sexspec)
        x["both"] = contrasts(core, keep_cmp & ~qonly, base & notbroad & ~sexspec)
        keep_sym = base & ~sexspec
        x["events_excluded"] = contrasts(narrow, keep_sym, base & notbroad & ~sexspec)
        x["events_excluded_strict"] = contrasts(core, keep_sym & ~qonly, base & notbroad & ~sexspec)
        x["hepatic_with_term_female_pct"] = 100 * female[hep & sexspec].mean()
        x["noncases_with_term_female_pct"] = 100 * female[base & sexspec & ~narrow].mean()
        x["hepatic_with_term_fatal_pct"] = 100 * death[hep & sexspec].mean()
        x["by_class"] = {}
        for c in CLASSES:
            mc = bycls[c]
            x["by_class"][c] = {
                "reports": int((base & mc).sum()), "hepatic_reports": int((hep & mc).sum()),
                "hepatic_female_pct": 100 * female[hep & mc].mean(), "hepatic_fatal_pct": 100 * death[hep & mc].mean(),
                "excluded": contrasts(narrow, base & ~mc, base & notbroad & ~mc)}
        x["hepatic_without_term_fatal_pct"] = 100 * death[hep & ~sexspec].mean()
        res[name] = x

    longhep = narrow & ~le20
    res["long_list_hepatic_reports"] = int(longhep.sum())
    res["long_list_hepatic_with_term_pct"] = 100 * (longhep & sexspec).sum() / longhep.sum()

    # scan by primary suspect stratum under each check (restricted set)
    ref = pd.read_csv(os.path.join(rdir, "stratum_scan_restricted.csv"), keep_default_na=False)
    rflag = set(ref.loc[ref["q_lrt"] < 0.05, "stratum"].astype(str))
    for label, m, case in (("comparison_restricted", le20 & (narrow | ~sexspec), narrow),
                           ("events_excluded", le20 & ~sexspec, narrow),
                           ("strict_definition", le20 & ~qonly, core)):
        cnt = np.bincount(dcode[m] * 4 + female[m] * 2 + case[m].astype(np.int64), minlength=nd * 4).reshape(-1, 4).astype(float)
        sc = D.scan(cnt[:, 3], cnt[:, 2], cnt[:, 1], cnt[:, 0], names=dnames)
        t = sc.pop("table")
        flag = set(t.loc[t["q_lrt"] < 0.05, "stratum"].astype(str))
        allx = set(t["stratum"].astype(str))
        sc.update({"reference_flags": len(rflag), "flagged_in_both": len(rflag & flag),
                   "reference_flags_lost": len((rflag - flag) & allx), "reference_flags_untested": len(rflag - allx),
                   "new_flags": len(flag - rflag)})
        res["stratum_scan_" + label] = sc

    # two drugs whose indication bears on the reading of a flag (restricted set, primary suspect)
    indi = pd.read_parquet(os.path.join(adir, "ps_indi.parquet"), columns=["primaryid", "ps_indi"])
    ind = pd.Series(pid).map(indi.drop_duplicates("primaryid").set_index("primaryid")["ps_indi"]).fillna("").astype(str).str.upper()
    rif = le20 & (cat == "RIFAXIMIN").to_numpy()
    res["rifaximin"] = {"reports": int(rif.sum()), "hepatic_reports": int((rif & narrow).sum()),
                        "hepatic_with_encephalopathy_term": int((rif & narrow & (quest["HEPATIC ENCEPHALOPATHY"] | quest["COMA HEPATIC"])).sum()),
                        "hepatic_questionable_only": int((rif & qonly).sum()),
                        "indication_hepatic_encephalopathy_pct": 100 * ind[rif].str.contains("ENCEPHALOPATHY").mean()}
    len_ = le20 & (cat == "LENVATINIB").to_numpy()
    liver = ind.str.contains("HEPATOCELLULAR|HEPATIC CANCER|HEPATIC NEOPLASM|LIVER").to_numpy()
    known = (ind != "").to_numpy()
    res["lenvatinib"] = {
        "reports_women": int((len_ & (female == 1)).sum()), "reports_men": int((len_ & (female == 0)).sum()),
        "liver_cancer_indication_pct_women": 100 * (len_ & liver & (female == 1)).sum() / max(1, (len_ & known & (female == 1)).sum()),
        "liver_cancer_indication_pct_men": 100 * (len_ & liver & (female == 0)).sum() / max(1, (len_ & known & (female == 0)).sum())}
    json.dump(res, open(os.path.join(rdir, "definition_checks.json"), "w"), indent=1)
    print("definition checks:", json.dumps(res)[:2500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
