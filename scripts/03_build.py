"""Build the report-level analysis table and the suspect-ingredient pair table.

Usage: python3 03_build.py <merged_dir> <analysis_dir> [--terms-only]

Step 1 writes terms.csv from TOTAL preferred-term counts only (no sex).
With --terms-only the script stops there so the term table can be reviewed
and frozen before any outcome is tabulated by sex.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
US_CODES = {"US", "USA", "UNITED STATES"}
NO_COUNTRY = {"", "COUNTRY NOT SPECIFIED", "NAN", "NONE"}
HP_CODES = {"MD", "PH", "OT", "HP", "RN"}


def load_merge_map():
    path = os.path.join(HERE, "ingredient_merge.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ingredient synonym table not found at {path}")
    m = pd.read_csv(path, dtype=str, comment="#")
    return dict(zip(m["from"].str.upper().str.strip(), m["to"].str.upper().str.strip()))


def load_dilirank():
    path = os.path.join(HERE, "..", "data", "ext", "dilirank.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DILIrank class list not found at {path} (see README, inputs/dilirank.csv)")
    # Rows in LTKBID order, so that the result does not depend on the order of the file. Three ingredients have
    # two entries with different classes once salt names are removed (amifampridine, metoprolol, norethindrone).
    # The entry with the higher LTKBID is used.
    d = pd.read_csv(path, dtype=str).sort_values("LTKBID", kind="stable")
    d["cls"] = d["vDILI-Concern"].str.lower().str.replace("v", "", n=1).str.split("-").str[0]
    d["ing"] = d["CompoundName"].map(lib.norm_ingredient)
    return dict(zip(d["ing"], d["cls"]))


def main(merged, out, terms_only=False):
    os.makedirs(out, exist_ok=True)

    # ---- 1. term table from total counts only
    ptc = pd.read_parquet(os.path.join(merged, "ptcounts.parquet"))
    cap = ptc[ptc["pt"].str.contains(lib.CAPTURE_RULE)].copy()
    flags = pd.DataFrame([lib.classify_pt(t) for t in cap["pt"]], index=cap.index)
    terms = pd.concat([cap, flags], axis=1).sort_values("n", ascending=False)
    terms.to_csv(os.path.join(out, "terms.csv"), index=False)
    print("captured terms", len(terms), "| narrow", int(terms.narrow.sum()),
          "| broad", int(terms.broad.sum()), "| overdose", int(terms.overdose.sum()))
    if terms_only:
        return

    # ---- 2. per-report reaction flags
    flagcols = ["hc", "ch", "mx", "un", "narrow", "broad", "severe", "overdose", "alcohol"]
    tmap = terms.set_index("pt")[flagcols]
    rows = pd.read_parquet(os.path.join(merged, "ptrows.parquet"))
    rows["pt"] = rows["pt"].astype(str)
    rows = rows.join(tmap, on="pt")
    rf = rows.groupby("primaryid")[flagcols].any()
    del rows

    # ---- 3. indication flags
    indi = pd.read_parquet(os.path.join(merged, "indirows.parquet"))
    indi["indi_pt"] = indi["indi_pt"].astype(str)
    ucat = pd.Series(indi["indi_pt"].unique())
    liv = set(ucat[ucat.map(lib.is_liverdx_indication)])
    alc = set(ucat[ucat.map(lib.is_alcohol_term)])
    liver_ids = pd.Index(indi.loc[indi["indi_pt"].isin(liv), "primaryid"].unique())
    alc_ids = pd.Index(indi.loc[indi["indi_pt"].isin(alc), "primaryid"].unique())
    pd.Series(sorted(liv)).to_csv(os.path.join(out, "liver_indication_terms.csv"),
                                  index=False, header=["indi_pt"])
    del indi

    # ---- 4. report table
    rep = pd.read_parquet(os.path.join(merged, "reports.parquet"))
    rep = rep.join(rf, on="primaryid")
    for c in flagcols:
        rep[c] = rep[c].fillna(False).astype(bool)
    rep["sex"] = rep["sex"].astype(str).where(rep["sex"].astype(str).isin(["F", "M"]))
    rep["female"] = (rep["sex"] == "F")
    rep["pheno"] = np.select(
        [rep["mx"] | (rep["hc"] & rep["ch"]), rep["hc"], rep["ch"]],
        ["MX", "HC", "CH"], default="")
    age = rep["age_yr"]
    rep["age_band"] = pd.cut(age, [18, 50, 65, 121], right=False,
                             labels=["18-49", "50-64", "65+"]).astype(object)
    rep["adult"] = age >= 18
    yr = rep["fda_dt"] // 10000
    rep["year"] = yr.astype("int16")
    rep["period"] = pd.cut(yr, [0, 2017, 2021, 9999], labels=["2014-2017", "2018-2021", "2022+"]
                           ).astype(object)
    # Reporter country, or country of occurrence when the reporter country is not given.
    # FAERS writes the literal "COUNTRY NOT SPECIFIED" for a missing country. The first
    # version of this script tested only for an empty string and so counted those reports
    # as non-US (corrected 2026-11-24, PROTOCOL change 18).
    rc = rep["reporter_country"].astype(str).where(
        ~rep["reporter_country"].astype(str).isin(NO_COUNTRY), "")
    oc = rep["occr_country"].astype(str).where(
        ~rep["occr_country"].astype(str).isin(NO_COUNTRY), "")
    country = rc.where(rc != "", oc)
    rep["country"] = country
    rep["region"] = np.where(country == "", "missing",
                             np.where(country.isin(US_CODES), "US", "nonUS"))
    occ = rep["occp_cod"].astype(str)
    rep["reporter"] = np.select(
        [occ.isin(HP_CODES), occ == "CN", occ == "LW"], ["HP", "consumer", "lawyer"],
        default="missing")
    rep["ning_band"] = pd.cut(rep["n_ing_all"], [-1, 1, 4, 9, 10000],
                              labels=["1", "2-4", "5-9", "10+"]).astype(object)
    rep["alcohol_proxy"] = rep["alcohol"] | rep["alc_drug"] | rep["primaryid"].isin(alc_ids)
    rep["liverdx_proxy"] = rep["primaryid"].isin(liver_ids)

    mm = load_merge_map()
    keys = pd.Series(rep["ps_key"].astype(str).unique())

    def remap(k):
        if not k:
            return ""
        return "\\".join(sorted({mm.get(p, p) for p in k.split("\\")}))
    kmap = dict(zip(keys, keys.map(remap)))
    rep["ps_drug"] = rep["ps_key"].astype(str).map(kmap).astype("category")

    dr = load_dilirank()
    order = {"most": 3, "less": 2, "ambiguous": 1, "no": 0}

    def dcls(k):
        if not k:
            return "unlisted"
        cl = [dr.get(p) for p in k.split("\\")]
        cl = [c for c in cl if c in order]
        return max(cl, key=lambda c: order[c]) if cl else "unlisted"
    ukeys = pd.Series(rep["ps_drug"].cat.categories)
    rep["dilirank"] = rep["ps_drug"].astype(str).map(dict(zip(ukeys, ukeys.map(dcls))))

    keep = ["primaryid", "caseid", "quarter", "year", "period", "sex", "female", "age_yr",
            "age_band", "adult", "wt_kg", "region", "country", "reporter", "rept_cod", "n_ing_all",
            "ning_band", "ps_drug", "dilirank", "death", "lifethreat", "hosp", "dup2",
            "hc", "ch", "mx", "un", "narrow", "broad", "severe", "overdose", "pheno",
            "alcohol_proxy", "liverdx_proxy"]
    rep = rep[keep]
    for c in ["quarter", "period", "sex", "age_band", "region", "country", "reporter", "rept_cod",
              "ning_band", "dilirank", "pheno"]:
        rep[c] = rep[c].astype("category")
    rep.to_parquet(os.path.join(out, "analysis.parquet"), index=False)

    # ---- 5. suspect pairs with merged ingredient names
    sp = pd.read_parquet(os.path.join(merged, "suspect.parquet"))
    cats = pd.Series(sp["ingredient"].cat.categories)
    sp["ingredient"] = sp["ingredient"].astype(str).map(dict(zip(cats, cats.map(lambda p: mm.get(p, p)))))
    sp = sp.drop_duplicates(["primaryid", "role_cod", "ingredient"])
    sp["ingredient"] = sp["ingredient"].astype("category")
    sp.to_parquet(os.path.join(out, "pairs.parquet"), index=False)
    print("reports", len(rep), "| pairs", len(sp), "| narrow cases", int(rep.narrow.sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], terms_only="--terms-only" in sys.argv)
