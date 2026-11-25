"""Reduce one FAERS quarterly ASCII zip to compact parquet tables.

Usage: python3 01_reduce.py <faers_ascii_YYYYQn.zip> <out_dir>

Writes, under <out_dir>/<quarter>/:
  demo.parquet       one row per primaryid (pre-deduplication)
  suspect.parquet    one row per (primaryid, role, ingredient) for PS and SS
  perreport.parquet  per-report derived fields from DRUG, REAC, OUTC, INDI
  ptrows.parquet     (primaryid, pt) rows matching the capture rule
  ptcounts.parquet   total count of every preferred term in the quarter
  indirows.parquet   (primaryid, indi_pt) rows matching the indication rule
  deleted.parquet    caseids FDA lists as deleted (if the zip has such a file)
  qc.json            row counts and parse diagnostics
No tabulation by sex happens here.
"""
import hashlib
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402


def find_member(zf, prefix):
    pat = re.compile(rf"(^|/){prefix}\d\dQ\d[^/]*\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{prefix}: expected one member, found {hits}")
    return hits[0]


def read_table(zf, member, wanted, aliases=None):
    """Read a $-delimited FAERS table as strings, keeping only wanted columns."""
    aliases = aliases or {}
    raw = zf.read(member)
    n_lines = raw.count(b"\n")
    header = raw.split(b"\n", 1)[0].decode("latin-1").strip().lower().split("$")
    header = [aliases.get(h, h) for h in header]
    use = [i for i, h in enumerate(header) if h in wanted]
    df = pd.read_csv(
        io.BytesIO(raw), sep="$", header=None, skiprows=1, dtype=str,
        quoting=3, encoding="latin-1", on_bad_lines="skip", usecols=use,
        na_filter=False, engine="c",
    )
    df.columns = [header[i] for i in use]
    missing = [w for w in wanted if w not in df.columns]
    for w in missing:
        df[w] = ""
    return df, {"member": member, "lines": int(n_lines), "rows": int(len(df)),
                "missing_cols": missing}


def to_int(s):
    return pd.to_numeric(s, errors="coerce")


def set_hash(values):
    h = hashlib.blake2b("|".join(sorted(set(values))).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big", signed=True)


def main(zip_path, out_root):
    q = re.search(r"(\d{4})[Qq](\d)", os.path.basename(zip_path))
    quarter = f"{q.group(1)}Q{q.group(2)}"
    out = os.path.join(out_root, quarter)
    os.makedirs(out, exist_ok=True)
    qc = {"quarter": quarter, "zip": os.path.basename(zip_path)}
    zf = zipfile.ZipFile(zip_path)

    # ---------------- DEMO ----------------
    demo, qc["demo"] = read_table(
        zf, find_member(zf, "DEMO"),
        ["primaryid", "caseid", "caseversion", "fda_dt", "event_dt", "rept_cod",
         "age", "age_cod", "sex", "wt", "wt_cod", "occp_cod", "reporter_country",
         "occr_country"],
        aliases={"gndr_cod": "sex"},
    )
    demo["primaryid"] = to_int(demo["primaryid"])
    demo["caseid"] = to_int(demo["caseid"])
    bad = demo["primaryid"].isna() | demo["caseid"].isna()
    qc["demo"]["bad_id_rows"] = int(bad.sum())
    demo = demo[~bad].copy()
    demo["primaryid"] = demo["primaryid"].astype("int64")
    demo["caseid"] = demo["caseid"].astype("int64")
    demo["caseversion"] = to_int(demo["caseversion"]).fillna(0).astype("int32")
    demo["fda_dt"] = to_int(demo["fda_dt"]).fillna(0).astype("int32")
    age = to_int(demo["age"])
    cod = demo["age_cod"].str.upper().str.strip()
    factor = cod.map({"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52.1775,
                      "DY": 1 / 365.25, "HR": 1 / 8766.0})
    age_yr = age * factor
    age_yr = age_yr.where((age_yr >= 0) & (age_yr <= 120))
    demo["age_yr"] = age_yr.astype("float32")
    wt = to_int(demo["wt"])
    wcod = demo["wt_cod"].str.upper().str.strip()
    wfac = wcod.map({"KG": 1.0, "KGS": 1.0, "LBS": 0.45359237, "LB": 0.45359237,
                     "GMS": 0.001, "GM": 0.001})
    wt_kg = wt * wfac
    wt_kg = wt_kg.where((wt_kg >= 1) & (wt_kg <= 400))
    demo["wt_kg"] = wt_kg.astype("float32")
    demo["sex"] = demo["sex"].str.upper().str.strip()
    demo["occp_cod"] = demo["occp_cod"].str.upper().str.strip()
    demo["rept_cod"] = demo["rept_cod"].str.upper().str.strip()
    demo["reporter_country"] = demo["reporter_country"].str.upper().str.strip()
    demo["occr_country"] = demo["occr_country"].str.upper().str.strip()
    demo["quarter"] = quarter
    demo = demo.drop(columns=["age", "age_cod", "wt", "wt_cod"])
    demo = demo.drop_duplicates("primaryid")
    demo.to_parquet(os.path.join(out, "demo.parquet"), index=False)
    ids = pd.Index(demo["primaryid"])

    # ---------------- DRUG ----------------
    drug, qc["drug"] = read_table(
        zf, find_member(zf, "DRUG"), ["primaryid", "role_cod", "prod_ai", "drugname"])
    drug["primaryid"] = to_int(drug["primaryid"])
    drug = drug.dropna(subset=["primaryid"])
    drug["primaryid"] = drug["primaryid"].astype("int64")
    drug["role_cod"] = drug["role_cod"].str.upper().str.strip()
    drug["prod_ai"] = drug["prod_ai"].str.upper().str.strip()
    qc["drug"]["prod_ai_blank_rows"] = int((drug["prod_ai"] == "").sum())
    qc["drug"]["prod_ai_blank_suspect_rows"] = int(
        ((drug["prod_ai"] == "") & drug["role_cod"].isin(["PS", "SS"])).sum())
    uniq = pd.Series(drug["prod_ai"].unique())
    split_map = {u: lib.split_prod_ai(u) for u in uniq}
    key_map = {u: "\\".join(v) for u, v in split_map.items()}
    drug["prod_key"] = drug["prod_ai"].map(key_map)
    # exploded ingredient rows (all roles)
    ing = drug[["primaryid", "role_cod", "prod_ai"]].copy()
    ing["ingredient"] = ing["prod_ai"].map(split_map)
    ing = ing.explode("ingredient").dropna(subset=["ingredient"])
    ing = ing[ing["ingredient"] != ""]
    n_ing = ing.groupby("primaryid")["ingredient"].nunique().rename("n_ing_all")
    alc_drug = ing.loc[ing["ingredient"].isin(["ALCOHOL", "ETHANOL"]), "primaryid"].unique()
    susp = ing[ing["role_cod"].isin(["PS", "SS"])][["primaryid", "role_cod", "ingredient"]]
    susp = susp.drop_duplicates()
    susp.to_parquet(os.path.join(out, "suspect.parquet"), index=False)
    ps = drug[drug["role_cod"] == "PS"].drop_duplicates("primaryid")
    ps_key = ps.set_index("primaryid")["prod_key"].rename("ps_key")
    n_ps = drug[drug["role_cod"] == "PS"].groupby("primaryid").size()
    qc["drug"]["reports_with_gt1_ps"] = int((n_ps > 1).sum())
    susp_hash = susp.groupby("primaryid")["ingredient"].agg(set_hash).rename("susp_hash")
    del drug, ing

    # ---------------- REAC ----------------
    reac, qc["reac"] = read_table(zf, find_member(zf, "REAC"), ["primaryid", "pt"])
    reac["primaryid"] = to_int(reac["primaryid"])
    reac = reac.dropna(subset=["primaryid"])
    reac["primaryid"] = reac["primaryid"].astype("int64")
    reac["pt"] = reac["pt"].str.upper().str.strip()
    reac = reac[reac["pt"] != ""].drop_duplicates()
    ptcounts = reac["pt"].value_counts().rename_axis("pt").reset_index(name="n")
    ptcounts.to_parquet(os.path.join(out, "ptcounts.parquet"), index=False)
    cap = reac["pt"].str.contains(lib.CAPTURE_RULE)
    reac[cap].to_parquet(os.path.join(out, "ptrows.parquet"), index=False)
    pt_hash = reac.groupby("primaryid")["pt"].agg(set_hash).rename("pt_hash")
    n_pt = reac.groupby("primaryid").size().rename("n_pt")
    del reac

    # ---------------- OUTC ----------------
    outc, qc["outc"] = read_table(
        zf, find_member(zf, "OUTC"), ["primaryid", "outc_cod"], aliases={"outc_code": "outc_cod"})
    outc["primaryid"] = to_int(outc["primaryid"])
    outc = outc.dropna(subset=["primaryid"])
    outc["primaryid"] = outc["primaryid"].astype("int64")
    outc["outc_cod"] = outc["outc_cod"].str.upper().str.strip()
    death = pd.Index(outc.loc[outc["outc_cod"] == "DE", "primaryid"].unique())
    lifet = pd.Index(outc.loc[outc["outc_cod"] == "LT", "primaryid"].unique())
    hosp = pd.Index(outc.loc[outc["outc_cod"] == "HO", "primaryid"].unique())
    del outc

    # ---------------- INDI ----------------
    indi, qc["indi"] = read_table(zf, find_member(zf, "INDI"), ["primaryid", "indi_pt"])
    indi["primaryid"] = to_int(indi["primaryid"])
    indi = indi.dropna(subset=["primaryid"])
    indi["primaryid"] = indi["primaryid"].astype("int64")
    indi["indi_pt"] = indi["indi_pt"].str.upper().str.strip()
    icap = indi["indi_pt"].str.contains(lib.INDI_CAPTURE_RULE)
    indi[icap].drop_duplicates().to_parquet(os.path.join(out, "indirows.parquet"), index=False)
    del indi

    # ---------------- per-report table ----------------
    per = pd.DataFrame(index=ids)
    per.index.name = "primaryid"
    per = per.join(n_ing).join(ps_key).join(susp_hash).join(pt_hash).join(n_pt)
    per["n_ing_all"] = per["n_ing_all"].fillna(0).astype("int16")
    per["n_pt"] = per["n_pt"].fillna(0).astype("int16")
    per["ps_key"] = per["ps_key"].fillna("")
    per["death"] = per.index.isin(death)
    per["lifethreat"] = per.index.isin(lifet)
    per["hosp"] = per.index.isin(hosp)
    per["alc_drug"] = per.index.isin(alc_drug)
    per.reset_index().to_parquet(os.path.join(out, "perreport.parquet"), index=False)

    # ---------------- deleted cases ----------------
    dele = [n for n in zf.namelist() if re.search(r"delet", n, re.I) and n.lower().endswith(".txt")]
    qc["deleted_members"] = dele
    if dele:
        vals = []
        for n in dele:
            for line in zf.read(n).decode("latin-1").splitlines():
                line = line.strip().strip("$")
                if line.isdigit():
                    vals.append(int(line))
        pd.DataFrame({"caseid": pd.Series(vals, dtype="int64")}).to_parquet(
            os.path.join(out, "deleted.parquet"), index=False)
        qc["deleted_caseids"] = len(vals)

    qc["reports"] = int(len(demo))
    with open(os.path.join(out, "qc.json"), "w") as f:
        json.dump(qc, f, indent=1)
    print(quarter, "reports", len(demo), "suspect rows", len(susp))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
