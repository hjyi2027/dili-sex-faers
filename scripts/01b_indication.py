"""Second pass over one FAERS quarterly zip (post hoc change 18).

Usage: python3 01b_indication.py <faers_ascii_YYYYQn.zip> <out_root>

Writes, under <out_root>/<quarter>/:
  psindi.parquet   one row per primaryid: first-listed indication term of the
                   primary suspect product ('' when none is given)
  compete.parquet  primaryids that carry a competing-cause reaction term, with the group
No tabulation by sex happens here.
"""
import os
import re
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from importlib import import_module

red = import_module("01_reduce")


def main(zip_path, out_root):
    q = re.search(r"(\d{4})[Qq](\d)", os.path.basename(zip_path))
    quarter = f"{q.group(1)}Q{q.group(2)}"
    out = os.path.join(out_root, quarter)
    os.makedirs(out, exist_ok=True)
    zf = zipfile.ZipFile(zip_path)

    drug, _ = red.read_table(zf, red.find_member(zf, "DRUG"),
                             ["primaryid", "drug_seq", "role_cod"])
    drug["role_cod"] = drug["role_cod"].str.upper().str.strip()
    ps = drug[drug["role_cod"] == "PS"].copy()
    del drug
    ps["primaryid"] = red.to_int(ps["primaryid"])
    ps["drug_seq"] = red.to_int(ps["drug_seq"])
    ps = ps.dropna(subset=["primaryid", "drug_seq"])
    ps["primaryid"] = ps["primaryid"].astype("int64")
    ps["drug_seq"] = ps["drug_seq"].astype("int64")
    # one primary suspect row per report: the lowest drug_seq
    ps = ps.sort_values(["primaryid", "drug_seq"]).drop_duplicates("primaryid")

    indi, _ = red.read_table(zf, red.find_member(zf, "INDI"),
                             ["primaryid", "indi_drug_seq", "indi_pt"])
    indi["primaryid"] = red.to_int(indi["primaryid"])
    indi["indi_drug_seq"] = red.to_int(indi["indi_drug_seq"])
    indi = indi.dropna(subset=["primaryid", "indi_drug_seq"])
    indi["primaryid"] = indi["primaryid"].astype("int64")
    indi["indi_drug_seq"] = indi["indi_drug_seq"].astype("int64")
    indi["indi_pt"] = indi["indi_pt"].str.upper().str.strip()
    indi["row"] = range(len(indi))
    m = ps.merge(indi, left_on=["primaryid", "drug_seq"],
                 right_on=["primaryid", "indi_drug_seq"], how="left")
    del indi
    m["indi_pt"] = m["indi_pt"].fillna("")
    m["unknown"] = m["indi_pt"].map(lib.is_unknown_indication)
    # first-listed informative indication of the primary suspect product
    m = m.sort_values(["primaryid", "unknown", "row"]).drop_duplicates("primaryid")
    m["ps_indi"] = m["indi_pt"].where(~m["unknown"], "")
    m[["primaryid", "ps_indi"]].to_parquet(os.path.join(out, "psindi.parquet"), index=False)

    reac, _ = red.read_table(zf, red.find_member(zf, "REAC"), ["primaryid", "pt"])
    reac["pt"] = reac["pt"].str.upper().str.strip()
    uniq = pd.Series(reac["pt"].unique())
    grp = {u: lib.competing_cause_group(u) for u in uniq}
    reac["grp"] = reac["pt"].map(grp)
    reac = reac[reac["grp"] != ""]
    reac["primaryid"] = red.to_int(reac["primaryid"])
    reac = reac.dropna(subset=["primaryid"])
    reac["primaryid"] = reac["primaryid"].astype("int64")
    reac[["primaryid", "pt", "grp"]].drop_duplicates().to_parquet(
        os.path.join(out, "compete.parquet"), index=False)
    print(quarter, "ps rows", len(m), "with indication", int((m["ps_indi"] != "").sum()),
          "compete rows", len(reac), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
