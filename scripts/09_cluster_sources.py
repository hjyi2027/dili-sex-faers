"""Describe the source fields of reports with more than 20 suspect ingredients.

Usage: python3 09_cluster_sources.py <data_dir> <analysis_dir> <out_json>
Reads manufacturer sender, manufacturer control number, literature reference and
report type from the raw DEMO files for the reports in the cluster.
"""
import glob, io, json, os, re, sys, zipfile
import pandas as pd

def main(data_dir, adir, out):
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    ns = pr.groupby("primaryid", observed=True, sort=False)["ingredient"].nunique().rename("n_susp").reset_index()
    del pr
    a = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                        columns=["primaryid", "sex", "overdose", "narrow", "death", "age_yr", "year"])
    a = a.merge(ns, on="primaryid")
    cl = a[(a.n_susp > 20) & a.sex.notna() & ~a.overdose]
    ids = set(cl.primaryid)
    rows = []
    for z in sorted(glob.glob(os.path.join(data_dir, "faers_ascii_*.zip"))):
        zf = zipfile.ZipFile(z)
        m = [n for n in zf.namelist() if re.search(r"(^|/)DEMO\d\dQ\d[^/]*\.txt$", n, re.I)][0]
        raw = zf.read(m)
        header = raw.split(b"\n", 1)[0].decode("latin-1").strip().lower().split("$")
        want = ["primaryid", "caseid", "mfr_sndr", "mfr_num", "lit_ref", "rept_cod", "auth_num", "init_fda_dt", "fda_dt"]
        use = [i for i, h in enumerate(header) if h in want]
        d = pd.read_csv(io.BytesIO(raw), sep="$", header=None, skiprows=1, dtype=str, quoting=3,
                        encoding="latin-1", on_bad_lines="skip", usecols=use, na_filter=False)
        d.columns = [header[i] for i in use]
        d["primaryid"] = pd.to_numeric(d["primaryid"], errors="coerce")
        rows.append(d[d.primaryid.isin(ids)])
    s = pd.concat(rows).drop_duplicates("primaryid").merge(cl, on="primaryid")
    hep = s[s.narrow]
    def top(x, k=8):
        return {str(i): int(v) for i, v in x.value_counts().head(k).items()}
    res = {
        "cluster_reports": int(len(cl)), "matched_in_demo": int(len(s)), "hepatic": int(len(hep)),
        "all": {"mfr_sndr_top": top(s.mfr_sndr), "distinct_senders": int(s.mfr_sndr.nunique()),
                "lit_ref_nonblank": int((s.lit_ref.str.strip() != "").sum()),
                "rept_cod": top(s.rept_cod), "mfr_num_prefix": top(s.mfr_num.str.slice(0, 3)),
                "distinct_caseids": int(s.caseid.nunique())},
        "hepatic_only": {"mfr_sndr_top": top(hep.mfr_sndr), "distinct_senders": int(hep.mfr_sndr.nunique()),
                         "lit_ref_nonblank": int((hep.lit_ref.str.strip() != "").sum()),
                         "lit_ref_top": top(hep.lit_ref.str.slice(0, 80), 5),
                         "age_top": top(hep.age_yr.round(), 6), "year": top(hep.year, 8),
                         "mfr_num_prefix": top(hep.mfr_num.str.slice(0, 3)),
                         "fatal": int(hep.death.sum()), "female": int((hep.sex == "F").sum())},
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res)[:1500])

if __name__ == "__main__":
    main(*sys.argv[1:4])
