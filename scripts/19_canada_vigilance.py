"""Canada Vigilance check of the Canadian linked groups (REPLICATION_PLAN.md, section 6 and its addendum).

Usage: python3 19_canada_vigilance.py <faers_data_dir> <analysis_dir> <cv_dir> <results_dir>

<cv_dir> holds the unpacked Health Canada data extract (reports.txt, report_links.txt,
report_drug.txt, drug_product_ingredients.txt). The files are '$' delimited, quoted, without header.

Step 1. FAERS reports in linked groups (Jaccard index at least 0.8) with country Canada are matched
to Canada Vigilance on the manufacturer control number: FAERS mfr_num against MAH_NO and the two
further manufacturer number columns of reports.txt, after upper-casing and removal of every
character that is not a letter or a digit. Equality of the whole key. No outcome is used.

Step 2. A FAERS group with two or more matched reports is confirmed when all its matched Canada
Vigilance reports are one report or are joined by Health Canada report links.

Step 3. The linkage rule of the paper is applied to Canada Vigilance itself.

Details fixed after the file layout was seen and before any match was computed (addendum to section 6):
 - reports.txt has 42 columns. Column 6 is MAH_NO. Columns 40 to 42 are not on the data structure
   page. By content, 40 and 42 carry manufacturer case numbers and 41 the authority number
   (CA-HEALTHCANVIG-...). Columns 6, 40 and 42 are the match targets of the plan.
 - Keys shorter than 6 characters are ignored. A FAERS key that equals keys of several Canada
   Vigilance reports keeps all of them.
 - Health Canada links: a row of report_links.txt joins its report to the report whose REPORT_NO
   equals the link number, or to the link number itself when no such report is in the extract.
   Rows without a link number join nothing. Reports are linked to each other when they fall in one
   connected component. Both record types (Duplicate, Linked) count, as the plan says "links".
   The result with Duplicate records only is given beside it.

Everything under the key "not_in_plan" was added after the results of steps 1 to 3 had been seen.
"""
import glob
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import norm_ingredient, split_prod_ai  # noqa: E402

MIN_SUSP = 11
THR = 0.8
MIN_KEY = 6
SIZE_BINS = (("2", 2, 2), ("3-9", 3, 9), ("10-99", 10, 99), ("100+", 100, 10 ** 9))
_NONALNUM = re.compile(r"[^A-Z0-9]")


def norm_key(s):
    return _NONALNUM.sub("", str(s).upper())


def read_cv(cv_dir, name, usecols):
    return pd.read_csv(os.path.join(cv_dir, name), sep="$", header=None, dtype=str, quotechar='"',
                       encoding="utf-8", na_filter=False, usecols=usecols)


def read_faers(data_dir, table, wanted, ids, keep=None):
    """Rows of one FAERS table for the given primaryids, as strings."""
    rows = []
    for z in sorted(glob.glob(os.path.join(data_dir, "faers_ascii_*.zip"))):
        zf = zipfile.ZipFile(z)
        m = [n for n in zf.namelist() if re.search(rf"(^|/){table}\d\dQ\d[^/]*\.txt$", n, re.I)][0]
        raw = zf.read(m)
        header = raw.split(b"\n", 1)[0].decode("latin-1").strip().lower().split("$")
        use = [i for i, h in enumerate(header) if h in wanted]
        d = pd.read_csv(io.BytesIO(raw), sep="$", header=None, skiprows=1, dtype=str, quoting=3,
                        encoding="latin-1", on_bad_lines="skip", usecols=use, na_filter=False)
        del raw
        d.columns = [header[i] for i in use]
        d["primaryid"] = pd.to_numeric(d["primaryid"], errors="coerce")
        for c in wanted:
            if c not in d.columns:
                d[c] = ""
        d = d[d["primaryid"].isin(ids)]
        if keep is not None:
            d = keep(d)
        rows.append(d)
    return pd.concat(rows)


def year_of(s):
    yy = pd.to_numeric(s.str[-2:], errors="coerce")
    return np.where(yy <= 30, 2000 + yy, 1900 + yy)


def hc_components(report_ids, links, no_to_id, types):
    """Component label per Canada Vigilance REPORT_ID from the report links of the given types."""
    l = links[links["type"].isin(types) & (links["link_no"] != "")]
    target = l["link_no"].map(no_to_id)
    ext = l["link_no"][target.isna()]
    ext_codes = {k: i for i, k in enumerate(pd.unique(ext))}
    n_rep = len(report_ids)
    pos = pd.Series(np.arange(n_rep), index=report_ids)
    src = pos.reindex(l["report_id"]).to_numpy()
    dst = np.where(target.notna(), pos.reindex(target.fillna(report_ids[0])).to_numpy(),
                   n_rep + l["link_no"].map(ext_codes).fillna(0).to_numpy())
    ok = ~np.isnan(src)
    src, dst = src[ok].astype(np.int64), dst[ok].astype(np.int64)
    n = n_rep + len(ext_codes)
    G = sp.csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    info = {"link_rows_used": int(len(src)), "link_rows_to_a_report_in_the_extract": int(target.notna().sum()),
            "link_rows_to_a_number_not_in_the_extract": int(target.isna().sum())}
    return comp[:n_rep], info


def complete_linkage_groups(M, size, blocks, age):
    """Labels per row: complete-linkage groups of pairs with Jaccard >= THR inside blocks, same age."""
    n = M.shape[0]
    edges = []
    for idx in blocks:
        if len(idx) < 2:
            continue
        Mb = M[idx]
        for s in range(0, len(idx), 2000):
            P = (Mb[s:s + 2000] @ Mb.T).tocoo()
            i, j, inter = P.row + s, P.col, P.data
            m = i < j
            i, j, inter = idx[i[m]], idx[j[m]], inter[m]
            jac = inter / (size[i] + size[j] - inter)
            k = (jac >= THR) & ((age[i] == age[j]) | (np.isnan(age[i]) & np.isnan(age[j])))
            edges.append(np.column_stack([i[k], j[k]]).astype(np.int32))
            del P
    E = np.vstack(edges) if edges else np.zeros((0, 2), dtype=np.int32)
    G = sp.csr_matrix((np.ones(len(E), dtype=np.int8), (E[:, 0], E[:, 1])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    lab = np.full(n, -1, dtype=np.int64)
    nxt = 0
    for _, idx in sorted(pd.Series(np.arange(n)).groupby(comp).indices.items()):
        if len(idx) <= 2:
            lab[idx] = nxt
            nxt += 1
            continue
        sub_m = M[idx]
        inter = np.asarray((sub_m @ sub_m.T).todense(), dtype=np.float32)
        sz = size[idx].astype(np.float32)
        dist = 1.0 - inter / (sz[:, None] + sz[None, :] - inter)
        del inter
        np.fill_diagonal(dist, 0.0)
        cl = fcluster(linkage(squareform(dist, checks=False), method="complete"), t=1.0 - THR + 1e-6,
                      criterion="distance")
        del dist
        lab[idx] = nxt + cl - 1
        nxt += int(cl.max())
    return lab


def size_bin(n):
    for name, lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return name
    return "1"


def faers_product_counts(data_dir, adir):
    """Not in the plan. Do FAERS reports reach 11 suspect ingredients through many products or through one?"""
    import pyarrow.parquet as pq
    a = pq.read_table(os.path.join(adir, "analysis.parquet"), columns=["primaryid", "sex", "overdose"]).to_pandas()
    a = a[a["sex"].notna() & ~a["overdose"]]
    pr = pq.read_table(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"]).to_pandas()
    pr = pr[pr["primaryid"].isin(a["primaryid"])].drop_duplicates()
    ns = pr.groupby("primaryid", sort=False).size()
    del pr
    el = pd.DataFrame({"primaryid": ns.index[ns >= MIN_SUSP]})
    del a, ns
    ids = set(el["primaryid"])
    d = read_faers(data_dir, "DRUG", ["primaryid", "role_cod", "prod_ai"], ids,
                   keep=lambda t: t.loc[t["role_cod"].isin(["PS", "SS"]), ["primaryid", "prod_ai"]].drop_duplicates())
    d = d.drop_duplicates()
    d["prod_ai"] = d["prod_ai"].str.upper().str.strip()
    d = d[d["prod_ai"] != ""]
    u = pd.Series(d["prod_ai"].unique())
    ning = dict(zip(u, u.map(lambda s: len(split_prod_ai(s)))))
    d["ning"] = d["prod_ai"].map(ning)
    g = d.groupby("primaryid").agg(products=("prod_ai", "nunique"), max_ing=("ning", "max")).reindex(el["primaryid"]).fillna(0)
    link = pd.read_parquet(os.path.join(adir, "link.parquet"), columns=["primaryid", "group_j80"])
    linked = g.index.isin(set(link.loc[link["group_j80"] >= 0, "primaryid"]))
    return {"eligible_reports": int(len(g)),
            "with_11_or_more_distinct_suspect_products": int((g["products"] >= MIN_SUSP).sum()),
            "with_one_product_of_11_or_more_ingredients": int((g["max_ing"] >= MIN_SUSP).sum()),
            "linked_reports": int(linked.sum()),
            "linked_with_11_or_more_distinct_suspect_products": int((linked & (g["products"] >= MIN_SUSP)).sum()),
            "linked_with_one_product_of_11_or_more_ingredients": int((linked & (g["max_ing"] >= MIN_SUSP)).sum()),
            "median_products_linked": float(g.loc[linked, "products"].median())}


def main(data_dir, adir, cv_dir, rdir):
    res = {"plan": "REPLICATION_PLAN.md section 6", "threshold": THR, "min_suspect_ingredients": MIN_SUSP}
    extra = {"faers_product_counts": faers_product_counts(data_dir, adir)}

    # ---------- FAERS side
    link = pd.read_parquet(os.path.join(adir, "link.parquet"), columns=["primaryid", "group_j80"])
    link = link[link["group_j80"] >= 0]
    lid = link["primaryid"].tolist()
    a = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                        columns=["primaryid", "country", "female", "narrow", "death", "age_yr", "wt_kg"],
                        filters=[("primaryid", "in", lid)])
    link = link.merge(a, on="primaryid", how="left")
    del a
    link["gsize"] = link["group_j80"].map(link["group_j80"].value_counts())
    ca = link[link["country"].astype(str) == "CA"].copy()
    keys = read_faers(data_dir, "DEMO", ["primaryid", "mfr_num", "auth_num", "mfr_sndr"], set(ca["primaryid"]))
    keys = keys.drop_duplicates("primaryid", keep="last")
    ca = ca.merge(keys, on="primaryid", how="left")
    for c in ("mfr_num", "auth_num", "mfr_sndr"):
        ca[c] = ca[c].fillna("")
    ca["k_mfr"] = ca["mfr_num"].map(norm_key)
    ca["k_auth"] = ca["auth_num"].map(norm_key)
    res["faers"] = {
        "linked_reports_all_countries": int(len(link)), "linked_groups_all_countries": int(link["group_j80"].nunique()),
        "linked_reports_canada": int(len(ca)), "linked_groups_canada": int(ca["group_j80"].nunique()),
        "canada_with_mfr_num": int((ca["k_mfr"].str.len() >= MIN_KEY).sum()),
        "canada_with_auth_num": int((ca["k_auth"].str.len() >= MIN_KEY).sum()),
        "canada_hepatic": int(ca["narrow"].sum()),
        "mfr_num_prefix": ca["mfr_num"].str.slice(0, 3).value_counts().head(5).to_dict(),
    }

    # ---------- Canada Vigilance
    r = read_cv(cv_dir, "reports.txt", [0, 1, 3, 5, 7, 10, 13, 19, 20, 37, 39, 40, 41])
    r.columns = ["report_id", "report_no", "datreceived", "mah_no", "type", "gender", "age_y", "weight", "weight_unit",
                 "source", "num_a", "authority", "num_b"]
    assert r["report_id"].is_unique
    w = pd.to_numeric(r["weight"], errors="coerce")
    unit = r["weight_unit"].str.lower()
    wt_all = np.where(unit.str.startswith("kilo"), w, np.where(unit.str.startswith("pound"), w * 0.45359237, np.nan))
    wt_all[(wt_all <= 0) | (wt_all > 400)] = np.nan
    year = year_of(r["datreceived"])
    age_all = pd.to_numeric(r["age_y"], errors="coerce").to_numpy()
    gender = r["gender"].to_numpy()
    sexed = np.isin(gender, ["Male", "Female"])
    rid = r["report_id"].to_numpy()
    nrep = len(r)
    res["canada_vigilance"] = {
        "reports": int(nrep), "reports_with_sex": int(sexed.sum()), "received_from_2014": int((year >= 2014).sum()),
        "with_mah_no": int((r["mah_no"] != "").sum()), "with_number_col40": int((r["num_a"] != "").sum()),
        "with_authority_number": int((r["authority"] != "").sum()), "with_number_col42": int((r["num_b"] != "").sum()),
        "report_type": r["type"].value_counts().head(6).to_dict(), "source": r["source"].value_counts().head(5).to_dict(),
    }
    pos = np.arange(nrep)
    key_maps = {}
    wanted = set(ca["k_mfr"]) | set(ca["k_auth"])
    for col in ("mah_no", "num_a", "num_b", "authority"):
        k = r[col].map(norm_key)
        ok = (k.str.len() >= MIN_KEY).to_numpy() & k.isin(wanted).to_numpy()
        key_maps[col] = pd.Series(pos[ok], index=k[ok].to_numpy()).groupby(level=0).agg(list).to_dict()

    # ---------- Step 1
    def match(keycol, targets):
        hits, by = [], {t: 0 for t in targets}
        for k in ca[keycol]:
            found = set()
            if len(k) >= MIN_KEY:
                for t in targets:
                    v = key_maps[t].get(k)
                    if v:
                        found.update(v)
                        by[t] += 1
            hits.append(found)
        return hits, by
    ca["m_plan"], by_plan = match("k_mfr", ("mah_no", "num_a", "num_b"))
    ca["m_auth"], by_auth = match("k_auth", ("authority",))
    ca["m_any"] = [p | q for p, q in zip(ca["m_plan"], ca["m_auth"])]
    del key_maps
    n_plan = int((ca["m_plan"].map(len) > 0).sum())
    mpos = np.array(sorted(set().union(*ca["m_plan"])), dtype=np.int64)
    res["step1"] = {
        "canadian_linked_reports": int(len(ca)),
        "matched_on_plan_keys": n_plan, "share_matched_on_plan_keys": n_plan / len(ca),
        "matched_by_target": {"MAH_NO": by_plan["mah_no"], "column_40": by_plan["num_a"], "column_42": by_plan["num_b"]},
        "matched_to_more_than_one_cv_report": int((ca["m_plan"].map(len) > 1).sum()),
        "distinct_cv_reports_matched": int(len(mpos)),
        "inconclusive_by_plan": bool(n_plan / len(ca) < 0.30),
        "share_matched_by_group_size": {b: float((ca.loc[ca["gsize"].map(size_bin) == b, "m_plan"].map(len) > 0).mean())
                                        for b, _, _ in SIZE_BINS if (ca["gsize"].map(size_bin) == b).any()},
        "matched_cv_reports": {"report_type": r["type"].iloc[mpos].value_counts().head(5).to_dict(),
                               "source": r["source"].iloc[mpos].value_counts().head(5).to_dict(),
                               "received_from_2014": int((year[mpos] >= 2014).sum())},
    }

    # ---------- Health Canada links
    links = read_cv(cv_dir, "report_links.txt", [1, 2, 4])
    links.columns = ["report_id", "type", "link_no"]
    no_to_id = pd.Series(rid, index=r["report_no"].to_numpy())
    no_to_id = no_to_id[~no_to_id.index.duplicated()]
    comp_any, info_any = hc_components(rid, links, no_to_id, ("Duplicate", "Linked"))
    comp_dup, info_dup = hc_components(rid, links, no_to_id, ("Duplicate",))
    sz_any = pd.Series(comp_any).map(pd.Series(comp_any).value_counts()).to_numpy()
    dup_partner = pd.Series(comp_dup).map(pd.Series(comp_dup).value_counts()).to_numpy() > 1
    has_blank = r["report_id"].isin(set(links.loc[links["link_no"] == "", "report_id"])).to_numpy()
    res["health_canada_links"] = {
        "rows": int(len(links)), "rows_by_type": links["type"].value_counts().to_dict(),
        "rows_without_link_number": int((links["link_no"] == "").sum()),
        "rows_without_link_number_by_type": links.loc[links["link_no"] == "", "type"].value_counts().to_dict(),
        "reports_with_a_link_row": int(links["report_id"].nunique()),
        "any_type": info_any, "duplicate_type": info_dup,
        "reports_in_a_component_of_two_or_more_any_type": int((sz_any > 1).sum()),
        "largest_component_any_type": int(sz_any.max()),
        "reports_in_a_component_of_two_or_more_duplicate_type": int(dup_partner.sum()),
        "share_of_all_reports_with_a_duplicate_record_partner": float(dup_partner.mean()),
    }
    del links, no_to_id

    # ---------- Step 2
    def step2(col):
        m = ca[ca[col].map(len) > 0]
        g = m.groupby("group_j80")[col].agg(lambda v: sorted(set().union(*v)))
        nm = m.groupby("group_j80").size()
        g = g[nm >= 2]
        gs = ca.drop_duplicates("group_j80").set_index("group_j80")["gsize"]
        rows = []
        for k, ps in g.items():
            ps = np.asarray(ps, dtype=np.int64)
            ww = wt_all[ps]
            ww = ww[~np.isnan(ww)]
            rows.append({"group": int(k), "group_size": int(gs[k]), "matched_reports": int(nm[k]), "cv_reports": int(len(ps)),
                         "components_any_link": int(len(set(comp_any[ps]))), "components_duplicate_link": int(len(set(comp_dup[ps]))),
                         "link_row_without_number": bool(has_blank[ps].any()), "weights_known": int(len(ww)),
                         "weight_range": float(ww.max() - ww.min()) if len(ww) >= 2 else np.nan})
        t = pd.DataFrame(rows)
        if not len(t):
            return {"checkable_groups": 0}, t
        t["one_report"] = t["cv_reports"] == 1
        t["confirmed"] = t["components_any_link"] == 1
        t["confirmed_duplicate"] = t["components_duplicate_link"] == 1
        t["bin"] = t["group_size"].map(size_bin)

        def summ(v):
            k = v[v["weights_known"] >= 2]
            return {"checkable_groups": int(len(v)), "reports_in_these_groups": int(v["group_size"].sum()),
                    "matched_reports_in_these_groups": int(v["matched_reports"].sum()),
                    "confirmed": int(v["confirmed"].sum()), "share_confirmed": float(v["confirmed"].mean()),
                    "of_which_one_cv_report": int(v["one_report"].sum()),
                    "not_confirmed": int((~v["confirmed"]).sum()),
                    "confirmed_duplicate_records_only": int(v["confirmed_duplicate"].sum()),
                    "share_confirmed_duplicate_records_only": float(v["confirmed_duplicate"].mean()),
                    "groups_with_two_weights": int(len(k)),
                    "weights_within_1kg": float((k["weight_range"] <= 1).mean()) if len(k) else None}
        out = summ(t)
        out["groups_canada_total"] = int(ca["group_j80"].nunique())
        out["by_group_size"] = {b: summ(t[t["bin"] == b]) for b, _, _ in SIZE_BINS if (t["bin"] == b).any()}
        full = t[t["matched_reports"] == t["group_size"]]
        out["groups_with_every_report_matched"] = summ(full) if len(full) else {"checkable_groups": 0}
        return out, t
    res["step2"], table = step2("m_plan")
    if len(table):
        table.drop(columns=["bin"]).to_csv(os.path.join(rdir, "canada_vigilance_groups.csv"), index=False)
    s2 = res["step2"]
    if res["step1"]["inconclusive_by_plan"] or not s2.get("checkable_groups"):
        res["reading_by_plan"] = "inconclusive"
    elif s2["share_confirmed"] >= 0.80:
        res["reading_by_plan"] = "most checkable groups confirmed"
    elif s2["share_confirmed"] < 0.50:
        res["reading_by_plan"] = "not shown to be single cases"
    else:
        res["reading_by_plan"] = "mixed"

    # ---------- Step 3: suspect ingredients per report, as integer codes
    pos_of = pd.Series(pos, index=rid)
    rd = read_cv(cv_dir, "report_drug.txt", [1, 2, 3, 4])
    rd.columns = ["report_id", "product_id", "drugname", "role"]
    rd = rd.loc[rd["role"] == "Suspect", ["report_id", "product_id", "drugname"]]
    rp = pos_of.reindex(rd["report_id"]).to_numpy()
    okr = ~np.isnan(rp)
    pcode, puniq = pd.factorize(rd["product_id"] + "\x1f" + rd["drugname"])
    rp, pcode = rp[okr].astype(np.int64), pcode[okr].astype(np.int64)
    del rd
    prod = pd.DataFrame({"prod": np.arange(len(puniq)), "product_id": [s.split("\x1f")[0] for s in puniq],
                         "drugname": [s.split("\x1f")[1] for s in puniq]})
    ing = read_cv(cv_dir, "drug_product_ingredients.txt", [1, 4])
    ing.columns = ["product_id", "ingredient"]
    pi_ = prod.merge(ing, on="product_id", how="left")
    pi_["ingredient"] = pi_["ingredient"].where(pi_["ingredient"].notna() & (pi_["ingredient"] != ""), pi_["drugname"])
    pi_["ingredient"] = pi_["ingredient"].map(norm_ingredient)
    pi_ = pi_[pi_["ingredient"] != ""][["prod", "ingredient"]].drop_duplicates()
    icode, inames = pd.factorize(pi_["ingredient"], sort=True)
    pi_["icode"] = icode
    pkey = pi_.groupby("prod")["ingredient"].agg(lambda v: "|".join(sorted(set(v))))
    pkc = pd.Series(pd.factorize(pkey)[0], index=pkey.index)
    # report x product-key pairs and report x ingredient pairs
    npu = len(puniq) + 1
    rpu = np.unique(rp * npu + pcode)
    rp_u, pc_u = rpu // npu, rpu % npu
    kc = pkc.reindex(pc_u).to_numpy()
    okk = ~np.isnan(kc)
    nk = int(pkc.max()) + 2
    nprod = np.bincount(np.unique(rp_u[okk] * nk + kc[okk].astype(np.int64)) // nk, minlength=nrep)
    e = pd.DataFrame({"rp": rp_u, "prod": pc_u}).merge(pi_[["prod", "icode"]], on="prod", how="inner")
    ni = len(inames) + 1
    ri = np.unique(e["rp"].to_numpy().astype(np.int64) * ni + e["icode"].to_numpy().astype(np.int64))
    del e, rpu
    ri_r, ri_i = ri // ni, ri % ni
    ns = np.bincount(ri_r, minlength=nrep)
    A = sp.csr_matrix((np.ones(len(ri_r), dtype=np.float32), (ri_r, ri_i)), shape=(nrep, len(inames)))
    del ri, ri_r, ri_i

    def step3(elig):
        epos = np.flatnonzero(elig)
        M = A[epos]
        size = np.asarray(M.sum(axis=1)).ravel()
        age = np.round(age_all[epos])
        g_ = gender[epos]
        blocks = [np.flatnonzero(g_ == s) for s in ("Female", "Male")]
        lab = complete_linkage_groups(M, size, blocks, age)
        gsz = pd.Series(lab).map(pd.Series(lab).value_counts()).to_numpy()
        linked = gsz > 1
        yr = year[epos]
        lp = epos[linked]
        gdf = pd.DataFrame({"g": lab[linked], "w": wt_all[lp], "blank": has_blank[lp], "noage": np.isnan(age[linked]),
                            "ca": comp_any[lp], "cd": comp_dup[lp], "yr": yr[linked]})
        gt = gdf.groupby("g").agg(n=("g", "size"), nw=("w", "count"), wmin=("w", "min"), wmax=("w", "max"),
                                  blank=("blank", "any"), noage=("noage", "mean"), ca=("ca", "nunique"),
                                  cd=("cd", "nunique"), y0=("yr", "min"), y1=("yr", "max"))

        def gsum(t):
            if not len(t):
                return {"groups": 0}
            k = t[t["nw"] >= 2]
            return {"groups": int(len(t)), "reports": int(t["n"].sum()),
                    "all_joined_by_health_canada_links": float((t["ca"] == 1).mean()),
                    "all_joined_by_duplicate_records": float((t["cd"] == 1).mean()),
                    "not_all_joined_and_a_link_row_without_number": float(((t["ca"] > 1) & t["blank"]).mean()),
                    "groups_with_two_weights": int(len(k)),
                    "weights_within_1kg": float(((k["wmax"] - k["wmin"]) <= 1).mean()) if len(k) else None,
                    "share_of_reports_without_age": float((t["noage"] * t["n"]).sum() / t["n"].sum()),
                    "median_span_years": float((t["y1"] - t["y0"]).median()), "largest_group": int(t["n"].max())}

        def bysize(t):
            return {b: gsum(t[(t["n"] >= lo) & (t["n"] <= hi)]) for b, lo, hi in SIZE_BINS if ((t["n"] >= lo) & (t["n"] <= hi)).any()}
        same_any = same_dup = npairs = 0
        for _, idx in pd.Series(np.flatnonzero(linked)).groupby(lab[linked]).indices.items():
            members = epos[np.flatnonzero(linked)[idx]]
            m = len(members)
            npairs += m * (m - 1) // 2
            c = pd.Series(comp_any[members]).value_counts().to_numpy()
            same_any += int((c * (c - 1) // 2).sum())
            c = pd.Series(comp_dup[members]).value_counts().to_numpy()
            same_dup += int((c * (c - 1) // 2).sum())
        out = {"eligible_reports": int(len(epos)), "eligible_share_of_reports_with_sex": float(len(epos) / sexed.sum()),
               "eligible_received_from_2014": int((yr >= 2014).sum()),
               "eligible_share_of_reports_with_sex_received_from_2014": float((yr >= 2014).sum() / (sexed & (year >= 2014)).sum()),
               "reports_in_a_group": int(linked.sum()), "share_of_eligible_in_a_group": float(linked.mean()),
               "share_of_eligible_in_a_group_received_from_2014": float(linked[yr >= 2014].mean()) if (yr >= 2014).any() else None,
               "groups": gsum(gt), "by_group_size": bysize(gt),
               "within_group_pairs": {"pairs": int(npairs),
                                      "also_joined_by_health_canada_links": float(same_any / npairs) if npairs else None,
                                      "also_joined_by_duplicate_records": float(same_dup / npairs) if npairs else None},
               "eligible_report_type": r["type"].iloc[epos].value_counts().head(5).to_dict(),
               "eligible_source": r["source"].iloc[epos].value_counts().head(5).to_dict(),
               "eligible_female_share": float((g_ == "Female").mean()),
               "eligible_share_with_a_duplicate_record_partner": float(dup_partner[epos].mean()),
               "eligible_by_year": {int(k): int(v) for k, v in pd.Series(yr).value_counts().sort_index().items() if k >= 2008}}
        recent = gt[gt["y0"] >= 2014]
        out["groups_received_from_2014"] = gsum(recent)
        out["groups_received_from_2014"]["by_group_size"] = bysize(recent)
        return out

    res["step3"] = step3(sexed & (ns >= MIN_SUSP))
    res["step3"]["note"] = ("Rule of section 6 as written: at least 11 distinct suspect ingredients. In Canada Vigilance most of "
                            "these reports reach 11 through one product with many listed ingredients (see not_in_plan).")

    # ---------- Not in section 6 (added after the results of steps 1 to 3 were seen)
    extra["step3_reports_with_11_or_more_suspect_ingredients"] = int((sexed & (ns >= MIN_SUSP)).sum())
    extra["step3_of_which_11_or_more_distinct_suspect_products"] = int((sexed & (ns >= MIN_SUSP) & (nprod >= MIN_SUSP)).sum())
    extra["step3_by_products"] = step3(sexed & (ns >= MIN_SUSP) & (nprod >= MIN_SUSP))
    extra["all_reports_by_year"] = {int(k): int(v) for k, v in pd.Series(year).value_counts().sort_index().items() if k >= 2008}
    # step 1 by FAERS sender
    snd = ca["mfr_sndr"].str.upper().str.strip()
    top = snd.value_counts().head(12)
    extra["step1_match_by_faers_sender"] = {k: {"reports": int(v), "matched": int((ca.loc[snd == k, "m_plan"].map(len) > 0).sum())}
                                            for k, v in top.items()}
    extra["step1_faers_senders"] = int(snd.nunique())
    extra["step1_groups_with_a_matched_report"] = int(ca.loc[ca["m_plan"].map(len) > 0, "group_j80"].nunique())
    extra["step1_matched_on_authority_number"] = int((ca["m_auth"].map(len) > 0).sum())
    extra["step1_matched_on_manufacturer_or_authority_number"] = int((ca["m_any"].map(len) > 0).sum())
    extra["step2_with_authority_number"] = {k: v for k, v in step2("m_any")[0].items() if k != "by_group_size"}
    # agreement of the key-matched pairs on fields that the key does not use
    fp = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"],
                         filters=[("primaryid", "in", ca["primaryid"].tolist())]).drop_duplicates()
    fset = fp.groupby("primaryid")["ingredient"].agg(lambda v: frozenset(map(str, v))).to_dict()
    inames = np.asarray(inames)
    jac, sexok, ageok, wtok = [], [], [], []
    for pid_, fem, fage, fw, hits in zip(ca["primaryid"], ca["female"], ca["age_yr"], ca["wt_kg"], ca["m_plan"]):
        for h in hits:
            a_ = fset.get(pid_, frozenset())
            b_ = frozenset(inames[A[h].indices])
            if a_ or b_:
                jac.append(len(a_ & b_) / len(a_ | b_))
            if gender[h] in ("Male", "Female"):
                sexok.append((gender[h] == "Female") == bool(fem))
            if not pd.isna(fage) and not np.isnan(age_all[h]):
                ageok.append(round(float(fage)) == round(float(age_all[h])))
            if not pd.isna(fw) and not np.isnan(wt_all[h]):
                wtok.append(abs(float(fw) - float(wt_all[h])) <= 1)
    jac = np.asarray(jac)
    extra["step1_key_matched_pairs"] = {
        "pairs": int(len(jac)), "median_jaccard_of_suspect_ingredient_sets": float(np.median(jac)) if len(jac) else None,
        "share_jaccard_at_least_0_8": float((jac >= 0.8).mean()) if len(jac) else None,
        "share_jaccard_at_least_0_5": float((jac >= 0.5).mean()) if len(jac) else None,
        "pairs_with_sex_on_both": int(len(sexok)), "same_sex": float(np.mean(sexok)) if sexok else None,
        "pairs_with_age_on_both": int(len(ageok)), "same_age_in_years": float(np.mean(ageok)) if ageok else None,
        "pairs_with_weight_on_both": int(len(wtok)), "weight_within_1kg": float(np.mean(wtok)) if wtok else None}
    if len(table):
        def wsum(t):
            k = t[t["weights_known"] >= 2]
            return {"groups": int(len(t)), "with_a_link_row_without_number": int(t["link_row_without_number"].sum()),
                    "groups_with_two_weights": int(len(k)),
                    "weights_within_1kg": float((k["weight_range"] <= 1).mean()) if len(k) else None}
        extra["step2_confirmed"] = wsum(table[table["confirmed"]])
        extra["step2_not_confirmed"] = wsum(table[~table["confirmed"]])
        extra["step2_confirmed_by_duplicate_records"] = wsum(table[table["confirmed_duplicate"]])
        extra["step2_not_confirmed_by_duplicate_records"] = wsum(table[~table["confirmed_duplicate"]])
    res["not_in_plan"] = extra
    with open(os.path.join(rdir, "canada_vigilance.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    show = {k: res[k] for k in ("faers", "step1", "reading_by_plan")}
    print(json.dumps(show, indent=1, default=str))
    print("step2", json.dumps({k: v for k, v in res["step2"].items() if k != "by_group_size"}, indent=1, default=str))
    print("step3", json.dumps({k: v for k, v in res["step3"].items() if k not in ("eligible_by_year", "by_group_size", "groups_received_from_2014")}, indent=1, default=str))
    print("extra", json.dumps({k: v for k, v in extra.items() if k not in ("all_reports_by_year",)}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
