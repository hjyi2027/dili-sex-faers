"""Link reports with long, near-identical suspect lists (post hoc change 19).

Usage: python3 03c_linkage.py <data_dir> <analysis_dir> <merged_dir>

Population: reports with sex recorded, no overdose-context term and at least MIN_SUSP
distinct suspect ingredients. Two reports are linked when they agree on sex, reporter
country and age in whole years (or both lack an age) and the Jaccard index of their
suspect-ingredient sets is at least the threshold. Linked pairs form connected components,
and reports inside each component are grouped by complete linkage, so that every pair of
reports in a group reaches the threshold. Fields NOT used for linkage (body weight, event date, manufacturer sender)
are compared on linked pairs and on random pairs of the same sex and country, as a check.

Writes <analysis_dir>/link.parquet (primaryid, group and keep flags for thresholds 0.8 and
0.9; the report kept in a group is the one with the latest FDA receipt date, ties by the
highest primaryid) and <analysis_dir>/link_summary.json. No outcome is used for linkage.
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

MIN_SUSP = 11
THRESHOLDS = (0.8, 0.9)
SEED = 20261124


def read_senders(data_dir, ids):
    rows = []
    for z in sorted(glob.glob(os.path.join(data_dir, "faers_ascii_*.zip"))):
        zf = zipfile.ZipFile(z)
        m = [n for n in zf.namelist() if re.search(r"(^|/)DEMO\d\dQ\d[^/]*\.txt$", n, re.I)][0]
        raw = zf.read(m)
        header = raw.split(b"\n", 1)[0].decode("latin-1").strip().lower().split("$")
        use = [i for i, h in enumerate(header) if h in ("primaryid", "mfr_sndr", "lit_ref")]
        d = pd.read_csv(io.BytesIO(raw), sep="$", header=None, skiprows=1, dtype=str, quoting=3,
                        encoding="latin-1", on_bad_lines="skip", usecols=use, na_filter=False)
        d.columns = [header[i] for i in use]
        d["primaryid"] = pd.to_numeric(d["primaryid"], errors="coerce")
        if "lit_ref" not in d.columns:
            d["lit_ref"] = ""
        rows.append(d[d["primaryid"].isin(ids)])
    return pd.concat(rows).drop_duplicates("primaryid", keep="last")


def agreement(i, j, wt, ev, snd):
    w = ~np.isnan(wt[i]) & ~np.isnan(wt[j])
    d = (ev[i] > 0) & (ev[j] > 0)
    s = (snd[i] != "") & (snd[j] != "")
    return {"pairs": int(len(i)),
            "weight_both_known": float(w.mean()),
            "weight_within_1kg": float((np.abs(wt[i] - wt[j])[w] <= 1).mean()) if w.any() else None,
            "event_date_both_known": float(d.mean()),
            "event_date_equal": float((ev[i] == ev[j])[d].mean()) if d.any() else None,
            "different_sender": float((snd[i] != snd[j])[s].mean()) if s.any() else None}


def main(data_dir, adir, mdir):
    a = pd.read_parquet(os.path.join(adir, "analysis.parquet"),
                        columns=["primaryid", "sex", "overdose", "female", "country", "age_yr",
                                 "wt_kg", "narrow", "death"])
    a = a[a["sex"].notna() & ~a["overdose"]]
    pr = pd.read_parquet(os.path.join(adir, "pairs.parquet"), columns=["primaryid", "ingredient"])
    pr = pr[pr["primaryid"].isin(a["primaryid"])].drop_duplicates()
    ns = pr.groupby("primaryid", sort=False)["ingredient"].size()
    ids = ns.index[ns >= MIN_SUSP]
    sub = a[a["primaryid"].isin(ids)].sort_values("primaryid").reset_index(drop=True)
    ev = pd.read_parquet(os.path.join(mdir, "reports.parquet"), columns=["primaryid", "event_dt", "fda_dt"])
    sub = sub.merge(ev, on="primaryid", how="left")
    sub = sub.merge(read_senders(data_dir, set(sub["primaryid"])), on="primaryid", how="left")
    sub["ns"] = sub["primaryid"].map(ns).astype(int)
    # literature reference, not used for linkage: letters and digits only, upper case
    sub["lit"] = sub["lit_ref"].fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    # The same article is written differently by different senders. Two references are taken to agree
    # when their sets of words of five or more letters have a Jaccard index of at least 0.3.
    toks = [frozenset(re.findall(r"[A-Z]{5,}", str(x).upper())) for x in sub["lit_ref"].fillna("")]

    def lit_same(a, b):
        ta, tb = toks[a], toks[b]
        if len(ta) < 3 or len(tb) < 3:
            return None
        return len(ta & tb) / len(ta | tb) >= 0.3

    def lit_pairs(ii, jj):
        r = [lit_same(a, b) for a, b in zip(ii, jj)]
        r = [x for x in r if x is not None]
        return {"pairs_with_two_references": len(r), "same_reference": float(np.mean(r)) if r else None}
    p = pr[pr["primaryid"].isin(ids)]
    row = pd.Series(np.arange(len(sub)), index=sub["primaryid"].to_numpy())
    codes, names = pd.factorize(p["ingredient"].astype(str), sort=True)
    M = sp.csr_matrix((np.ones(len(p), dtype=np.float32), (row.loc[p["primaryid"]].to_numpy(), codes)),
                      shape=(len(sub), len(names)))
    size = np.asarray(M.sum(axis=1)).ravel()
    age = sub["age_yr"].round().to_numpy()
    edges = []
    for _, idx in sorted(sub.groupby(["female", "country"], observed=True).indices.items()):
        if len(idx) < 2:
            continue
        Mb = M[idx]
        for s in range(0, len(idx), 3000):
            P = (Mb[s:s + 3000] @ Mb.T).tocoo()
            i, j, inter = P.row + s, P.col, P.data
            m = i < j
            i, j, inter = idx[i[m]], idx[j[m]], inter[m]
            jac = inter / (size[i] + size[j] - inter)
            k = (jac >= min(THRESHOLDS)) & ((age[i] == age[j]) | (np.isnan(age[i]) & np.isnan(age[j])))
            edges.append(np.column_stack([i[k], j[k], jac[k]]))
    E = np.vstack(edges)
    wt = sub["wt_kg"].to_numpy(dtype=float)
    evd = sub["event_dt"].fillna(0).to_numpy()
    snd = sub["mfr_sndr"].fillna("").str.upper().str.strip().to_numpy()
    rng = np.random.default_rng(SEED)
    samp = []
    for _, idx in sorted(sub.groupby(["female", "country"], observed=True).indices.items()):
        if len(idx) > 50:
            x = rng.choice(idx, size=min(20000, len(idx) * 5))
            y = rng.choice(idx, size=len(x))
            samp.append(np.column_stack([x, y]))
    S = np.vstack(samp)
    S = S[S[:, 0] != S[:, 1]]
    # second comparator: random pairs that also share the age in whole years (or both lack an age)
    samp2 = []
    akey = pd.Series(age).fillna(-1).to_numpy()
    for _, idx in sorted(sub.assign(akey=akey).groupby(["female", "country", "akey"], observed=True).indices.items()):
        if len(idx) > 5:
            x = rng.choice(idx, size=min(5000, len(idx) * 5))
            y = rng.choice(idx, size=len(x))
            samp2.append(np.column_stack([x, y]))
    S2 = np.vstack(samp2)
    S2 = S2[S2[:, 0] != S2[:, 1]]
    summary = {"min_suspect_ingredients": MIN_SUSP, "reports_eligible": int(len(sub)),
               "eligible_hepatic": int(sub["narrow"].sum()),
               "random_pairs_same_sex_country": agreement(S[:, 0], S[:, 1], wt, evd, snd),
               "random_pairs_same_sex_country_age": agreement(S2[:, 0], S2[:, 1], wt, evd, snd)}
    out = sub[["primaryid"]].copy()
    long = (sub["ns"] > 20).to_numpy()

    def describe(lab, thr):
        """Summary of one grouping (labels per eligible report, groups are labels shared by 2+ reports)."""
        g = pd.Series(lab)
        gs = g.map(g.value_counts()).to_numpy()
        linked = gs > 1
        order = sub.assign(g=lab).sort_values(["g", "fda_dt", "primaryid"])
        keep_last = sub["primaryid"].isin(set(order.groupby("g").tail(1)["primaryid"])).to_numpy()
        keep_first = sub["primaryid"].isin(set(order.groupby("g").head(1)["primaryid"])).to_numpy()
        drop, drop_first = linked & ~keep_last, linked & ~keep_first
        L = sub[linked].assign(g=lab[linked], gsize=gs[linked])
        gg = L.groupby("g").agg(n=("primaryid", "size"), hep=("narrow", "mean"), fatal=("death", "mean"),
                                senders=("mfr_sndr", "nunique"))
        gw = L[L["wt_kg"].notna()].groupby("g")["wt_kg"].agg(["size", "min", "max"])
        gw = gw[gw["size"] >= 2]
        gd = L[L["event_dt"].fillna(0) > 0].groupby("g")["event_dt"].agg(["size", "nunique"])
        gd = gd[gd["size"] >= 2]
        gg["w_ok"] = ((gw["max"] - gw["min"]) <= 1).reindex(gg.index)
        gg["d_ok"] = (gd["nunique"] == 1).reindex(gg.index)
        gl = L[L["lit"] != ""].groupby("g")["lit"].agg(["size", "nunique"])
        gl = gl[gl["size"] >= 2]
        fz = []
        for _, idx in pd.Series(np.flatnonzero(linked)).groupby(lab[linked]).indices.items():
            members = [m for m in np.flatnonzero(linked)[idx] if len(toks[m]) >= 3]
            if len(members) >= 2:
                fz.append(all(lit_same(members[0], m) for m in members[1:]))
        gg["mixed_hep"] = (gg["hep"] > 0) & (gg["hep"] < 1)
        gg["mixed_fatal"] = (gg["fatal"] > 0) & (gg["fatal"] < 1)
        # all pairs inside groups
        pi, pj = [], []
        for _, idx in pd.Series(np.flatnonzero(linked)).groupby(lab[linked]).indices.items():
            members = np.flatnonzero(linked)[idx]
            a, b = np.triu_indices(len(members), 1)
            pi.append(members[a]); pj.append(members[b])
        pi, pj = np.concatenate(pi), np.concatenate(pj)
        jac = np.empty(len(pi))
        for s0 in range(0, len(pi), 200000):
            a_, b_ = pi[s0:s0 + 200000], pj[s0:s0 + 200000]
            inter = np.asarray(M[a_].multiply(M[b_]).sum(axis=1)).ravel()
            jac[s0:s0 + 200000] = inter / (size[a_] + size[b_] - inter)
        apart = lab[S2[:, 0]] != lab[S2[:, 1]]
        by_size = {}
        for name, lo, hi in (("2", 2, 2), ("3-9", 3, 9), ("10-99", 10, 99), ("100+", 100, 10 ** 9)):
            x = gg[(gg["n"] >= lo) & (gg["n"] <= hi)]
            tested = x[x["w_ok"].notna() | x["d_ok"].notna()]
            fail = tested[(tested["w_ok"] == False) | (tested["d_ok"] == False)]  # noqa: E712
            by_size[name] = {
                "groups": int(len(x)), "reports": int(x["n"].sum()),
                "groups_with_two_weights": int(x["w_ok"].notna().sum()),
                "weights_within_1kg": float(x["w_ok"].dropna().astype(float).mean()) if x["w_ok"].notna().any() else None,
                "groups_with_two_event_dates": int(x["d_ok"].notna().sum()),
                "single_event_date": float(x["d_ok"].dropna().astype(float).mean()) if x["d_ok"].notna().any() else None,
                "reports_in_groups_that_fail_a_check": int(fail["n"].sum()),
                "reports_in_groups_that_pass_all_available_checks": int(tested["n"].sum() - fail["n"].sum()),
                "reports_in_groups_with_nothing_to_check": int(x["n"].sum() - tested["n"].sum()),
                "groups_mixed_hepatic_status": int(x["mixed_hep"].sum()),
                "reports_in_groups_mixed_hepatic_status": int(x.loc[x["mixed_hep"], "n"].sum()),
            }
        nar = sub["narrow"].to_numpy().astype(bool)
        dth = sub["death"].to_numpy().astype(bool)
        res = {
            "threshold": thr, "groups": int(len(gg)), "reports_in_groups": int(linked.sum()),
            "share_of_eligible_in_groups": float(linked.mean()),
            "reports_dropped": int(drop.sum()),
            "median_group_size": float(gg["n"].median()), "largest_group": int(gg["n"].max()),
            "min_pairwise_jaccard_within_groups": float(jac.min()),
            "share_of_within_group_pairs_below_threshold": float((jac < thr - 1e-9).mean()),
            "groups_with_more_than_one_sender": float((gg["senders"] > 1).mean()),
            "groups_mixed_hepatic_status": float(gg["mixed_hep"].mean()),
            "groups_mixed_fatal_status": float(gg["mixed_fatal"].mean()),
            "reports_in_groups_mixed_hepatic_status": int(gg.loc[gg["mixed_hep"], "n"].sum()),
            "over20_reports": int(long.sum()),
            "over20_in_groups": int((long & linked).sum()),
            "over20_remaining_after_collapse": int((long & ~drop).sum()),
            "dropped_female_share": float(sub.loc[drop, "female"].mean()),
            "dropped_canada_share": float((sub.loc[drop, "country"] == "CA").mean()),
            "dropped_hepatic": int(nar[drop].sum()),
            "dropped_fatal_hepatic": int((nar & dth)[drop].sum()),
            "dropped_age_missing_share": float(sub.loc[drop, "age_yr"].isna().mean()),
            "linked_hepatic_reports": int(nar[linked].sum()),
            "hepatic_kept_latest_rule": int(nar[linked & keep_last].sum()),
            "hepatic_kept_earliest_rule": int(nar[linked & keep_first].sum()),
            "groups_with_any_hepatic_report": int((gg["hep"] > 0).sum()),
            "fatal_hepatic_kept_latest_rule": int((nar & dth)[linked & keep_last].sum()),
            "fatal_hepatic_kept_earliest_rule": int((nar & dth)[linked & keep_first].sum()),
            "linked_pairs": agreement(pi, pj, wt, evd, snd),
            "random_pairs_same_sex_country_age_not_linked": agreement(S2[apart, 0], S2[apart, 1], wt, evd, snd),
            "random_pairs_same_sex_country_age_share_in_same_group": float(1 - apart.mean()),
            "groups_with_two_known_weights": int(gg["w_ok"].notna().sum()),
            "groups_weight_range_within_1kg": float(gg["w_ok"].dropna().astype(float).mean()),
            "groups_with_two_known_event_dates": int(gg["d_ok"].notna().sum()),
            "groups_single_event_date": float(gg["d_ok"].dropna().astype(float).mean()),
            "eligible_reports_with_literature_reference": int((sub["lit"] != "").sum()),
            "linked_reports_with_literature_reference": int((L["lit"] != "").sum()),
            "share_linked_reports_with_literature_reference": float((L["lit"] != "").mean()),
            "share_unlinked_reports_with_literature_reference": float((sub.loc[~linked, "lit"] != "").mean()),
            "groups_with_two_literature_references": int(len(gl)),
            "groups_single_literature_reference": float((gl["nunique"] == 1).mean()) if len(gl) else None,
            "groups_with_two_usable_literature_references": int(len(fz)),
            "groups_literature_references_agree": float(np.mean(fz)) if fz else None,
            "literature_reference_unlinked_pairs": lit_pairs(S2[apart, 0], S2[apart, 1]),
            "literature_reference_random_pairs_same_sex_country": lit_pairs(S[:, 0], S[:, 1]),
            "groups_of_two": int((gg["n"] == 2).sum()),
            "groups_of_ten_or_more": int((gg["n"] >= 10).sum()),
            "reports_in_groups_of_ten_or_more": int(gg.loc[gg["n"] >= 10, "n"].sum()),
            "by_group_size": by_size,
        }
        tested_g = gg["w_ok"].notna() | gg["d_ok"].notna()
        fail_g = (gg["w_ok"] == False) | (gg["d_ok"] == False)  # noqa: E712
        stat = pd.Series("none", index=gg.index)
        stat[tested_g] = "pass"
        stat[fail_g] = "fail"
        status = pd.Series(lab).map(stat).fillna("").to_numpy()
        return res, linked, drop, drop_first, status

    for thr in THRESHOLDS:
        tag = f"j{int(thr * 100)}"
        e = E[E[:, 2] >= thr]
        i, j = e[:, 0].astype(int), e[:, 1].astype(int)
        G = sp.csr_matrix((np.ones(len(e)), (i, j)), shape=(len(sub), len(sub)))
        _, comp = connected_components(G, directed=False)
        # Connected components chain reports that are not alike. Inside each component, reports are
        # therefore grouped by complete linkage, so that every pair in a group reaches the threshold.
        lab = comp.astype(np.int64) * 0 - 1
        nxt = 0
        for _, idx in sorted(pd.Series(np.arange(len(comp))).groupby(comp).indices.items()):
            if len(idx) == 1:
                lab[idx] = nxt; nxt += 1
                continue
            if len(idx) == 2:
                lab[idx] = nxt; nxt += 1
                continue
            sub_m = M[idx]
            inter = np.asarray((sub_m @ sub_m.T).todense(), dtype=np.float64)
            sz = size[idx].astype(np.float64)
            dist = 1.0 - inter / (sz[:, None] + sz[None, :] - inter)
            np.fill_diagonal(dist, 0.0)
            cl = fcluster(linkage(squareform(dist, checks=False), method="complete"),
                          t=1.0 - thr + 1e-9, criterion="distance")
            lab[idx] = nxt + cl - 1
            nxt += int(cl.max())
        res_cc = describe(comp, thr)[0]
        res, linked, drop, drop_first, status = describe(lab, thr)
        res["connected_components"] = {k: res_cc[k] for k in (
            "groups", "reports_in_groups", "reports_dropped", "largest_group", "min_pairwise_jaccard_within_groups",
            "share_of_within_group_pairs_below_threshold", "groups_weight_range_within_1kg",
            "groups_single_event_date", "by_group_size")}
        out[f"group_{tag}"] = np.where(linked, lab, -1)
        out[f"drop_{tag}"] = drop
        out[f"dropfirst_{tag}"] = drop_first
        out[f"check_{tag}"] = status
        summary[tag] = res
    out = out[(out["group_j80"] >= 0) | (out["group_j90"] >= 0)]
    out.to_parquet(os.path.join(adir, "link.parquet"), index=False)
    with open(os.path.join(adir, "link_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("linkage:", {k: (summary[k]["groups"], summary[k]["reports_dropped"]) for k in ("j80", "j90")})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
