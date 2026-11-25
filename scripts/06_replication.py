"""Compare this study's per-ingredient sex odds ratios with Lu et al. 2023, Table 8.

Usage: python3 06_replication.py <result_dir> <lu_csv>
Their data: FAERS Jan 2004 to Jun 2015, primary suspect only, 53 preferred terms.
Ours: FAERS 2014Q3 onward. The two periods overlap by four quarters at most.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

NAME_MAP = {
    "Leuprolide": "LEUPROLIDE", "Isotretinoin": "ISOTRETINOIN", "Etanercept": "ETANERCEPT",
    "Rosuvastatin": "ROSUVASTATIN", "Fingolimod": "FINGOLIMOD", "Azithromycin": "AZITHROMYCIN",
    "Esomeprazole": "ESOMEPRAZOLE", "Adalimumab": "ADALIMUMAB", "Cyclosporine": "CYCLOSPORINE",
    "Amlodipine": "AMLODIPINE", "Metformin": "METFORMIN", "Sorafenib": "SORAFENIB",
    "Aripiprazole": "ARIPIPRAZOLE", "Doxorubicin": "DOXORUBICIN", "Propranolol": "PROPRANOLOL",
    "Amoxicillin-clavulanate": "CLAVULANATE", "Insulin Glargine": "INSULIN GLARGINE",
    "Clarithromycin": "CLARITHROMYCIN", "Doxycycline": "DOXYCYCLINE", "Valsartan": "VALSARTAN",
    "Ranitidine": "RANITIDINE", "Topotecan": "TOPOTECAN", "Fluvastatin": "FLUVASTATIN",
    "Peginterferon Alfa-2b": "PEGINTERFERON ALFA-2B",
}


def main(rdir, lu_csv):
    lu = pd.read_csv(lu_csv, comment="#")
    lu["lu_log_or"] = np.log((lu.female_liver_tox / lu.female_no_liver_tox)
                             / (lu.male_liver_tox / lu.male_no_liver_tox))
    lu["ingredient"] = lu["drug"].map(NAME_MAP)
    out = {}
    for variant in ["primary", "ps_only", "le20_ps", "le20", "broad"]:
        path = os.path.join(rdir, variant, "n1_ingredients.csv")
        if not os.path.exists(path):
            continue
        mine = pd.read_csv(path)
        res = json.load(open(os.path.join(rdir, variant, "results.json")))["N1"]["ingredients"]
        mu = res["mu"]
        m = lu.merge(mine[["ingredient", "f_case", "m_case", "log_or", "var", "q_lrt", "q_one"]],
                     on="ingredient", how="left")
        ok = m["log_or"].notna()
        mm = m[ok]
        rho, p = spearmanr(mm["lu_log_or"], mm["log_or"]) if ok.sum() > 2 else (float("nan"), float("nan"))
        out[variant] = {
            "lu_drugs": int(len(lu)), "matched_with_enough_cases": int(ok.sum()),
            "not_estimable_here": m.loc[~ok, "drug"].tolist(),
            "pooled_mu_here": mu, "pooled_or_here": math.exp(mu),
            "same_side_of_1": int((np.sign(mm["lu_log_or"]) == np.sign(mm["log_or"])).sum()),
            "same_side_of_pooled": int((np.sign(mm["lu_log_or"] - mu) == np.sign(mm["log_or"] - mu)).sum()),
            "flagged_here_vs_pooled_q05": int((mm["q_lrt"] < 0.05).sum()),
            "flagged_here_vs_one_q05": int((mm["q_one"] < 0.05).sum()),
            "spearman_rho": float(rho), "spearman_p": float(p),
            "lu_or_below_1": int((lu["lu_log_or"] < 0).sum()),
        }
        # base rates: how often would agreement occur for arbitrary ingredients of this scan
        base_below = res["n_or_below_1"] / res["k"]
        k_same1 = out[variant]["same_side_of_1"]
        n_est = int(ok.sum())
        out[variant]["base_rate_below_1"] = base_below
        out[variant]["p_same_side_of_1_vs_base"] = float(binom.sf(k_same1 - 1, n_est, base_below))
        out[variant]["p_same_side_of_pooled_vs_half"] = float(binom.sf(out[variant]["same_side_of_pooled"] - 1, n_est, 0.5))
        m.assign(lu_or=np.exp(m["lu_log_or"]), or_here=np.exp(m["log_or"])).to_csv(
            os.path.join(rdir, variant, "replication_lu2023.csv"), index=False)
    with open(os.path.join(rdir, "replication_lu2023.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
