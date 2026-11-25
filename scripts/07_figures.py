"""Draw the manuscript figures from the result files.

Usage: python3 07_figures.py <result_dir> <fig_dir>
Static print figures. Palette: two categorical hues validated for colour-vision
deficiency (blue #2a78d6, orange #eb6834) plus neutral gray for unflagged marks.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
Z = 1.959964
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})


UPPER_TOKENS = {"amg-510": "AMG-510", "lu-177": "Lu-177"}


def pretty(name):
    """Sentence case for a stratum name. Single letters (immunoglobulin G, amphotericin B, T-lymphocyte) stay capital."""
    import re
    s = name.replace("\\", " + ").capitalize()
    s = re.sub(r"(?<![\w-])([a-z])(?=$|\s|-[a-z]{3,})", lambda m: m.group(1).upper(), s)
    return " ".join(UPPER_TOKENS.get(t.lower(), t) for t in s.split(" "))


def logticks(ax, vals, axis="x"):
    f = ax.set_xticks if axis == "x" else ax.set_yticks
    g = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    f([math.log(v) for v in vals])
    g([f"{v:g}" for v in vals])


def whisk(ax, y, e, color, marker="o"):
    ax.plot([math.log(e["lo"]), math.log(e["hi"])], [y, y], color=color, lw=2,
            solid_capstyle="round", zorder=2)
    ax.plot(e["log_or"], y, marker, color=color, ms=7, mec="white", mew=1.5, zorder=3)


def fig_n1(res, rdir, out):
    t = pd.read_csv(os.path.join(rdir, "le20_ps", "n1_ingredients.csv"))
    scan_figure(t, res["N1"]["ingredients"]["mu"], "ingredient", "a  All ingredients",
                os.path.join(out, "fig1_n1_ingredients.pdf"))


def fig_stratum(rdir, out):
    """Scan by primary suspect stratum (restricted set), the scan used for inference."""
    t = pd.read_csv(os.path.join(rdir, "stratum_scan_restricted.csv"), keep_default_na=False)
    mu = json.load(open(os.path.join(rdir, "stratum_scan_sets.json")))["restricted"]["mu"]
    t.loc[t["stratum"] == "", "stratum"] = "No primary suspect"
    scan_figure(t, mu, "stratum", "a  All primary suspect strata", os.path.join(out, "fig4_stratum_scan.pdf"))


def scan_figure(t, mu, namecol, title_a, path):
    t = t.copy()
    t["prec"] = 1 / np.sqrt(t["var"])
    flag = t["q_lrt"] < 0.05
    fem = flag & (t["log_or"] > mu)
    mal = flag & (t["log_or"] < mu)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1, 1.0]})
    ax = axes[0]
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(mu, color=INK2, lw=1, zorder=1)
    ax.scatter(t.loc[~flag, "log_or"], t.loc[~flag, "prec"], s=14, color=GRAY, lw=0, zorder=2,
               label="Not flagged")
    ax.scatter(t.loc[fem, "log_or"], t.loc[fem, "prec"], s=22, color=BLUE, ec="white", lw=0.6,
               zorder=3, label="Shifted toward women")
    ax.scatter(t.loc[mal, "log_or"], t.loc[mal, "prec"], s=22, color=ORANGE, ec="white", lw=0.6,
               zorder=3, label="Shifted toward men")
    lim = max(1.6, float(np.abs(t["log_or"] - mu).quantile(0.995)) + 0.2)
    ax.set_xlim(mu - lim, mu + lim)
    logticks(ax, [v for v in (0.125, 0.25, 0.5, 1, 2, 4) if mu - lim < math.log(v) < mu + lim])
    ax.set_xlabel("Odds ratio, women vs men (log scale)")
    ax.set_ylabel("Precision (1 / standard error)")
    ax.set_title(title_a)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=3, fontsize=7,
              handletextpad=0.2, columnspacing=1.0)
    # forest of the strongest flagged ingredients by shrunken estimate
    ax = axes[1]
    f = t[flag].copy() if flag.any() else t.copy()
    f["dev"] = f["eb_log_or"] - mu
    top = pd.concat([f.sort_values("dev", ascending=False).head(12),
                     f.sort_values("dev").head(12).iloc[::-1]]).drop_duplicates(namecol)
    top = top.sort_values("dev")
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(mu, color=INK2, lw=1, zorder=1)
    for i, r in enumerate(top.itertuples()):
        se = math.sqrt(r.eb_var)
        col = BLUE if r.dev > 0 else ORANGE
        ax.plot([r.eb_log_or - Z * se, r.eb_log_or + Z * se], [i, i], color=col, lw=2,
                solid_capstyle="round", zorder=2)
        ax.plot(r.eb_log_or, i, "o", color=col, ms=6, mec="white", mew=1.2, zorder=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([pretty(n) for n in top[namecol]], fontsize=7, color=INK)
    lo = min(top["eb_log_or"] - Z * np.sqrt(top["eb_var"])) - 0.1
    hi = max(top["eb_log_or"] + Z * np.sqrt(top["eb_var"])) + 0.1
    ax.set_xlim(lo, hi)
    logticks(ax, [v for v in (0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8) if lo < math.log(v) < hi])
    ax.set_xlabel("Shrunken odds ratio, women vs men", loc="right")
    ax.set_title("b  Largest flagged deviations", loc="right")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_n2(res, out):
    n2 = res["N2"]
    va = n2["validity_age"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2]})
    ax = axes[0]
    bands = ["18-49", "50-64", "65+"]
    bs = res["N2"].get("ch_share_by_age_sex")
    if bs:
        for sex, col in (("female", BLUE), ("male", ORANGE)):
            ys = [100 * bs[sex][b] for b in bands]
            ax.plot(range(3), ys, "-o", color=col, lw=2, ms=7, mec="white", mew=1.5,
                    label="Women" if sex == "female" else "Men")
            ax.annotate(f"{ys[-1]:.0f}%", (2, ys[-1]), xytext=(7, -3), textcoords="offset points",
                        color=INK, fontsize=7)
    else:
        ys = [100 * va[b]["ch_share"] for b in bands]
        ax.plot(range(3), ys, "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["18 to 49", "50 to 64", "65 and over"])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylabel("Cholestatic share (%)")
    ax.set_xlabel("Age (years)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("a  Cholestatic share by age and sex")
    if bs:
        ax.legend(loc="upper left", fontsize=7)
    ax = axes[1]
    ad = n2["adult_set"]
    rows = [("Crude", ad["crude"]), ("Stratified by age band", ad["mh_age"]),
            ("Stratified by drug", ad["mh_ps_drug"]),
            ("Stratified by drug and age band", ad["mh_ps_drug_age"]),
            ("Drug, age, period, region, reporter", ad["mh_full"])]
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    for i, (lab, e) in enumerate(rows[::-1]):
        whisk(ax, i, e, BLUE)
        ax.annotate(f"{e['or']:.2f} ({e['lo']:.2f} to {e['hi']:.2f})", (1.0, i),
                    xycoords=("axes fraction", "data"), xytext=(8, -2.5), textcoords="offset points",
                    color=INK, fontsize=7, ha="left", annotation_clip=False)
    ax.annotate("OR (95% CI)", (1.0, len(rows) - 0.45), xycoords=("axes fraction", "data"), xytext=(8, 0),
                textcoords="offset points", color=INK2, fontsize=7, ha="left", annotation_clip=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], color=INK, fontsize=7)
    los = min(math.log(e["lo"]) for _, e in rows) - 0.05
    his = max(math.log(e["hi"]) for _, e in rows) + 0.05
    ax.set_xlim(min(los, -0.05), max(his, 0.08))
    ax.set_ylim(-0.6, len(rows) - 0.2)
    logticks(ax, [v for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25) if ax.get_xlim()[0] < math.log(v) < ax.get_xlim()[1]])
    ax.set_xlabel("Odds ratio, women vs men\n(cholestatic vs hepatocellular)")
    ax.set_title("b  Sex odds ratio by adjustment (adults)")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_n2_phenotype.pdf"))
    plt.close(fig)


def fig_n3(res, out):
    n3 = res["N3"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    blocks = [("All reports, stratified by drug", n3["sex_set_ps_drug"]),
              ("Adults, fully stratified", n3["adult_full"])]
    y, yt, yl = 0, [], []
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    for lab, b in blocks[::-1]:
        for name, key, col, mk in [("Ratio (hepatic / other)", "calibrated_mh", INK, "D"),
                                   ("Other reports, same drugs", "nonhepatic_mh", ORANGE, "o"),
                                   ("Hepatic injury reports", "hepatic_mh", BLUE, "o")]:
            e = b[key]
            whisk(ax, y, e, col, mk)
            ax.annotate(f"{e['or']:.2f} ({e['lo']:.2f} to {e['hi']:.2f})", (1.0, y),
                        xycoords=("axes fraction", "data"), xytext=(8, -2.5), textcoords="offset points",
                        color=INK, fontsize=7, ha="left", annotation_clip=False)
            yt.append(y)
            yl.append(name)
            y += 1
        ax.annotate(lab, (0, y - 0.3), xycoords=("axes fraction", "data"), color=INK2,
                    fontsize=7, fontweight="bold", xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", annotation_clip=False)
        y += 0.9
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, color=INK, fontsize=7)
    ax.set_ylim(-0.6, y - 0.3)
    xs = [math.log(b[k][s]) for _, b in blocks for k in ("calibrated_mh", "nonhepatic_mh", "hepatic_mh")
          for s in ("lo", "hi")]
    ax.set_xlim(min(xs + [-0.05]) - 0.05, max(xs + [0.05]) + 0.05)
    ax.annotate("OR (95% CI)", (1.0, y - 1.2), xycoords=("axes fraction", "data"), xytext=(8, 0),
                textcoords="offset points", color=INK2, fontsize=7, ha="left", va="center", annotation_clip=False)
    logticks(ax, [v for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3) if ax.get_xlim()[0] < math.log(v) < ax.get_xlim()[1]])
    ax.set_xlabel("Odds ratio of a fatal outcome, women vs men (log scale)")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_n3_fatal.pdf"))
    plt.close(fig)


TERM_LABELS = {
    "C(age_grp, Treatment('18-49'))[T.50-64]": "Age 50 to 64 (vs 18 to 49)",
    "C(age_grp, Treatment('18-49'))[T.65+]": "Age 65 or older",
    "C(ning_band, Treatment('1'))[T.2-4]": "2 to 4 drugs (vs 1)",
    "C(ning_band, Treatment('1'))[T.5-9]": "5 to 9 drugs",
    "C(ning_band, Treatment('1'))[T.10+]": "10 or more drugs",
    "C(dilirank, Treatment('no'))[T.less]": "DILIrank less concern (vs no concern)",
    "C(dilirank, Treatment('no'))[T.most]": "DILIrank most concern",
    "C(dilirank, Treatment('no'))[T.ambiguous]": "DILIrank ambiguous",
    "C(dilirank, Treatment('no'))[T.unlisted]": "Not in DILIrank",
    "C(reporter, Treatment('HP'))[T.consumer]": "Consumer (vs health professional)",
    "C(reporter, Treatment('HP'))[T.lawyer]": "Lawyer",
    "C(reporter, Treatment('HP'))[T.missing]": "Reporter type not recorded",
    "C(region, Treatment('US'))[T.nonUS]": "Outside the United States",
    "C(period, Treatment('2014-2017'))[T.2018-2021]": "2018 to 2021 (vs 2014 to 2017)",
    "C(period, Treatment('2014-2017'))[T.2022+]": "2022 onward",
    "alcohol_proxy": "Alcohol proxy",
    "liverdx_proxy": "Prior liver disease proxy",
}


def fig_n4(rdir, out):
    t = pd.read_csv(os.path.join(rdir, "le20", "n4_models.csv"))
    terms = [k for k in TERM_LABELS if ((t["model"] == "women") & (t["term"] == k)).any()]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    inter = t[t["model"] == "interaction"].set_index("term")
    for i, k in enumerate(terms[::-1]):
        for sex, col, dy in (("women", BLUE, 0.16), ("men", ORANGE, -0.16)):
            r = t[(t["model"] == sex) & (t["term"] == k)].iloc[0]
            ax.plot([math.log(r["lo"]), math.log(r["hi"])], [i + dy, i + dy], color=col, lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(math.log(r["or"]), i + dy, "o", color=col, ms=6, mec="white", mew=1.2,
                    zorder=3, label=("Women" if sex == "women" else "Men") if i == 0 else None)
        if k in inter.index and inter.loc[k, "q"] < 0.05:
            ax.annotate("*", (1.005, i), xycoords=("axes fraction", "data"), color=INK,
                        fontsize=10, va="center")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([TERM_LABELS[k] for k in terms[::-1]], color=INK, fontsize=7)
    xl = ax.get_xlim()
    logticks(ax, [v for v in (0.1, 0.2, 0.33, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20) if xl[0] < math.log(v) < xl[1]])
    ax.set_xlabel("Adjusted odds ratio of a hepatic injury report (log scale, 95% interval)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_n4_correlates.pdf"))
    plt.close(fig)


def fig_flow(rdir, out):
    root = os.path.dirname(rdir.rstrip("/"))
    fm = json.load(open(os.path.join(root, "merged", "flow.json")))
    fc = json.load(open(os.path.join(root, "cubes", "flow.json")))
    P = json.load(open(os.path.join(rdir, "primary", "results.json")))
    R = json.load(open(os.path.join(rdir, "le20", "results.json")))
    f = lambda n: f"{int(n):,}"
    boxes = [
        f"Records in {len(fm['quarters'])} quarterly files (all case versions)\nn = {f(fm['records_all_versions'])}",
        f"Latest version per case, FDA-deleted cases removed\nn = {f(fm['reports_after_caseid_dedup'])}",
        f"Sex recorded as female or male\nn = {f(fc['sex_known'])}",
        f"Full set (overdose-context reports excluded)\nn = {f(P['n_reports'])}, hepatic injury reports {f(P['n_cases'])}",
        f"Restricted set (at most 20 suspect ingredients, post hoc)\nn = {f(R['n_reports'])}, hepatic injury reports {f(R['n_cases'])}",
    ]
    side = [
        f"Earlier versions {f(fm['records_all_versions'] - fm['distinct_caseids'])}\nCases on FDA deletion lists {f(fm['distinct_caseids'] - fm['reports_after_caseid_dedup'])}",
        f"Sex missing or other code {f(fc['reports_deduplicated'] - fc['sex_known'])}",
        f"Overdose-context terms {f(fc['overdose_context'])}",
        f"More than 20 suspect ingredients {f(P['n_reports'] - R['n_reports'])}",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ys = [9.1, 7.1, 5.1, 3.1, 1.1]
    for y, txt in zip(ys, boxes):
        ax.text(3.3, y, txt, ha="center", va="center", fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=INK2, lw=0.8))
    for i in range(4):
        ax.annotate("", xy=(3.3, ys[i + 1] + 0.62), xytext=(3.3, ys[i] - 0.62),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
        ym = (ys[i] + ys[i + 1]) / 2
        ax.annotate("", xy=(6.3, ym), xytext=(3.3, ym), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
        ax.text(6.45, ym, "Excluded: " + side[i], ha="left", va="center", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig0_flow.pdf"))
    plt.close(fig)


def main(rdir, out):
    os.makedirs(out, exist_ok=True)
    fig_flow(rdir, out)
    res = json.load(open(os.path.join(rdir, "le20", "results.json")))
    fig_n1(json.load(open(os.path.join(rdir, "le20_ps", "results.json"))), rdir, out)
    fig_stratum(rdir, out)
    fig_n2(res, out)
    fig_n3(res, out)
    fig_n4(rdir, out)
    print("figures written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
