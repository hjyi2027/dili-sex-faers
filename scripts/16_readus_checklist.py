"""READUS-PV checklist with section and page references taken from the compiled manuscript.

Usage: python3 scripts/16_readus_checklist.py <paper_dir> <version_label>
Reads <paper_dir>/main.pdf (pdftotext must be installed) and writes <paper_dir>/readus_checklist.tex.
Item descriptions are abbreviated from Fusaroli et al., Drug Saf 2024;47:575-584 (Tables 1 and 2).
Status codes: R reported, P partly reported, NA not applicable, A to be completed by the submitting author.
"""
import re
import subprocess
import sys

HEAD = {
    "T": None, "A": None,
    "1": r"1\s+Introduction", "2.1": r"2\.1\s+Design, plan and reporting", "2.2": r"2\.2\s+Data source",
    "2.3": r"2\.3\s+Report selection", "2.4": r"2\.4\s+Exposure", "2.5": r"2\.5\s+Outcomes",
    "2.6": r"2\.6\s+Covariates", "2.7": r"2\.7\s+Statistical analysis", "2.8": r"2\.8\s+Suspected duplicate reports",
    "2.9": r"2\.9\s+Checks of the reported phenotype", "2.10": r"2\.10\s+Robustness, sensitivity and null checks",
    "3.1": r"3\.1\s+Reports", "3.2": r"3\.2\s+Drug mix", "3.3": r"3\.3\s+Suspected duplicate reports",
    "3.4": r"3\.4\s+Reported phenotype", "3.5": r"3\.5\s+Fatal outcome", "3.6": r"3\.6\s+Drug-level scan",
    "3.7": r"3\.7\s+Missing age", "4.1": r"4\.1\s+Principal findings", "4.2": r"4\.2\s+Duplicate reports",
    "4.3": r"4\.3\s+Reported phenotype", "4.4": r"4\.4\s+Fatal outcome", "4.5": r"4\.5\s+Recommendations",
    "4.6": r"4\.6\s+Strengths and limitations", "5": r"5\s+Conclusions", "D": r"Declarations",
}
STATUS = {"R": "Reported", "P": "Partly reported", "NA": "Not applicable", "A": "Author to complete"}

MAIN = [
    ("1a", "Title: study type and data source", "R", ["T"], ""),
    ("1b", "Title: adverse event and drugs", "R", ["T"], "The event is named. The study covers all drugs."),
    ("2a", "Background: drug, event, frequency, existing knowledge", "P", ["1"],
     "No single drug is studied. The event is described as uncommon, without a frequency estimate."),
    ("2b", "Reason for the analysis", "R", ["1"], ""),
    ("2c", "Why report databases and disproportionality suit the question", "R", ["1"], ""),
    ("3", "Objectives, events, drugs, reference group, hypotheses", "R", ["1", "2.1"],
     "The plan posed three comparisons without directional hypotheses (Online Resource 1)."),
    ("4a", "Study identified as a disproportionality analysis, data type", "R", ["2.1"], ""),
    ("4b", "Design: primary, sensitivity and other analyses", "R", ["2.1", "2.7", "2.8", "2.9", "2.10"],
     "Planned and post hoc analyses are labelled."),
    ("5a", "Database, custodian, coverage, coding dictionaries", "R", ["2.2", "2.4"],
     "MedDRA version of each quarterly release. Drugs by the active ingredient field."),
    ("5b", "Extraction dates and pre-processing", "R", ["2.2", "2.3", "2.4"],
     "File names and checksums in Online Resource 2. Access date in the reference list."),
    ("6a", "Study population and restrictions", "R", ["2.3", "2.8"], ""),
    ("6b", "Key variables", "R", ["2.4", "2.5", "2.6"], ""),
    ("6c", "Drug and event groupings", "R", ["2.4", "2.5"],
     "Term lists in Online Resource 2. Strict definition and the FDA custom medical queries as checks (Sect. 3.2). "
     "Not compared with the SMQ (Sect. 4.6)."),
    ("6d", "Additional data sources", "R", ["2.6", "2.8", "2.10"],
     "DILIrank 2.0, matched by ingredient name. Canada Vigilance data extract for the check of the Canadian groups. FDA Custom Medical Queries 4.1 as a second case definition. Counts from Lu et al. in Online Resource 5."),
    ("7a", "Descriptive analysis", "R", ["2.6", "3.1"], "Table 1. No tests on descriptive data."),
    ("7b", "Disproportionality measure and thresholds", "R", ["2.7"],
     "Mantel-Haenszel odds ratio, women versus men, within primary suspect drug. No signal threshold, because "
     "the study does not detect signals. Nominal 5\\% procedure for the drug-level scan."),
    ("7c", "Sensitivity analyses and control of confounding", "R", ["2.7", "2.8", "2.9", "2.10"], ""),
    ("7d", "Case-by-case analysis: methods", "NA", ["2.1"], "Not performed, and stated."),
    ("7e", "Statistical methods for other data sources", "R", ["2.8", "2.10", "OR5"],
     "Matching of Canadian linked reports to Canada Vigilance. Case definition from the FDA queries. Comparison with Lu et al."),
    ("8a", "Number of reports at each stage", "R", ["3.1"], "Fig. 1."),
    ("8b", "Case characteristics against a reference group", "R", ["3.1"], "Table 1."),
    ("9", "Disproportionality results with intervals and sensitivity results", "R",
     ["3.2", "3.3", "3.4", "3.5", "3.6", "3.7"], "Tables 2 to 4, Figs. 2 to 4, Online Resource 5."),
    ("10", "Case-by-case analysis: results", "NA", [], "Not performed."),
    ("11", "Findings against objectives and literature", "R", ["4.1", "4.2", "4.3", "4.4"],
     "No drug-event signal is claimed, so the split into expected reactions and new signals does not apply."),
    ("12a", "Applicability to the general population", "R", ["4.6"], ""),
    ("12b", "Relevance for clinical practice", "R", ["5"], "Stated as none at present."),
    ("12c", "Further study designs", "R", ["4.6", "5"],
     "Registries with biochemical phenotype. Replication plan for a second database in Online Resource 1."),
    ("13", "Limitations", "R", ["4.6"], ""),
    ("14a", "Funding and role of sponsors", "R", ["D"], "No funding."),
    ("14b", "Conflicts of interest", "R", ["D"], "None declared."),
    ("14c", "Institutional approval", "R", ["D"], "Public, de-identified data. Approval not sought, and stated."),
    ("14d", "Data and code availability, software, protocol registration", "P", ["D", "2.1", "2.7"],
     "Scripts in Online Resource 3 and in a public repository (Declarations). The plan was not registered, and this is stated."),
]
ABSTRACT = [
    ("1a", "Aim or rationale", "R", ""),
    ("1b", "Adverse events and drugs", "R", "All drugs."),
    ("1c", "Population or setting", "R", "Women compared with men, all reports."),
    ("2a", "Disproportionality analysis and data type", "R", ""),
    ("2b", "Database and type of access", "R", "Public extracts."),
    ("2c", "Timeframe and region", "R", ""),
    ("2d", "Measures and thresholds", "R", "No signal threshold applies."),
    ("2e", "Case-by-case analysis performed or not", "R", "Not performed, and stated."),
    ("3", "Main findings with precision", "P",
     "Intervals for the main odds ratios and the fatality ratios. Absolute contrasts and two country estimates "
     "without intervals (in Tables 3 and 4)."),
    ("4a", "Key conclusions", "R", ""),
    ("4b", "Hypothesis generating", "R", ""),
    ("4c", "Implications and clinical relevance", "R", ""),
]


def pages(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    return [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), pdf, "-"], capture_output=True, text=True).stdout
            for i in range(1, n + 1)]


def main(pdir, label):
    txt = pages(f"{pdir}/main.pdf")
    where = {"T": 1, "A": 1}
    for k, rx in HEAD.items():
        if rx is None:
            continue
        hits = [i + 1 for i, t in enumerate(txt) if re.search(r"(?m)^" + rx, t)]
        assert hits, f"heading not found: {k}"
        where[k] = hits[0] if k != "D" else hits[-1]

    def loc(keys):
        if not keys:
            return ""
        names = {"T": "Title", "A": "Abstract", "D": "Declarations", "OR5": "Online Resource 5"}
        secs = ", ".join(names.get(k, "Sect. " + k) for k in keys)
        pg = sorted({where[k] for k in keys if k in where})
        return f"{secs} (p. {', '.join(map(str, pg))})" if pg else secs

    out = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=2cm]{geometry}", r"\usepackage{longtable,booktabs,array}",
           r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}", r"\setlength{\parindent}{0pt}", r"\begin{document}",
           r"{\large\bfseries READUS-PV checklist}\par\medskip",
           r"Manuscript: Drug mix, suspected duplicate reports and country-specific coding in sex comparisons of "
           r"drug-induced liver injury reports: a disproportionality analysis of the "
           r"FDA Adverse Event Reporting System, 2014 to 2026.\par\smallskip",
           f"Completed for manuscript version {label}. Item descriptions are abbreviated from Fusaroli et al., "
           r"Drug Saf 2024;47:575--584.\par\bigskip",
           r"\textbf{Main text}\par\smallskip", r"\small",
           r"\begin{longtable}{@{}p{0.7cm}>{\raggedright\arraybackslash}p{4.0cm}>{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{4.7cm}@{}}", r"\toprule",
           r"Item & Topic & Status & Where reported & Note \\", r"\midrule", r"\endhead"]
    for item, topic, st, keys, note in MAIN:
        out.append(f"{item} & {topic} & {STATUS[st]} & {loc(keys)} & {note} \\\\")
    out += [r"\bottomrule", r"\end{longtable}", r"\normalsize\textbf{Abstract}\par\smallskip\small",
            r"\begin{longtable}{@{}p{0.7cm}>{\raggedright\arraybackslash}p{5.6cm}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{6.9cm}@{}}", r"\toprule", r"Item & Topic & Status & Note \\",
            r"\midrule", r"\endhead"]
    for item, topic, st, note in ABSTRACT:
        out.append(f"{item} & {topic} & {STATUS[st]} & {note} \\\\")
    out += [r"\bottomrule", r"\end{longtable}", r"\end{document}"]
    open(f"{pdir}/readus_checklist.tex", "w").write("\n".join(out) + "\n")
    n = {s: sum(1 for r in MAIN if r[2] == s) for s in STATUS}
    print("readus checklist written:", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
