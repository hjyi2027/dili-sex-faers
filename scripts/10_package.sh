#!/bin/bash
# Assemble the online resources and the submission source package.
# Usage: scripts/10_package.sh <output_root> [deliver_dir]   (output_root holds results, merged, analysis)
set -e
ROOT=$1
# Claim lock and style checks: the build stops if a summary loses the hedge on the phenotype result.
python3 scripts/12_flatten.py paper main.tex /tmp/dili_main_flat_check.tex >/dev/null
python3 scripts/22_word_count.py /tmp/dili_main_flat_check.tex paper
D=${2:-deliver_new}; mkdir -p $D; S=$D/supplement
mkdir -p $S/OnlineResource1 $S/OnlineResource2 $S/OnlineResource3/results $S/OnlineResource3/scripts $S/OnlineResource4 $S/OnlineResource5 $D/source
cp PROTOCOL.md $S/OnlineResource1/analysis_plan_and_change_log.md
cp REPLICATION_PLAN.md $S/OnlineResource1/replication_plan.md
python3 scripts/validate_estimators.py > $S/OnlineResource1/estimator_checks.txt
cp "$ROOT/analysis/terms.csv" $S/OnlineResource2/preferred_term_table.csv
cp scripts/ingredient_merge.csv $S/OnlineResource2/ingredient_synonyms.csv
cp scripts/lib.py $S/OnlineResource2/term_rules_and_normalization_lib.py
cp data/manifest.csv $S/OnlineResource2/faers_file_manifest_sha256.csv
cp "$ROOT/analysis/liver_indication_terms.csv" $S/OnlineResource2/prior_liver_disease_indication_terms.csv
for v in primary le20 le10 le5 no_ca le20_hp le20_ps ps_only hp us broad with_overdose dedup2 no_top linked linked9 linked_ps linked_first; do
  mkdir -p $S/OnlineResource3/results/$v; cp "$ROOT/results/$v/"*.csv "$ROOT/results/$v/results.json" $S/OnlineResource3/results/$v/
done
mkdir -p $S/OnlineResource3/results/permuted_sex; cp "$ROOT/results_perm/primary/results.json" "$ROOT/results_perm/primary/"*.csv $S/OnlineResource3/results/permuted_sex/
cp "$ROOT/results/cluster_sources.json" "$ROOT/results/replication_lu2023.json" "$ROOT/results/robustness.json" "$ROOT/results/robustness_countries.csv" $S/OnlineResource3/results/
cp "$ROOT/results/null_sims.json" $S/OnlineResource3/results/
cp "$ROOT/results/dependence.json" "$ROOT/results/stratum_scan.csv" "$ROOT/results/stratum_scan_sets.json" "$ROOT/results/stratum_scan_restricted.csv" "$ROOT/results/multi_ps.json" $S/OnlineResource3/results/
cp "$ROOT/analysis/link_summary.json" $S/OnlineResource3/results/record_linkage_summary.json
cp paper/supp_correlates.pdf $S/OnlineResource4/exploratory_models.pdf
cp paper/supp_tables.pdf $S/OnlineResource5/additional_tables.pdf
cp "$ROOT/merged/flow.json" $S/OnlineResource3/results/flow_merge.json; cp "$ROOT/cubes/flow.json" $S/OnlineResource3/results/flow_cubes.json
cp lit/lu2023_table8.csv $S/OnlineResource3/results/
mkdir -p $S/OnlineResource3/inputs; cp data/ext/dilirank.csv lit/lu2023_table8.csv data/ext/ocmq_v4_1_hepatic_queries.csv $S/OnlineResource3/inputs/
python3 - "$ROOT" $S <<'PY'
import json, sys, csv
root, S = sys.argv[1], sys.argv[2]
r = json.load(open(f"{root}/results/le20/results.json"))
with open(f"{S}/OnlineResource3/results/anchor_drugs_restricted_set.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ingredient", "expected_pattern", "role_definition", "cholestatic_reports", "hepatocellular_plus_cholestatic_reports", "cholestatic_share"])
    for key, lab in (("validity_anchors_ps", "primary suspect only"), ("validity_anchors", "primary or secondary suspect")):
        for k, v in r["N2"][key]["drugs"].items():
            w.writerow([k, v["expected"], lab, int(v.get("ch", 0)), int(v.get("n", 0)), "" if not v.get("n") else round(v["ch_share"], 4)])
em = r["N1"]["posthoc_effect_modification"]
with open(f"{S}/OnlineResource3/results/reporting_or_subgroups_restricted_set.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["subgroup", "hepatic_reports", "fully_stratified_or", "lo", "hi"])
    for k, v in em.items(): w.writerow([k, v["n_cases"], round(v["full"]["or"], 4), round(v["full"]["lo"], 4), round(v["full"]["hi"], 4)])
PY
cp scripts/*.py scripts/*.sh scripts/ingredient_merge.csv $S/OnlineResource3/scripts/
python3 - > $S/OnlineResource3/requirements.txt <<'PY'
import sys, pandas, numpy, scipy, statsmodels, pyarrow, matplotlib
print(f"# Python {sys.version.split()[0]}")
for m in (pandas, numpy, scipy, statsmodels, pyarrow, matplotlib):
    print(f"{m.__name__}=={m.__version__}")
PY
cp "$ROOT/analysis/sex_specific_terms.csv" $S/OnlineResource2/pregnancy_obstetric_and_sex_linked_terms.csv
cp "$ROOT/results/definition_checks.json" $S/OnlineResource3/results/
cp "$ROOT/results/canada_vigilance.json" "$ROOT/results/canada_vigilance_groups.csv" "$ROOT/results/ocmq_definition.json" "$ROOT/results/dilirank_check.json" $S/OnlineResource3/results/
cp "$ROOT/results/ocmq_term_comparison.csv" $S/OnlineResource2/fda_query_term_comparison.csv
cat > $S/README.txt <<'TXT'
Online Resource 1: analysis plan with change log (pre-specified amendments and post hoc changes), replication plan for a second database with the plan of the Canada Vigilance check and its addendum, checks of the estimators on simulated data.
Online Resource 2: preferred-term table with all flags, ingredient synonym table, term rules and ingredient normalization (lib.py), FAERS file manifest with SHA-256 checksums, indication terms used for the prior liver disease proxy, list of pregnancy, obstetric and sex-linked reaction terms used in a sensitivity analysis, comparison of the hepatic terms of the paper with those of the FDA queries (fda_query_term_comparison.csv, counts for both sexes together).
Online Resource 3: results for every analysis set (results.json and per-ingredient CSV files), including the record linkage sets (linked, linked9, linked_ps, linked_first), results with sex permuted, robustness results (robustness.json) with the table of phenotype-coded reports by country, record linkage summary, null simulation results (null_sims.json), the scan by primary suspect stratum (stratum_scan.csv, stratum_scan_restricted.csv with shrunken estimates, and stratum_scan_sets.json for the scan after record linkage), cluster bootstrap, duplicate handling variants and the joint fatality model (dependence.json), the check on reports with more than one primary suspect record (multi_ps.json), the checks of the case definition and of pregnancy, obstetric and sex-linked reports (definition_checks.json), the comparison with the FDA queries (ocmq_definition.json), the Canada Vigilance check (canada_vigilance.json and canada_vigilance_groups.csv), the check of the DILIrank input (dilirank_check.json), source fields of the reports with more than 20 suspect ingredients, comparison with Lu et al. 2023, anchor-drug table, subgroup table, and all analysis scripts. The folder inputs holds the three input files that do not come from FAERS: dilirank.csv (sheet version 2 of the FDA DILIrank 2.0 workbook, 1,336 drugs, downloaded from the FDA DILIrank 2.0 page on 2026-11-25, SHA-256 of the workbook 1ca1352ff727af68e68e250eae2ed775bca8492335140ac0afd2233248694993), ocmq_v4_1_hepatic_queries.csv (rows of the Hepatic Injury and Hepatic Failure queries from the sheet Consolidated List of the FDA Office of New Drugs Custom Medical Queries, version 4.1, downloaded from the FDA page on 2026-11-25, SHA-256 of the workbook c3390eae7a4477f7433cbeff97c686533f4edd77a3a0e39974afdf35e3af1330) and lu2023_table8.csv (counts from Table 8 of Lu et al. 2023). The Canada Vigilance data extract is not included (344 MB). It is the file extract_extrait.zip on the Health Canada data extract page (version used: data to 2026-07-31, 361,045,238 bytes, SHA-256 b13e1690e11840b0f26b04e408e237ee694a8db84c1a36e4f3688e78352c11a4). Health Canada updates it monthly, so a later download will give other counts. To rerun: make a project folder with scripts/ (from this resource), data/ with the FAERS quarterly zip files named as in the manifest of Online Resource 2 (scripts/00_download.py fetches them), data/ext/dilirank.csv, data/ext/ocmq_v4_1_hepatic_queries.csv, lit/lu2023_table8.csv and the unpacked Canada Vigilance extract in data/ext/canada_vigilance/cvponline_extract_20260731. Then run bash scripts/run_all.sh <output folder> from the project folder. It reproduces every number in the paper. requirements.txt gives the versions of Python and of the packages used.
Online Resource 4: exploratory logistic models of hepatic injury reporting in women and in men (methods, results, one figure, one table).
Online Resource 5: additional methods and results, with Fig. S1 (planned scan by ingredient) and Tables S2 to S14 (sensitivity analyses, record linkage checks, agreement by group size, informative strata, null simulations, phenotype subgroups, flagged ingredients, cluster bootstrap, duplicate handling variants, case definition and pregnancy, obstetric and sex-linked reports, comparison with the FDA queries, Canada Vigilance check by group size, FDA query terms by class).
TXT
(cd $D && rm -f supplement.zip && zip -q -r supplement.zip supplement)
# manuscript source and Online Resource sources in separate folders, each of which compiles on its own
MS=$D/source/manuscript; OR=$D/source/online_resources; CK=$D/source/checklist
mkdir -p $MS/tables $MS/figs $OR/tables $OR/figs $CK
cp paper/main.tex paper/main.bbl paper/numbers.tex paper/refs.bib paper/sn-jnl.cls paper/sn-vancouver.bst paper/author_contributions.tex $MS/
TP=$D/source/title_page; mkdir -p $TP; cp paper/title_page.tex $TP/; cp paper/title_page.pdf $D/title_page.pdf
cp paper/tables/table1.tex paper/tables/table2.tex paper/tables/table5.tex paper/tables/tableL.tex $MS/tables/
cp paper/figs/fig0_flow.pdf paper/figs/fig2_n2_phenotype.pdf paper/figs/fig3_n3_fatal.pdf paper/figs/fig4_stratum_scan.pdf $MS/figs/
cp paper/supp_correlates.tex paper/supp_tables.tex paper/numbers.tex $OR/
cp paper/tables/table3.tex paper/tables/table4.tex paper/tables/tableS1.tex $OR/tables/
cp paper/figs/fig1_n1_ingredients.pdf paper/figs/fig4_n4_correlates.pdf $OR/figs/
cp paper/readus_checklist.tex $CK/; cp paper/readus_checklist.pdf $D/READUS_PV_checklist.pdf
cat > $D/source/README.txt <<'TXT'
manuscript: main.tex with its bibliography (refs.bib, main.bbl), class and style files, numbers.tex (all numbers as macros), author_contributions.tex, four tables and four figures.
title_page: the separate title page (title, authors, affiliation, corresponding author, acknowledgments, author contributions, counts). Update its word count when the text changes (scripts/22_word_count.py).
  Printed Table 1 is tables/table1.tex, Table 2 is tables/table2.tex, Table 3 is tables/table5.tex and Table 4 is tables/tableL.tex.
  Fig. 1 is figs/fig0_flow.pdf, Fig. 2 is figs/fig2_n2_phenotype.pdf, Fig. 3 is figs/fig3_n3_fatal.pdf and Fig. 4 is figs/fig4_stratum_scan.pdf.
online_resources: supp_correlates.tex (Online Resource 4) and supp_tables.tex (Online Resource 5) with their tables and figures. They are submitted as PDF files.
checklist: READUS-PV checklist. Regenerate it last with scripts/16_readus_checklist.py, because it reads page numbers from the compiled manuscript.
TXT
(cd $D && rm -f source.zip && zip -q -r source.zip source)
ls -la $D | tail -5
