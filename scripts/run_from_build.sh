#!/bin/bash
# Steps 03 onward, from an output root that already holds q/ and merged/.
# Usage: scripts/run_from_build.sh <output_root>   (run from the project directory)
set -e
ROOT=$1
mkdir -p "$ROOT/paper"
python3 scripts/03_build.py "$ROOT/merged" "$ROOT/analysis"
python3 scripts/03b_indication.py "$ROOT/q" "$ROOT/analysis"
python3 scripts/03c_linkage.py data "$ROOT/analysis" "$ROOT/merged"
python3 scripts/04_cubes.py "$ROOT/analysis" "$ROOT/cubes"
python3 scripts/04_cubes.py "$ROOT/analysis" "$ROOT/cubes_perm" --permute-sex 20261123
for v in primary le20 le10 le5 no_ca le20_hp le20_ps ps_only hp us broad with_overdose dedup2 no_top linked linked9 linked_ps linked_first; do
  python3 scripts/05_analyze.py "$ROOT/cubes" "$ROOT/results" $v
done
python3 scripts/05_analyze.py "$ROOT/cubes_perm" "$ROOT/results_perm" primary
python3 scripts/06_replication.py "$ROOT/results" lit/lu2023_table8.csv
python3 scripts/09_cluster_sources.py data "$ROOT/analysis" "$ROOT/results/cluster_sources.json"
python3 scripts/11_robustness.py "$ROOT/analysis" "$ROOT/merged" "$ROOT/results"
python3 scripts/13_null_sims.py "$ROOT/analysis" "$ROOT/results" 200 1000
python3 scripts/14_dependence.py "$ROOT/analysis" "$ROOT/results" 200
python3 scripts/15_stratum_linked.py "$ROOT/analysis" "$ROOT/results"
python3 scripts/17_multi_ps.py data "$ROOT/analysis" "$ROOT/results"
python3 scripts/18_definition_checks.py data "$ROOT/analysis" "$ROOT/results"
# checks against external files (change 31). 19 needs the unpacked Health Canada extract, see the README.
python3 scripts/19_canada_vigilance.py data "$ROOT/analysis" data/ext/canada_vigilance/cvponline_extract_20260731 "$ROOT/results"
python3 scripts/20_ocmq_definition.py data "$ROOT/analysis" data/ext/ocmq_v4_1_hepatic_queries.csv "$ROOT/results"
python3 scripts/21_dilirank_check.py "$ROOT/analysis" "$ROOT/results"
python3 scripts/07_figures.py "$ROOT/results" "$ROOT/paper/figs"
python3 scripts/08_tables.py "$ROOT/results" "$ROOT/cubes" "$ROOT/merged" "$ROOT/paper"
echo RUN_DONE
