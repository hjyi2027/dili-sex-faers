# Sex comparisons of drug-induced liver injury reports in FAERS, 2014 to 2026

Code, analysis plan, term lists and result files for the paper

Daina Oh, Hyunjun Yi (Seoul International School). Drug mix, suspected duplicate reports and country-specific coding in sex comparisons of drug-induced liver injury reports: a disproportionality analysis of the FDA Adverse Event Reporting System, 2014 to 2026.

Manuscript version: 3.23 (27 November 2026). Not yet peer reviewed.

Licence: MIT for all code (file `LICENSE`), CC BY 4.0 for the plan, term lists and result files, third-party files on their sources' terms (details in `LICENSE-CONTENT.md`).

Use of a large language model: the literature search, plan, code, analyses and drafts were produced with Claude (Anthropic) under the authors' direction. See Sect. 2.11 of the paper and changes 36 to 39 of `plan/analysis_plan_and_change_log.md`.

## What is here

| Folder | Content |
|---|---|
| `plan/` | Analysis plan with its dated change log, replication plan for a second database with the plan of the Canada Vigilance check, checks of the estimators on simulated data |
| `terms/` | Preferred-term table with all flags, ingredient synonym table, term rules, FAERS file manifest with SHA-256 checksums, comparison with the FDA query terms |
| `scripts/` | All analysis scripts, numbered in the order in which they run |
| `inputs/` | The three input files that do not come from FAERS |
| `results/` | Result files for every analysis set |
| `requirements.txt` | Python and package versions |

## Data

No report-level data are stored here. All sources are public.

- FAERS quarterly data extract, ASCII format, 2014Q3 to 2026Q3 (49 files), from the FDA. `scripts/00_download.py` fetches them. File names and SHA-256 checksums are in `terms/faers_file_manifest_sha256.csv`.
- Canada Vigilance data extract, file `extract_extrait.zip` on the Health Canada data extract page. Version used: data to 2026-07-31, 361,045,238 bytes, SHA-256 `b13e1690e11840b0f26b04e408e237ee694a8db84c1a36e4f3688e78352c11a4`. Health Canada updates the file monthly, so a later download gives other counts.
- FDA Office of New Drugs Custom Medical Queries, version 4.1. `inputs/ocmq_v4_1_hepatic_queries.csv` holds the rows of the two hepatic queries. SHA-256 of the FDA workbook: `c3390eae7a4477f7433cbeff97c686533f4edd77a3a0e39974afdf35e3af1330`.
- FDA DILIrank 2.0. `inputs/dilirank.csv` is sheet "version 2" of the FDA workbook. SHA-256 of the workbook: `1ca1352ff727af68e68e250eae2ed775bca8492335140ac0afd2233248694993`.
- `inputs/lu2023_table8.csv`: counts from Table 8 of Lu, Suzuki and Wang, BMC Med Res Methodol 2023;23:71.

## Rerun

1. Make a project folder with `scripts/` from this repository.
2. Put the FAERS zip files in `data/`, named as in the manifest.
3. Copy `inputs/dilirank.csv` and `inputs/ocmq_v4_1_hepatic_queries.csv` to `data/ext/`, and `inputs/lu2023_table8.csv` to `lit/`.
4. Unpack the Canada Vigilance extract to `data/ext/canada_vigilance/cvponline_extract_20260731`.
5. Run `bash scripts/run_all.sh <output folder>` from the project folder.

The run needs about 7 GB of memory and reproduces every number in the paper.

## Plan and its history

The plan was written before any FAERS file was downloaded. It was not registered. `plan/analysis_plan_and_change_log.md` holds the plan and every later change with its date, its reason and whether outcome data had been seen. Analyses added after the first results are labelled post hoc in the plan and in the paper. `plan/replication_plan.md` was frozen before any second database was opened. Its section 6 and addendum cover the Canada Vigilance check.

## Limits

The estimates are reporting associations from spontaneous reports. They are not incidences or risks. Reports were not adjudicated. See the limitations section of the paper.
