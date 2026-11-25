# Replication plan for a second database (frozen 2026-11-24, before any second-database file was opened)

Status: plan only. No second database has been downloaded or analysed. Database facts marked [confirm] must be checked against the current file description before the plan is used.

## 1. Purpose

Test whether four results from FAERS 2014Q3 to 2026Q3 hold in an independent spontaneous report database. First choice: JADER (Japan). Reason: Japanese reports are a small share of the FAERS phenotype data, so overlap is limited. Second choice: VigiBase, which needs a data agreement and contains the FAERS reports, so it is a pooled analysis and not an independent one.

## 2. Hypotheses and decision rules (fixed now)

H1, drug mix. The crude odds ratio (OR) of a hepatic injury report, women versus men, lies below the OR stratified by primary suspect drug.
Replicated if the stratified OR is closer to 1 than the crude OR and the 95% confidence interval of the stratified OR excludes the crude point estimate.

H2, phenotype. Among reports with a hepatocellular or cholestatic term, women's reports name a cholestatic term less often within drug and age band. FAERS estimate: OR 0.89 (0.84 to 0.94).
Replicated if the 95% confidence interval lies below 1. Contradicted if it lies above 0.95. Otherwise undecided.
Power check before unblinding by sex: FAERS gave a standard error of 0.029 for the log OR from 26,785 adult reports. An OR of 0.89 needs a standard error of about 0.042 for 80% power at a two-sided 5% level, which is about 12,700 adult reports with a hepatocellular or cholestatic term if strata are as sparse as in FAERS. With fewer, state in advance that H2 cannot be tested and report the estimate as descriptive.

H3, fatality on the ratio scale. The fatality OR (women versus men) among hepatic reports divided by the fatality OR among other reports of the same drugs is above 1. FAERS estimate: 1.18 by drug, 1.13 fully stratified.
Replicated if the 95% confidence interval of the ratio lies above 1.

H4, fatality on the difference scale. The difference in differences of fatal share changes sign or loses its interval's exclusion of zero when the standard population is switched from hepatic reports to other reports.
Replicated if the two standards give intervals of which one includes zero and the other does not, or point estimates of opposite sign.

H5, duplicates. Reports with at least 11 suspect ingredients contain groups of near-identical reports (same sex, same age, Jaccard index of suspect lists at least 0.8).
Report the share of such reports in a group. No pass or fail rule. A national database with one reporting route may have few.

No other hypothesis will be tested for replication. Everything else is descriptive.

## 3. Definitions carried over unchanged

- Hepatic injury: the narrow preferred-term list of Online Resource 2. Map by MedDRA preferred-term code. JADER uses MedDRA/J, which shares codes with MedDRA [confirm].
- Phenotype: the hepatocellular, cholestatic and mixed term lists of Online Resource 2. Run H2 twice: all terms, and without reports whose only hepatocellular term is hepatic cytolysis.
- Exclusion: overdose, poisoning, self-harm, misuse and abuse terms.
- Exposure: active ingredient. Stratum: the first-listed suspect drug of each report. JADER has no primary suspect role [confirm]. Use the suspected drug with the lowest sequence number and repeat with single-suspect reports only.
- Fatal outcome: JADER records an outcome for each reaction [confirm]. Fatal means any reaction on the report with the outcome death. Repeat with death recorded for a hepatic reaction only. FAERS could not make that distinction.
- Age: JADER gives age in decades [confirm]. Bands: under 50, 50 to 69, 70 or older. This differs from FAERS and is fixed here.
- Covariates for full stratification: age band, reporting period in three equal spans, reporter type if available.

## 4. Analysis

Same estimators as the FAERS analysis: Mantel-Haenszel OR with Robins-Breslow-Greenland variance, direct standardization for the difference in differences, record linkage as in scripts/03c_linkage.py without the country block. Use the same scripts with a new reader. Report the share of reports in informative strata for each estimate.

Order of work: (1) build the reader and the term mapping without counts by sex, (2) run the power check, (3) freeze this file's hash in the report, (4) run H1 to H5 once.

## 5. Reporting

Report every hypothesis with its rule and outcome, including failures. Deviations from this plan are listed with reasons. A temporal split of FAERS is not a replication, because the FAERS analyses were chosen with all periods in view.

## 6. Canada Vigilance check of the Canadian linked groups (added 2026-11-25, before the extract was opened)

Purpose: test whether the FAERS linked groups whose reports come from Canada are single cases in the Canadian national database. This check concerns the suspected duplicates only. Sections 1 to 5 cover the sex contrasts.

Data: Health Canada data extract of the Canada Vigilance adverse reaction online database (one zip file, 344 MB, data to 2026-07-31 as stated on the Health Canada page on 2026-11-25). File and field names below were taken from the Health Canada data structure page and are marked [confirm] until checked against the files: Reports.txt (REPORT_ID, REPORT_NO, MAH_NO, COMPANY_NUMB, AUTHORITY_NUMB, E2B_IMP_SAFETYREPORT_ID, GENDER, AGE_Y, WEIGHT, DATRECEIVED), Report_Links_LX.txt (REPORT_ID, RECORD_TYPE, REPORT_LINK_NO), Report_Drug.txt, Drug_Product_Ingredients.txt, Reactions.txt.

Step 1, key match. FAERS reports in linked groups (Jaccard index at least 0.8) with country Canada are matched to Canada Vigilance on the manufacturer control number (FAERS mfr_num against MAH_NO, COMPANY_NUMB and E2B_IMP_SAFETYREPORT_ID), after upper-casing and removal of non-alphanumeric characters. Report the share matched. No outcome is used.

Step 2, classification. For each FAERS group with two or more matched reports: confirmed single case if all matched reports map to one REPORT_ID or to reports that Health Canada links to each other. Not confirmed if they map to two or more reports that Health Canada does not link. Report both shares with counts, overall and by group size (2, 3 to 9, 10 to 99, 100 or more).

Step 3, same rule inside Canada Vigilance. Apply the linkage rule of the paper (at least 11 suspect ingredients, same sex and age, Jaccard index at least 0.8, complete linkage) to Canada Vigilance. Report the share of eligible reports in a group and the share of linked pairs that Health Canada also links.

Reading fixed now. At least 80% of checkable groups confirmed: the paper may say that most Canadian groups that could be checked are single cases held under several manufacturer reports. Below 50%: the paper says the groups are not shown to be single cases. Between: report as mixed. If step 1 matches under 30% of the Canadian linked reports, the check is reported as inconclusive and only step 3 is interpreted.

Addendum to section 6 (2026-11-25, after the extract was unpacked and its file layout read, before any match between FAERS and Canada Vigilance was computed). Extract: extract_extrait.zip from the Health Canada data extract page, 361,045,238 bytes, SHA-256 b13e1690e11840b0f26b04e408e237ee694a8db84c1a36e4f3688e78352c11a4, folder cvponline_extract_20260731, 1,280,288 reports, data to 2026-07-31.
- Layout as found. reports.txt has 42 columns, three more than the data structure page lists. Column 6 is MAH_NO. Columns 40 and 42 hold manufacturer case numbers of the form CA-SENDER-NUMBER and column 41 holds an authority number of the form CA-HEALTHCANVIG-NUMBER. The match targets of step 1 are columns 6, 40 and 42. report_links.txt has five columns: link row identifier, REPORT_ID, record type in English (Duplicate or Linked) and in French, and a link number that is the REPORT_NO of another report. The file is named report_links.txt, not Report_Links_LX.txt.
- Step 1: equality of the whole key after normalization. Keys shorter than 6 characters are ignored. A FAERS key that equals the keys of several Canada Vigilance reports keeps all of them.
- Step 2: a link row joins its report to the report whose REPORT_NO equals the link number, or to the link number itself when that report is not in the extract. Rows without a link number join nothing. Reports are linked to each other when they fall in one connected component of these joins. Both record types count, as section 6 says "links". The result with Duplicate records only is given beside it.
- Step 3: suspect role is the value Suspect of the drug involvement field. Ingredients are the active ingredients of the product, normalized by the rule of the paper, and the product name where no ingredient is listed. Reports need a recorded sex. The extract has no country field, so country is not a blocking variable. The overdose exclusion of the paper is not applied, because step 3 uses no reaction terms.
- Not in section 6, reported separately and not used for the reading: match of the FAERS authority number (auth_num) against column 41.
