"""Shared definitions and estimators for the FAERS sex-difference analysis.

Everything that defines a case, an exposure or an estimator lives here so that
the term rules and the statistics are in one auditable place.
"""
import math
import re

import numpy as np

# ----------------------------------------------------------------------------
# Ingredient normalization (PROTOCOL section 4)
# ----------------------------------------------------------------------------
CATIONS = {
    "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM", "FERROUS", "FERRIC", "ZINC",
    "LITHIUM", "ALUMINUM", "ALUMINIUM", "AMMONIUM", "SILVER", "BARIUM", "COPPER",
    "MANGANESE", "CHROMIUM", "SELENIUM", "IRON", "GOLD", "BISMUTH", "STRONTIUM",
    "RADIUM", "CUPRIC", "COBALT", "TECHNETIUM", "GADOLINIUM",
}
SALT_WORDS = {
    "HYDROCHLORIDE", "DIHYDROCHLORIDE", "HCL", "HYDROBROMIDE", "SODIUM", "DISODIUM",
    "MONOSODIUM", "POTASSIUM", "DIPOTASSIUM", "CALCIUM", "MAGNESIUM", "SULFATE",
    "SULPHATE", "BISULFATE", "MESYLATE", "MESILATE", "DIMESYLATE", "MALEATE",
    "TARTRATE", "BITARTRATE", "FUMARATE", "HEMIFUMARATE", "ACETATE", "PHOSPHATE",
    "DIPHOSPHATE", "BESYLATE", "BESILATE", "TOSYLATE", "DITOSYLATE", "SUCCINATE",
    "CITRATE", "BROMIDE", "CHLORIDE", "IODIDE", "NITRATE", "MONOHYDRATE",
    "DIHYDRATE", "TRIHYDRATE", "HEMIHYDRATE", "SESQUIHYDRATE", "ANHYDROUS",
    "HYDRATE", "TROMETHAMINE", "MEGLUMINE", "PAMOATE", "EMBONATE", "DECANOATE",
    "PALMITATE", "ENANTHATE", "CYPIONATE", "PROPIONATE", "DIPROPIONATE",
    "FUROATE", "VALERATE", "ACETONIDE", "BUTYRATE", "BENZOATE", "STEARATE",
    "LACTATE", "GLUCONATE", "MALATE", "OXALATE", "MEDOXOMIL", "AXETIL",
    "PIVOXIL", "PROXETIL", "ETEXILATE", "MOFETIL", "CILEXETIL", "DISOPROXIL",
    "ALAFENAMIDE", "MARBOXIL", "ERBUMINE", "ARGININE", "OLAMINE", "DIOLAMINE",
    "EPOLAMINE", "SACCHARATE", "ESTOLATE", "ETHYLSUCCINATE", "XINAFOATE",
    "HYCLATE", "CARBONATE", "HYDROXIDE", "ACEPONATE", "PIVALATE", "TEBUTATE",
    "HEXACETONIDE", "DIACETATE", "TRIACETATE", "UNSPECIFIED", "RECOMBINANT",
    "TRIFENATATE", "S-MALATE", "CAMSYLATE", "LYSINE", "DL-LYSINE", "ASPARTATE",
    "PROPANEDIOL", "ETHANOLATE", "LAUROXIL", "DIPIVOXIL", "UNDECANOATE",
    "ETABONATE", "TEREPHTHALATE", "NAPSYLATE", "PIDOLATE", "TOSILATE",
    "MEPESUCCINATE", "METAZOATE",
}
# Alkyl and similar prefixes that are not drug names on their own.
ALKYL_PREFIXES = {
    "DIMETHYL", "MONOMETHYL", "METHYL", "ETHYL", "DIETHYL", "TRIETHYL", "TRIMETHYL",
    "BENZYL", "ISOPROPYL", "PROPYL", "BUTYL", "AMYL", "ISOAMYL", "GLYCERYL", "CETYL",
    "STEARYL", "PHENYL", "VINYL", "OCTYL", "LAURYL", "MYRISTYL", "ALLYL", "HEXYL",
    "BENZOYL", "ETHINYL", "POLYVINYL", "HYDROXYPROPYL", "HYDROXYETHYL", "PIPERONYL",
    "PHOSPHATIDYL",
}
_BIOSIM_SUFFIX = re.compile(r"-[A-Z]{4}$")
_WS = re.compile(r"\s+")


def norm_ingredient(part: str) -> str:
    """Normalize one active-ingredient string (no combination separator)."""
    s = _WS.sub(" ", part.upper().replace(",", " ").replace(".", " ")).strip()
    if not s:
        return ""
    s = _BIOSIM_SUFFIX.sub("", s)
    toks = s.split(" ")
    if toks[0] in CATIONS:
        return " ".join(toks)
    kept = list(toks)
    while len(kept) > 1 and kept[-1] in SALT_WORDS:
        kept.pop()
    # A lone alkyl fragment (DIMETHYL, ETHYL, BENZYL ...) is not a drug name:
    # the anion is part of the name (dimethyl fumarate), so keep the original.
    if len(kept) == 1 and kept[0] in ALKYL_PREFIXES and len(toks) > 1:
        return " ".join(toks)
    return " ".join(kept)


def split_prod_ai(prod_ai: str):
    """Return the sorted tuple of normalized ingredients of one prod_ai value."""
    if prod_ai is None:
        return ()
    parts = [norm_ingredient(p) for p in str(prod_ai).split("\\")]
    return tuple(sorted({p for p in parts if p}))


# ----------------------------------------------------------------------------
# Preferred-term rules (PROTOCOL section 5). Terms are compared upper-cased.
# ----------------------------------------------------------------------------
HC_TERMS = {"HEPATOCELLULAR INJURY", "HEPATIC CYTOLYSIS", "JAUNDICE HEPATOCELLULAR"}
CH_TERMS = {
    "CHOLESTASIS", "HEPATITIS CHOLESTATIC", "CHOLESTATIC LIVER INJURY",
    "JAUNDICE CHOLESTATIC", "DRUG-INDUCED CHOLESTASIS",
    "VANISHING BILE DUCT SYNDROME",  # Amendment 3
    "IMMUNE-MEDIATED CHOLESTASIS",  # Amendment 8
}
MX_TERMS = {"MIXED LIVER INJURY"}
UN_TERMS = {
    "DRUG-INDUCED LIVER INJURY", "HEPATOTOXICITY", "LIVER INJURY", "HEPATITIS",
    "HEPATITIS ACUTE", "HEPATITIS TOXIC", "HEPATITIS FULMINANT", "HEPATIC FAILURE",
    "ACUTE HEPATIC FAILURE", "SUBACUTE HEPATIC FAILURE", "HEPATIC NECROSIS",
    "JAUNDICE", "COMA HEPATIC", "HEPATIC ENCEPHALOPATHY",
    "IMMUNE-MEDIATED HEPATITIS", "AUTOIMMUNE HEPATITIS", "LIVER TRANSPLANT",
    # Amendment 8: terms of later MedDRA versions for concepts already on the list
    "IMMUNE-MEDIATED HEPATIC DISORDER", "SUSPECTED DRUG-INDUCED LIVER INJURY",
}
SEVERE_TERMS = {
    "HEPATIC FAILURE", "ACUTE HEPATIC FAILURE", "SUBACUTE HEPATIC FAILURE",
    "HEPATITIS FULMINANT", "HEPATIC ENCEPHALOPATHY", "COMA HEPATIC",
    "HEPATIC NECROSIS", "LIVER TRANSPLANT",
}
BROAD_EXTRA_TERMS = {"LIVER DISORDER", "HEPATIC FUNCTION ABNORMAL"}
# Laboratory terms for the broad definition, matched by rule.
_LAB_RULE = re.compile(
    r"^(TRANSAMINASES INCREASED|HYPERTRANSAMINASAEMIA|ALANINE AMINOTRANSFERASE INCREASED|"
    r"ASPARTATE AMINOTRANSFERASE INCREASED|HEPATIC ENZYME INCREASED|"
    r"LIVER FUNCTION TEST INCREASED|LIVER FUNCTION TEST ABNORMAL|"
    r"BLOOD BILIRUBIN INCREASED|HYPERBILIRUBINAEMIA|BILIRUBIN CONJUGATED INCREASED|"
    r"BLOOD ALKALINE PHOSPHATASE INCREASED|GAMMA-GLUTAMYLTRANSFERASE INCREASED|"
    r"ALANINE AMINOTRANSFERASE ABNORMAL|ASPARTATE AMINOTRANSFERASE ABNORMAL|"
    r"HEPATIC ENZYME ABNORMAL|TRANSAMINASES ABNORMAL|"
    # Amendment 4: abnormal-value variants of the same tests (broad definition only)
    r"BLOOD BILIRUBIN ABNORMAL|BLOOD ALKALINE PHOSPHATASE ABNORMAL|"
    r"GAMMA-GLUTAMYLTRANSFERASE ABNORMAL)$"
)
# Amendment 1 (2026-11-23, before any tabulation by sex): explicit list. The first
# draft rule also caught adverse reactions (suicidal ideation) and non-drug
# poisonings (food, metal, carbon monoxide), which are not overdose context.
OVERDOSE_TERMS = {
    "TOXICITY TO VARIOUS AGENTS", "COMPLETED SUICIDE", "SUICIDE ATTEMPT",
    "INTENTIONAL SELF-INJURY", "POISONING", "POISONING DELIBERATE",
    "ACCIDENTAL POISONING", "INTENTIONAL PRODUCT MISUSE", "DRUG ABUSE",
    "SUBSTANCE ABUSE", "DRUG ABUSER", "SUBSTANCE ABUSER", "ALCOHOL POISONING",
    # Amendment 8
    "SUSPECTED SUICIDE", "SUSPECTED SUICIDE ATTEMPT", "INTENTIONAL PRODUCT MISUSE TO CHILD",
}
_OVERDOSE_RULE = re.compile(r"^(?!RADIATION ).*OVERDOSE")
# Amendment 2: "non-alcoholic" terms are not an alcohol proxy.
_ALCOHOL_RULE = re.compile(r"ALCOHOL")
_NONALC_RULE = re.compile(r"NON-?ALCOHOLIC")
# Broad capture used only to decide which REAC rows are kept on disk.
CAPTURE_RULE = re.compile(
    r"HEPAT|LIVER|CHOLESTA|JAUNDICE|BILIRUBIN|TRANSAMINAS|AMINOTRANSFERASE|"
    r"GLUTAMYL|ALKALINE PHOSPHATASE|BILIARY|BILE|ICTER|OVERDOSE|POISONING|SUICID|"
    r"SELF-INJURY|MISUSE|ABUSE|TOXICITY TO VARIOUS AGENTS|ALCOHOL"
)
# Indication terms that mark prior liver disease (PROTOCOL section 5).
_LIVERDX_RULE = re.compile(
    r"HEPATITIS C|HEPATITIS B|CHRONIC HEPATITIS|CIRRHOSIS|HEPATIC STEATOSIS|"
    r"STEATOHEPATITIS|FATTY LIVER|CHOLANGITIS|LIVER TRANSPLANT|HEPATOCELLULAR CARCINOMA|"
    r"HEPATIC CANCER|HEPATIC NEOPLASM|LIVER DISORDER|HEPATIC FAILURE|HEPATIC FIBROSIS"
)
INDI_CAPTURE_RULE = re.compile(r"HEPAT|LIVER|CIRRHO|CHOLANGI|ALCOHOL|STEATO")


# Post hoc change 18: indication placeholders and competing-cause reaction terms.
_UNKNOWN_INDI = {"", "PRODUCT USED FOR UNKNOWN INDICATION", "DRUG USE FOR UNKNOWN INDICATION",
                 "ILL-DEFINED DISORDER", "UNEVALUABLE EVENT"}


def is_unknown_indication(t: str) -> bool:
    return t.upper().strip() in _UNKNOWN_INDI


_COMPETE_RULES = [
    ("sepsis", re.compile(r"SEPSIS|SEPTIC|MULTIPLE ORGAN DYSFUNCTION")),
    ("biliary", re.compile(
        r"CHOLELITHIASIS|CHOLEDOCHOLITHIASIS|BILE DUCT STONE|BILE DUCT OBSTRUCTION|"
        r"BILIARY OBSTRUCTION|BILE DUCT STENOSIS|CHOLANGITIS|CHOLECYSTITIS|BILIARY COLIC|"
        r"JAUNDICE EXTRAHEPATIC OBSTRUCTIVE")),
    ("neoplasm", re.compile(
        r"METASTASES TO LIVER|HEPATIC CANCER|HEPATOCELLULAR CARCINOMA|HEPATIC NEOPLASM|"
        r"CHOLANGIOCARCINOMA|BILE DUCT CANCER|GALLBLADDER CANCER|PANCREATIC CARCINOMA|"
        r"PANCREATIC NEOPLASM|HEPATOBILIARY CANCER|HEPATOBILIARY NEOPLASM")),
    ("viral", re.compile(
        r"^HEPATITIS [ABCDE]( |$)|HEPATITIS VIRAL|VIRAL HEPATITIS|CYTOMEGALOVIRUS HEPATITIS|"
        r"HERPES SIMPLEX HEPATITIS|HEPATITIS INFECTIOUS")),
    ("ischaemic", re.compile(r"ISCHAEMIC HEPATITIS|HEPATIC ISCHAEMIA|CONGESTIVE HEPATOPATHY")),
]


def competing_cause_group(pt: str) -> str:
    """Group of a reaction term that names another cause of liver test abnormalities, or ''."""
    t = pt.upper().strip()
    for name, rule in _COMPETE_RULES:
        if rule.search(t):
            return name
    return ""


def is_alcohol_term(t: str) -> bool:
    t = t.upper()
    return bool(_ALCOHOL_RULE.search(t)) and not _NONALC_RULE.search(t)


def classify_pt(pt: str) -> dict:
    """Classify one upper-cased preferred term. Returns boolean flags."""
    t = pt.upper().strip()
    hc = t in HC_TERMS
    ch = t in CH_TERMS
    mx = t in MX_TERMS
    un = t in UN_TERMS
    lab = bool(_LAB_RULE.match(t))
    return {
        "hc": hc, "ch": ch, "mx": mx, "un": un,
        "narrow": hc or ch or mx or un,
        "broad": hc or ch or mx or un or lab or (t in BROAD_EXTRA_TERMS),
        "severe": t in SEVERE_TERMS,
        "overdose": bool(_OVERDOSE_RULE.search(t)) or t in OVERDOSE_TERMS,
        "alcohol": is_alcohol_term(t),
    }


def is_liverdx_indication(indi_pt: str) -> bool:
    return bool(_LIVERDX_RULE.search(indi_pt.upper()))


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def woolf(a, b, c, d):
    """log OR and its variance for a 2x2 table [[a,b],[c,d]].

    a = exposed cases, b = exposed non-cases, c = unexposed cases,
    d = unexposed non-cases. Haldane-Anscombe 0.5 correction when a cell is 0.
    """
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = (np.where(zero, x + 0.5, x) for x in (a, b, c, d))
    y = np.log(a * d / (b * c))
    v = 1 / a + 1 / b + 1 / c + 1 / d
    return y, v


def mantel_haenszel(a, b, c, d):
    """Mantel-Haenszel OR over strata with Robins-Breslow-Greenland variance.

    Inputs are arrays over strata. Returns (log OR, variance of log OR, number
    of informative strata). Strata with a zero margin contribute nothing.
    """
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    n = a + b + c + d
    keep = (n > 0) & ((a + c) > 0) & ((b + d) > 0) & ((a + b) > 0) & ((c + d) > 0)
    a, b, c, d, n = a[keep], b[keep], c[keep], d[keep], n[keep]
    R = a * d / n
    S = b * c / n
    P = (a + d) / n
    Q = (b + c) / n
    sR, sS = R.sum(), S.sum()
    if sR <= 0 or sS <= 0:
        return float("nan"), float("nan"), int(keep.sum())
    log_or = math.log(sR / sS)
    var = (
        (P * R).sum() / (2 * sR ** 2)
        + (P * S + Q * R).sum() / (2 * sR * sS)
        + (Q * S).sum() / (2 * sS ** 2)
    )
    return log_or, float(var), int(keep.sum())


def mh_risk_difference(a, b, c, d):
    """Mantel-Haenszel weighted risk difference (exposed minus unexposed) over strata.

    Weights n1*n0/N per stratum, treated as fixed. Binomial variance within stratum.
    Returns (risk difference, variance).
    """
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    n1, n0 = a + b, c + d
    keep = (n1 > 0) & (n0 > 0)
    a, c, n1, n0 = a[keep], c[keep], n1[keep], n0[keep]
    w = n1 * n0 / (n1 + n0)
    p1, p0 = a / n1, c / n0
    rd = (w * (p1 - p0)).sum() / w.sum()
    var = (w ** 2 * (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)).sum() / w.sum() ** 2
    return float(rd), float(var)


def reml_tau2(y, v, tol=1e-10, max_iter=500):
    """REML estimate of between-unit variance tau^2, with pooled mean and SE."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    k = len(y)
    w = 1 / v
    mu_fe = (w * y).sum() / w.sum()
    Q = (w * (y - mu_fe) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c)  # DerSimonian-Laird start
    for _ in range(max_iter):
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        num = (w ** 2 * ((y - mu) ** 2 - v)).sum() + 1 / w.sum() * (w ** 2).sum()
        # Fisher scoring step for REML
        new = max(0.0, ((w ** 2 * ((y - mu) ** 2 - v)).sum() / (w ** 2).sum()) + 1 / w.sum())
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    se_mu = math.sqrt(1 / w.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return {"tau2": float(tau2), "mu": float(mu), "se_mu": se_mu, "Q": float(Q),
            "df": k - 1, "I2": float(I2), "mu_fixed": float(mu_fe)}


def eb_shrink(y, v, mu, tau2):
    """Empirical-Bayes posterior mean and variance for each unit."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    B = tau2 / (tau2 + v)
    return mu + B * (y - mu), B * v


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def norm_p(z):
    """Two-sided normal p-value."""
    from scipy.stats import norm
    return 2 * norm.sf(np.abs(z))


def fmt_or(log_or, var):
    lo = math.exp(log_or - 1.959964 * math.sqrt(var))
    hi = math.exp(log_or + 1.959964 * math.sqrt(var))
    return f"{math.exp(log_or):.2f} ({lo:.2f} to {hi:.2f})"
