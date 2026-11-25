"""Check the estimators against simulated data with a known answer."""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from importlib import import_module
an = import_module("05_analyze")
from scipy.stats import chi2
rng = np.random.default_rng(1)

# 1. Mantel-Haenszel recovers a planted common OR under strong confounding by stratum.
S, true_or = 400, 0.70
base = rng.uniform(0.002, 0.2, S)                 # stratum baseline risk
pf = rng.uniform(0.1, 0.9, S)                     # stratum share of women (drug mix)
n = rng.integers(200, 20000, S)
nf = rng.binomial(n, pf); nm = n - nf
odds_m = base / (1 - base); odds_f = odds_m * true_or
a = rng.binomial(nf, odds_f / (1 + odds_f)); c = rng.binomial(nm, base)
y, v, k = lib.mantel_haenszel(a, nf - a, c, nm - c)
yc, vc = lib.woolf(a.sum(), (nf - a).sum(), c.sum(), (nm - c).sum())
print(f"MH: planted {true_or}, estimate {math.exp(y):.3f}, 95% CI {math.exp(y-1.96*v**.5):.3f}-{math.exp(y+1.96*v**.5):.3f}; crude {math.exp(float(yc)):.3f}")

# 2. Coverage of the MH interval over repeated simulations.
cover = 0; R = 3000
for _ in range(R):
    a = rng.binomial(nf, odds_f / (1 + odds_f)); c = rng.binomial(nm, base)
    y, v, k = lib.mantel_haenszel(a, nf - a, c, nm - c)
    cover += (y - 1.96 * v ** .5 <= math.log(true_or) <= y + 1.96 * v ** .5)
print(f"MH 95% CI coverage over {R} runs: {cover / R:.3f}")

# 3. REML tau2 recovers planted heterogeneity.
K, mu, tau = 300, -0.2, 0.35
vi = rng.uniform(0.005, 0.2, K)
yi = rng.normal(mu, np.sqrt(tau ** 2 + vi))
h = lib.reml_tau2(yi, vi)
print(f"REML: planted mu {mu}, tau2 {tau**2:.4f}; estimate mu {h['mu']:.3f}, tau2 {h['tau2']:.4f}")

# 4. Type I error of the fixed-OR likelihood-ratio test at realistic counts.
psi = 0.8; rej05 = 0; rej001 = 0; R = 20000
for _ in range(R):
    r1 = rng.integers(400, 40000); r2 = rng.integers(400, 40000)
    pm = rng.uniform(0.01, 0.1); of = psi * pm / (1 - pm)
    a = rng.binomial(r1, of / (1 + of)); c = rng.binomial(r2, pm)
    if a < 20 or c < 20: continue
    g2 = an.lrt_fixed_or(a, r1 - a, c, r2 - c, psi)
    rej05 += chi2.sf(g2, 1) < 0.05; rej001 += chi2.sf(g2, 1) < 0.001
print(f"LRT type I error: {rej05 / R:.4f} at 0.05, {rej001 / R:.5f} at 0.001")

# 5. BH on a known example.
p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
print("BH q:", np.round(lib.bh_fdr(p), 4))
