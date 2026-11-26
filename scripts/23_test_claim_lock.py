"""Checks the claim lock in 22_word_count.py: the current text must pass, and rewordings of the
summary sentences that strengthen the phenotype claim or drop a qualifier must fail.

Usage: python3 23_test_claim_lock.py <paper_dir>
"""
import os, shutil, subprocess, sys, tempfile

paper = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(paper, 'main.tex')).read()

ABS = "The lower cholestatic share in women's reports is sensitive to one reporting term and is not established."
S41 = r", with the estimate outside France unchanged at \XPhNoCytolysisOnlyOtherCountriesDa"
S5 = "was sensitive to one reporting term used mainly in France"
KP = "The difference is sensitive to one term used mainly in France, and outside France its confidence interval reaches 1."
S43 = "standardization by drug and age band together removed"
CASES = [
    ('abstract, "largely drove"', ABS, "One reporting term largely drove the lower cholestatic share in women's reports."),
    ('abstract, "mostly reflects"', ABS, "The lower cholestatic share in women's reports mostly reflects one reporting term."),
    ('abstract, artefact', ABS, "The lower cholestatic share in women's reports is a reporting artefact."),
    ('abstract, hedge dropped', ABS, "Women's reports named a cholestatic injury less often than men's within drug."),
    ('abstract, hedge kept with an attribution', ABS, "The lower cholestatic share in women's reports is sensitive to one reporting term, which largely drove it."),
    ('abstract, wording of version 3.22', ABS, "One term used mainly in France carried more than half of the within-drug sex difference in reported phenotype."),
    ('Sect. 4.1, estimate outside France dropped', S41, ""),
    ('Sect. 5, "came mostly from"', S5, "came mostly from one reporting term used mainly in France"),
    ('Key Points, hedge dropped', KP, "The difference comes from one term used mainly in France."),
    ('Sect. 4.3, age band dropped', S43, "standardization by drug removed"),
]

def lock_exit(text):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'main.tex'), 'w').write(text)
        shutil.copy(os.path.join(paper, 'numbers.tex'), d)
        r = subprocess.run([sys.executable, os.path.join(here, '22_word_count.py'), os.path.join(d, 'main.tex'), d],
                           capture_output=True, text=True)
        return r.returncode

bad = []
if lock_exit(src) != 0:
    bad.append('the current text fails the lock')
for name, old, new in CASES:
    if src.count(old) != 1:
        bad.append(name + ': anchor text not found exactly once')
    elif lock_exit(src.replace(old, new)) == 0:
        bad.append(name + ': the rewording passed the lock')
for b in bad:
    print('  LOCK TEST FAIL:', b)
print('lock tests:', 'FAILED' if bad else 'passed, %d rewordings rejected' % len(CASES))
sys.exit(1 if bad else 0)
