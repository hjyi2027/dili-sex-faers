"""Word counts of the flattened manuscript and a few style checks.

Usage: python3 22_word_count.py <flattened main.tex> <paper_dir>
"""
import re, sys
s=open(sys.argv[1]).read()
body=s[s.index('\\section{Introduction}'):s.index('\\backmatter')]
body=re.sub(r'\\begin\{(table|figure|sidewaystable)\}.*?\\end\{\1\}','',body,flags=re.S)
def strip(t):
    t=re.sub(r'\\cite\{[^}]*\}','',t); t=re.sub(r'\\(label|ref)\{[^}]*\}','',t)
    t=re.sub(r'\{,\}',',',t); t=re.sub(r'\$-\$','-',t)
    t=re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?','',t); t=re.sub(r'[{}$~]',' ',t); return t
ab=s[s.index('\\abstract{'):s.index('\\keywords')]
print('main text words:',len(strip(body).split()),'| abstract words:',len(strip(ab).split()))
print(' '.join('%s %d'%(sec[:sec.index('}')][:12],len(strip(sec).split())) for sec in re.split(r'\\section\{',body)[1:]))
t=open(sys.argv[2]+'/main.tex').read()
defs=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}',open(sys.argv[2]+'/numbers.tex').read()))
abr=t[t.index('\\abstract{'):t.index('\\keywords')]; bod=t[t.index('\\section{Introduction}'):]
bm=set(m for m in re.findall(r'\\([A-Za-z]+)',bod) if m in defs)
for m in [m for m in re.findall(r'\\([A-Za-z]+)',abr) if m in defs]:
    if m in bm: continue
    full=[b for b in bm if defs[m] in defs[b] and m.startswith(b)]
    if not full: print(' ABSTRACT NUMBER NOT IN BODY:',m,'=',defs[m])
print('semicolons:',len(re.findall(r';',re.sub(r'\\[a-zA-Z]+','',t))),'emdash:',t.count('---')+t.count('—'))
for m in re.finditer(r'(?:(?<=\. )|(?<=\n))(And|But|Or|So|Yet|Because|While|Although|Since|However|Moreover|Furthermore|Additionally|Notably|Also|Thus|Hence|Nor|Whereas|Though)\b[^.]{0,60}',t):
    print('  START:',m.group(0)[:70])
for w in ['depends on','depended on one','decided','near-copy','retains the patient','rather than','not only','serves as']:
    for m in re.finditer(w,t): print('  PHRASE:',t[max(0,m.start()-50):m.end()+40].replace('\n',' '))

# Claim lock (version 3.24). In the summary sections (abstract Results and Conclusions, each Key Point,
# Sect. 4.1, Sect. 5) the phenotype result must keep its hedge or the estimate outside France, every
# sentence that names the term must carry the hedge itself and must not attribute the difference to it,
# and two share estimates must keep the words that define them. A failure stops the build.
# scripts/23_test_claim_lock.py checks that rewordings of these sentences fail.
fails = []
def between(start, ends, src=None):
    src = t if src is None else src
    if start not in src:
        fails.append('missing section: ' + start); return ''
    i = src.index(start) + len(start)
    stops = [src.index(e, i) for e in ends if e in src[i:]]
    return src[i:min(stops) if stops else len(src)]
abstract = between('\\abstract{', ['\\keywords'])
units = {
    'abstract Results': between('\\textbf{Results}', ['\\textbf{Conclusions}'], abstract),
    'abstract Conclusions': between('\\textbf{Conclusions}', ['\\keywords'], abstract),
    'Sect. 4.1': between('\\subsection{Principal findings}', ['\\subsection{']),
    'Sect. 5': between('\\section{Conclusions}', ['\\bmhead{', '\\backmatter']),
}
for n, item in enumerate(between('\\section*{Key Points}', ['\\end{itemize}']).split('\\item')[1:], 1):
    units['Key Point %d' % n] = item
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z\\])')
PHEN = re.compile(r'cholest|phenotype', re.I)
TERM = re.compile(r'cytolysis|reporting term|one term|term used mainly', re.I)
HEDGE = re.compile(r'\\XPhNoCytolysisOnlyOtherCountriesDa|outside France|not established|sensitive to', re.I)
PLACE = re.compile(r'used mainly in France|mainly by French reporters', re.I)
ATTRIB = re.compile(r'\b(largely|mostly|primarily|chiefly|predominantly|entirely|wholly|mainly|drove|drive[sn]?|driving'
                    r'|reflect\w*|explain\w*|caused|due to|artefact\w*|artifact\w*|spurious)\b'
                    r'|\b(carried|carries|accounted for|accounts for)\s+(more than half|most|the majority|much|nearly all|all)\b', re.I)
for name, u in units.items():
    if PHEN.search(u) and not HEDGE.search(u):
        fails.append(name + ': states the phenotype result without the hedge or the estimate outside France')
    if re.search(r'artefact|artifact|spurious', u, re.I):
        fails.append(name + ': calls the result an artefact')
    for x in SPLIT.split(u.strip()):
        if TERM.search(x):
            if not HEDGE.search(x):
                fails.append(name + ': names the term without the hedge in the same sentence: ' + x[:80])
            m = ATTRIB.search(PLACE.sub('', x))
            if m:
                fails.append(name + ': attribution "' + m.group(0) + '": ' + x[:80])
        if '\\XCytoShareLogor' in x and '\\XPhNoCytolysisOnlyOtherCountriesDa' not in x:
            fails.append(name + ': share carried by the term without the estimate outside France')
CONTEXT = {'DRStdAdultShare': 'age band', 'DRShareSex': 'log OR'}
for mac, need in CONTEXT.items():
    for x in SPLIT.split(t):
        if re.search(r'\\' + mac + r'(?![A-Za-z])', x) and need not in x:
            fails.append('\\' + mac + ' without "' + need + '" in the same sentence: ' + x.strip()[:80])
for f in fails:
    print('  CLAIM LOCK FAIL:', f)
print('claim lock:', 'FAILED' if fails else 'passed')
sys.exit(1 if fails else 0)
