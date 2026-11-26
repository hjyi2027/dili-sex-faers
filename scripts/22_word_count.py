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

# Claim lock (version 3.23). Summaries of the phenotype result must keep the estimate outside France
# or the hedge, and must not attribute most of the difference to the term. A failure stops the build.
fails = []
def block(start, ends):
    if start not in t:
        fails.append('missing section: ' + start); return ''
    i = t.index(start) + len(start)
    stops = [t.index(e, i) for e in ends if e in t[i:]]
    return t[i:min(stops) if stops else len(t)]
summaries = {
    'abstract': block('\\abstract{', ['\\keywords']),
    'Key Points': block('\\section*{Key Points}', ['\\end{itemize}']),
    'Sect. 4.1': block('\\subsection{Principal findings}', ['\\subsection{']),
    'Sect. 5': block('\\section{Conclusions}', ['\\bmhead{', '\\backmatter']),
}
TERM = re.compile(r'cytolysis|reporting term|one term', re.I)
KEEP = re.compile(r'\\XPhNoCytolysisOnlyOtherCountriesDa|outside France|not established|sensitive to', re.I)
DRIFT = re.compile(r'\b(carried|carries|accounted for|accounts for|explain\w*|due to)\s+(more than half|most|the majority|much|nearly all|all)\b', re.I)
for name, b in summaries.items():
    if TERM.search(b) and not KEEP.search(b):
        fails.append(name + ': names the term without the estimate outside France or the hedge')
    for m in DRIFT.finditer(b):
        fails.append(name + ': attribution "' + m.group(0) + '"')
    if '\\XCytoShareLogor' in b and '\\XPhNoCytolysisOnlyOtherCountriesDa' not in b:
        fails.append(name + ': share carried by the term without the estimate outside France')
for f in fails:
    print('  CLAIM LOCK FAIL:', f)
print('claim lock:', 'FAILED' if fails else 'passed')
sys.exit(1 if fails else 0)
