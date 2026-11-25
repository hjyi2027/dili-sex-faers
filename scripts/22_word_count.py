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
