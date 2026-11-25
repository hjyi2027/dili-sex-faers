"""Write a self-contained copy of the manuscript: number macros and tables inlined.

Usage: python3 12_flatten.py <paper_dir> <main_tex_name> <out_file>
"""
import os
import re
import sys


def main(pdir, name, out):
    s = open(os.path.join(pdir, name)).read()
    nd = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(pdir, "numbers.tex")).read()))
    s = s.replace("\\input{numbers.tex}\n", "")
    s = re.sub(r"\\input\{(tables/[^}]+)\}", lambda m: open(os.path.join(pdir, m.group(1))).read().rstrip("\n"), s)
    used = set()

    def sub(m):
        k = m.group(1)
        if k in nd:
            used.add(k)
            return nd[k]
        return m.group(0)
    s = re.sub(r"\\([A-Za-z]+)(?:\{\})?(?![A-Za-z])", lambda m: sub(m) if m.group(1) in nd else m.group(0), s)
    left = [k for k in nd if re.search(r"\\" + k + r"(?![A-Za-z])", s)]
    open(out, "w").write(s)
    print("inlined", len(used), "macros | unresolved", len(left), "| bytes", len(s))


if __name__ == "__main__":
    main(*sys.argv[1:4])
