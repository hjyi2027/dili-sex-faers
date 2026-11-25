"""Download FAERS quarterly ASCII zips and record a manifest.

Usage: python3 00_download.py <data_dir> <first_quarter> <last_quarter>
Example: python3 00_download.py ../data 2014Q3 2026Q4
A quarter that the FDA site does not serve as a valid zip is recorded as absent.
"""
import csv
import hashlib
import os
import subprocess
import sys
import zipfile

BASE = "https://fis.fda.gov/content/Exports/"


def quarters(first, last):
    y, q = int(first[:4]), int(first[5])
    ly, lq = int(last[:4]), int(last[5])
    while (y, q) <= (ly, lq):
        yield y, q
        q += 1
        if q == 5:
            y, q = y + 1, 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_zip(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and len(z.namelist()) > 0
    except Exception:
        return False


def main(data_dir, first, last):
    os.makedirs(data_dir, exist_ok=True)
    rows = []
    for y, q in quarters(first, last):
        name = f"faers_ascii_{y}Q{q}.zip"
        dest = os.path.join(data_dir, name)
        status = "present"
        url_used = ""
        if not (os.path.exists(dest) and valid_zip(dest)):
            status = "absent"
            for variant in (f"faers_ascii_{y}Q{q}.zip", f"faers_ascii_{y}q{q}.zip"):
                url = BASE + variant
                subprocess.run(["curl", "-s", "-L", "-m", "900", "--retry", "3",
                                "-o", dest, url])
                if os.path.exists(dest) and valid_zip(dest):
                    status, url_used = "downloaded", url
                    break
                if os.path.exists(dest):
                    os.remove(dest)
        if status != "absent":
            rows.append({"file": name, "url": url_used, "bytes": os.path.getsize(dest),
                         "sha256": sha256(dest), "status": status})
        else:
            rows.append({"file": name, "url": "", "bytes": 0, "sha256": "", "status": "absent"})
        print(rows[-1]["file"], rows[-1]["status"], rows[-1]["bytes"], flush=True)
    man = os.path.join(data_dir, "manifest.csv")
    old = {}
    if os.path.exists(man):
        with open(man) as f:
            old = {r["file"]: r for r in csv.DictReader(f)}
    for r in rows:
        if r["status"] == "present" and r["file"] in old and old[r["file"]]["url"]:
            r["url"] = old[r["file"]]["url"]
    old.update({r["file"]: r for r in rows})
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "url", "bytes", "sha256", "status"])
        w.writeheader()
        for k in sorted(old):
            w.writerow(old[k])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
