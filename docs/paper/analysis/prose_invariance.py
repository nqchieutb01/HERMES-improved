"""Check that a prose rewrite of a LaTeX section changed wording only.

For each file, compares the original (a git revision) with the working copy:
  - every table / figure / equation / tabular block is byte-identical;
  - the multiset of numbers in the prose is unchanged;
  - the sets of \\cite keys, \\cref/\\ref targets and \\label names are unchanged;
  - the number of \\TODO markers is unchanged.
Usage: python prose_invariance.py <git-rev> file.tex [file.tex ...]   (run from docs/paper/latex)
"""
import re
import subprocess
import sys
from collections import Counter

BLOCK = re.compile(r"\\begin\{(table\*?|figure\*?|equation\*?|align\*?|tabular|center)\}.*?\\end\{\1\}", re.S)
NUM = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,{}]\d+)*")


def parts(src):
    blocks = [m.group(0) for m in BLOCK.finditer(src)]
    prose = BLOCK.sub(" ", src)
    prose = re.sub(r"(?<!\\)%.*", "", prose)  # drop comments
    nums = Counter(n.replace("{,}", ",").lstrip("+") for n in NUM.findall(re.sub(r"\\(cref|Cref|ref|label|cite[pt]?)\{[^}]*\}", " ", prose)))
    cites = set(k.strip() for g in re.findall(r"\\cite[pt]?\{([^}]*)\}", prose) for k in g.split(","))
    refs = set(k.strip() for g in re.findall(r"\\[cC]?ref\{([^}]*)\}", prose) for k in g.split(","))
    labels = set(re.findall(r"\\label\{([^}]*)\}", prose))
    todos = len(re.findall(r"\\TODO\{", prose))
    return blocks, nums, cites, refs, labels, todos


def main():
    # --subset: prose may drop numbers, cites and refs (concision edits) but may not add any; labels, TODOs and
    # table/figure/equation blocks must still be unchanged.
    subset = "--subset" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--subset"]
    rev, files = args[0], args[1:]
    ok = True
    for f in files:
        old = subprocess.run(["git", "show", f"{rev}:./{f}"], capture_output=True, text=True).stdout
        new = open(f).read()
        (ob, on, oc, orf, ol, ot), (nb, nn, nc, nrf, nl, nt) = parts(old), parts(new)
        errs = []
        if ob != nb:
            errs.append(f"blocks changed ({len(ob)} -> {len(nb)}; first differing index "
                        f"{next((i for i, (a, b) in enumerate(zip(ob, nb)) if a != b), min(len(ob), len(nb)))})")
        if (nn - on) if subset else (on != nn):
            errs.append(f"numbers: removed {dict(on - nn)} added {dict(nn - on)}")
        for name, a, b in (("cites", oc, nc), ("refs", orf, nrf), ("labels", ol, nl)):
            bad = (b - a) if (subset and name != "labels") else (a ^ b)
            if bad:
                errs.append(f"{name}: removed {sorted(a - b)} added {sorted(b - a)}")
        if ot != nt:
            errs.append(f"TODOs {ot} -> {nt}")
        print(f"{'OK  ' if not errs else 'FAIL'} {f}" + "".join("\n     " + e for e in errs))
        ok &= not errs
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
