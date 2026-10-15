"""Mark the best (\\best = bold) and second-best (\\second = underline) numbers in the LaTeX tables.

Per column by default; per row in tables whose methods are columns. Cells "a / b / c" are compared position-wise.
Floor rows (\\floor), signed deltas, ranges and empty cells are not compared; a second best is only marked when at
least three distinct values compete, and nothing is marked when fewer than two values differ. Earlier numeric \\textbf marks are removed first, so the script is idempotent.
Usage: python mark_best.py  (run from docs/paper/latex)"""
import re

NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\\%)?")
MASK = re.compile(r"\\(?:phantom|pr|textsuperscript|cite|TODO)\{[^{}]*\}|\\multicolumn\{\d\}\{[^{}]*\}")
BOLD = re.compile(r"\\(?:textbf|best|second|underline)\{((?:[\d.\s/%]|\\phantom\{0\}|\\%|\\,)+)\}")

# label: (mode, columns, lower-is-better columns, row filter)
SPECS = {
    "retention": ("col", [6, 7], {7}, None),
    "causal": ("col", [2, 3, 4, 5], {3, 5}, None),
    "main": ("col", [1, 2, 3, 4], set(), None),
    "qa": ("col", [1, 2, 3], set(), None),
    "types": ("col", [1, 2, 3], set(), None),
    "ladder": ("col", [1, 2, 3], set(), None),
    "design": ("col", [2], set(), None),
    "shared": ("col", [1, 2], set(), None),
    "pcd8": ("row", [1, 2, 3], set(), None),
    "cf": ("col", [1, 2, 3], {3}, None),
    "scope": [("row", [1, 2], set(), lambda r: "multicolumn" not in r),
              ("col", [1], set(), lambda r: "multicolumn" in r)],
    "replay": ("col", [1, 2], {2}, None),
    "budget": ("col", list(range(1, 10)), set(), None),
    "frames": ("row", [2, 3, 4], set(), None),
    "oracle": ("col", [1, 2, 3, 4, 5], {5}, None),
    "closed": ("col", [1], set(), None),
    "prompt": ("col", [1, 2, 3, 4], set(), None),
    "standard": ("colblock", [2, 3], set(), None),
    "tl8b": ("col", [1, 2], set(), None),
    "nextgqa": ("colblock", [2, 3, 4, 5, 6], set(), None),
    "fullsweep": ("col", [1, 2], set(), None),
}


def numbers(cell):
    """Spans (start, end, value) of the compared number in each '/'-separated part; None where not comparable."""
    masked = MASK.sub(lambda m: " " * len(m[0]), cell)
    out, pos = [], 0
    for part in masked.split("/"):
        m = NUM.search(part)
        ok = m and "--" not in part[m.end():m.end() + 3] and not re.search(r"[+-]\s*$", part[:m.start()].replace("$", ""))
        out.append((pos + m.start(), pos + m.end(), float(m[1])) if ok else None)
        pos += len(part) + 1
    return out


def ranks(cands, lower):
    vals = sorted({v for _, v in cands}, reverse=not lower)
    best = vals[0] if len(cands) >= 2 and len(vals) >= 2 else None
    second = vals[1] if len(vals) >= 3 else None
    return best, second


def mark_table(block, mode, cols, lower, keep):
    lines = block.split("\n")
    body = [i for i, l in enumerate(lines) if "&" in l and l.rstrip().endswith("\\\\") and "\\floor" not in l
            and i > next(j for j, x in enumerate(lines) if x.strip().startswith("\\midrule")) and (keep is None or keep(l))]
    cells = {i: lines[i].rstrip()[:-2].split("&") for i in body}
    marks = {}  # (row, col, start, end) -> macro
    groups = []
    if mode == "colblock":
        mids = [j for j, x in enumerate(lines) if x.strip().startswith("\\midrule")]
        blocks = {}
        for i in body:
            blocks.setdefault(max(j for j in mids if j < i), []).append(i)
        for rows in blocks.values():
            for c in cols:
                n = max(len(numbers(cells[i][c])) for i in rows)
                for k in range(n):
                    groups.append(([(i, c, k) for i in rows], c in lower))
    elif mode == "col":
        for c in cols:
            n = max(len(numbers(cells[i][c])) for i in body if c < len(cells[i]))
            for k in range(n):
                groups.append(([(i, c, k) for i in body if c < len(cells[i])], c in lower))
    else:
        for i in body:
            n = max(len(numbers(cells[i][c])) for c in cols)
            for k in range(n):
                groups.append(([(i, c, k) for c in cols], False))
    for members, low in groups:
        cands = []
        for i, c, k in members:
            spans = numbers(cells[i][c])
            if k < len(spans) and spans[k]:
                cands.append(((i, c, spans[k][0], spans[k][1]), spans[k][2]))
        best, second = ranks(cands, low)
        for key, v in cands:
            if v == best:
                marks[key] = "best"
            elif v == second:
                marks[key] = "second"
    for i in body:
        row = cells[i]
        for c in range(len(row)):
            for (ri, ci, s, e), macro in sorted(((k, m) for k, m in marks.items() if k[0] == i and k[1] == c),
                                                key=lambda x: -x[0][2]):
                row[c] = row[c][:s] + f"\\{macro}{{{row[c][s:e]}}}" + row[c][e:]
        lines[i] = "&".join(row).rstrip() + " \\\\"
    return "\n".join(lines)


def process(path):
    src = open(path).read()
    parts = re.split(r"(\\begin\{table\*?\}.*?\\end\{table\*?\})", src, flags=re.S)
    for j, p in enumerate(parts):
        m = re.search(r"\\label\{tab:([\w-]+)\}", p)
        if not p.startswith("\\begin{table") or not m:
            continue
        p = BOLD.sub(lambda b: b[1], p)
        p = p.replace("$\\mathbf{-14.2^*}$", "$-14.2^*$")
        p = re.sub(r"\\textbf\{(\d+(?:\.\d+)?\\%)\}", r"\1", p)
        spec = SPECS.get(m[1])
        for s in (spec if isinstance(spec, list) else [spec] if spec else []):
            p = mark_table(p, *s)
        parts[j] = p
    open(path, "w").write("".join(parts))


for f in ["sec/3_diagnosis.tex", "sec/6_results.tex", "sec/6b_transfer.tex", "sec/X_suppl.tex"]:
    process(f)
print("ok")
