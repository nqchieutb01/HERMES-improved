"""Highlight numbers in the LaTeX tables: shade our method's rows, colour significant gains green and significant
losses red (only inside table environments). Idempotent: already-wrapped entries are left alone."""
import re
import sys

PAT_CI = re.compile(r"(?<!\\up\{)(?<!\\dn\{)\$([+-])(\d+(?:\.\d+)?)\$ \\pr\{([^{}]*)\}\$\^\*\$")
PAT_STAR = re.compile(r"(?<!\\up\{)(?<!\\dn\{)\$([+-])(\d+(?:\.\d+)?)\^\*\$")


def colour(sign, body):
    return (r"\up{" if sign == "+" else r"\dn{") + body + "}"


def fix_table(block):
    block = PAT_CI.sub(lambda m: colour(m[1], f"${m[1]}{m[2]}$ \\pr{{{m[3]}}}$^*$"), block)
    block = PAT_STAR.sub(lambda m: colour(m[1], f"${m[1]}{m[2]}^*$"), block)
    out = []
    for line in block.split("\n"):
        s = line.lstrip()
        first = s.split("&")[0] if "&" in s else ""
        if "\\method" in first and not s.startswith("\\oursrow") and "\\caption" not in s:
            line = line[: len(line) - len(s)] + "\\oursrow " + s
        out.append(line)
    return "\n".join(out)


for path in sys.argv[1:]:
    src = open(path).read()
    parts = re.split(r"(\\begin\{table\*?\}.*?\\end\{table\*?\})", src, flags=re.S)
    src = "".join(fix_table(p) if p.startswith("\\begin{table") else p for p in parts)
    open(path, "w").write(src)
    print(path, src.count("\\oursrow"), src.count("\\up{"), src.count("\\dn{"))
