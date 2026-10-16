"""Compare the answers of two runs question by question: exact matches and, otherwise, the first differing character."""
import sys

import pandas as pd

a, b = (pd.read_csv(f"{d}/1_0.csv").set_index("question_id") for d in sys.argv[1:3])
qs = [q for q in a.index if q in b.index]
same = [q for q in qs if str(a.loc[q, "pred_raw"]) == str(b.loc[q, "pred_raw"])]
print(f"{sys.argv[1]} vs {sys.argv[2]}: {len(same)}/{len(qs)} answers identical")
for q in qs:
    x, y = str(a.loc[q, "pred_raw"]), str(b.loc[q, "pred_raw"])
    if x != y:
        i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"  differs at char {i}/{len(x)}:\n    {x[max(0, i - 40):i + 30]!r}\n    {y[max(0, i - 40):i + 30]!r}")
