# CVPR draft (LaTeX)

Canonical version of the paper; `../main.md` is the earlier markdown draft.

- `main.tex` — entry point (CVPR 2026 author kit, review mode); sections in `sec/`, overview figure in `fig/overview.tex`
  (TikZ), bibliography in `main.bib`.
- `main.pdf` — compiled copy.
- Build: `tectonic -X compile main.tex` (XeTeX; the preamble switches to TeX Gyre Termes, the Times clone) or
  `latexmk -pdf main.tex` (pdfLaTeX, Times as in the CVPR kit).
- Tables: best value bold, second best underlined; regenerate with `python ../analysis/mark_best.py` after editing
  numbers.
- `\TODO{...}` marks missing experiments (red in the PDF); `[authors to verify]` / `note = {[verify]}` in `main.bib`
  marks references that still need checking. To hide TODOs for a submission, uncomment the two `\renewcommand` lines
  in `preamble.tex`.
- Numbers: `../analysis/floors.py` (floors, paired intervals) and `hermes_pool/reports/report_all.py --mss`.
