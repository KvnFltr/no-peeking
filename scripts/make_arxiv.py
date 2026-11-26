"""Bundle the paper for arXiv: a flat folder (and .tar.gz) with main.tex, the style file, the compiled .bbl
(arXiv does not run BibTeX), the generated macros/tables and the figure.

    python scripts/make_arxiv.py --results results/v1 --bbl paper/build/main.bbl
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/v1")
    p.add_argument("--bbl", default="paper/build/main.bbl")
    p.add_argument("--out", default="outputs/arxiv")
    args = p.parse_args()
    results, out = ROOT / args.results, ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "results" / "tables").mkdir(parents=True)
    (out / "results" / "figures").mkdir(parents=True)
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    tex = tex.replace(
        r"\providecommand{\resultsdir}{../results/v1}", r"\providecommand{\resultsdir}{results}"
    )
    (out / "main.tex").write_text(tex, encoding="utf-8")
    shutil.copy(ROOT / "paper" / "neurips_2025.sty", out)
    bbl = ROOT / args.bbl
    if not bbl.exists():
        raise SystemExit(f"{bbl} missing: compile the paper with bibtex first")
    shutil.copy(bbl, out / "main.bbl")
    shutil.copy(results / "macros.tex", out / "results" / "macros.tex")
    for f in (results / "tables").glob("*.tex"):
        shutil.copy(f, out / "results" / "tables" / f.name)
    shutil.copy(results / "figures" / "hero.png", out / "results" / "figures" / "hero.png")
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(out).as_posix())
    print(f"arXiv bundle: {archive}")


if __name__ == "__main__":
    main()
