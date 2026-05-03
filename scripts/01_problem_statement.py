"""Step 1 driver: prints the locked-in problem statement and notation table.

This is a thin runner — the problem statement lives in
docs/problem_statement_v2.md and the notation table is exported from there.
The script exists so the workflow is fully scriptable from step 1 to 10.

Usage:
    python scripts/01_problem_statement.py
"""

from __future__ import annotations

from pathlib import Path

from ccema.utils.config import config_hash, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "eval_config_v1.yaml")
    print(f"Loaded config v1 (hash={config_hash(cfg)})")

    statement_path = REPO_ROOT / "docs" / "problem_statement_v2.md"
    if not statement_path.exists():
        raise SystemExit(
            f"Missing {statement_path}. Step 1 deliverable not yet authored."
        )

    text = statement_path.read_text(encoding="utf-8")
    n_lines = text.count("\n")
    n_refs = text.lower().count("\n- ")
    print(f"Problem statement: {n_lines} lines, ~{n_refs} bulleted items")
    print(f"\nFirst 60 lines of {statement_path.relative_to(REPO_ROOT)}:")
    print("-" * 70)
    for line in text.splitlines()[:60]:
        print(line)


if __name__ == "__main__":
    main()
