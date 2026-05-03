"""Verify connectivity to all LLM providers.

Usage:
    PYTHONPATH=src python3 scripts/verify_apis.py [--skip-missing]

Reads keys from env (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).
Prints a colored pass/fail table and exits non-zero if any configured
provider fails. With --skip-missing, providers without env keys are
ignored rather than reported as failures.

If all 3 keys are present, additionally runs ONE end-to-end debate-judge
round to verify the full chain.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from ccema.judges.debate import DebateJudge
from ccema.judges.llm_client import (
    AnthropicClient,
    GoogleClient,
    LLMClient,
    OpenAIClient,
)
from ccema.judges.rubric import DEFAULT_RUBRIC


# ---------------------------------------------------------------------------
# Color helpers (ANSI; auto-disable if not a TTY)
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_COLOR = _supports_color()


def _c(s: str, code: str) -> str:
    if not _COLOR:
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def green(s: str) -> str:
    return _c(s, "32")


def red(s: str) -> str:
    return _c(s, "31")


def yellow(s: str) -> str:
    return _c(s, "33")


def bold(s: str) -> str:
    return _c(s, "1")


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass
class ProviderSpec:
    name: str
    env_var: str
    factory: callable

    def has_key(self) -> bool:
        return bool(os.environ.get(self.env_var))


PROVIDERS: list[ProviderSpec] = [
    ProviderSpec("anthropic", "ANTHROPIC_API_KEY", lambda: AnthropicClient()),
    ProviderSpec("openai", "OPENAI_API_KEY", lambda: OpenAIClient()),
    ProviderSpec("google", "GOOGLE_API_KEY", lambda: GoogleClient()),
]


# ---------------------------------------------------------------------------
# Health check + reporting
# ---------------------------------------------------------------------------


def print_table(rows: list[dict]) -> None:
    cols = ["provider", "key", "model", "status", "detail"]
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    sep = "+".join("-" * (widths[c] + 2) for c in cols)
    sep = f"+{sep}+"
    print(sep)
    header = " | ".join(bold(c.ljust(widths[c])) for c in cols)
    print(f"| {header} |")
    print(sep)
    for r in rows:
        cells = []
        for c in cols:
            val = str(r.get(c, ""))
            pad = " " * (widths[c] - len(val))
            if c == "status":
                if val == "PASS":
                    val = green(val)
                elif val == "FAIL":
                    val = red(val)
                elif val == "SKIP":
                    val = yellow(val)
            cells.append(val + pad)
        print("| " + " | ".join(cells) + " |")
    print(sep)


def run_health_checks(skip_missing: bool) -> tuple[list[dict], int]:
    """Returns (rows, n_failures)."""
    rows: list[dict] = []
    failures = 0
    for spec in PROVIDERS:
        has_key = spec.has_key()
        if not has_key and skip_missing:
            rows.append(
                {
                    "provider": spec.name,
                    "key": "missing",
                    "model": "-",
                    "status": "SKIP",
                    "detail": f"{spec.env_var} not set; skipped",
                }
            )
            continue
        client = spec.factory()
        result = client.health_check()
        status = "PASS" if result["ok"] else "FAIL"
        if not result["ok"]:
            failures += 1
        detail = result["error"] or "ok"
        if len(detail) > 80:
            detail = detail[:77] + "..."
        rows.append(
            {
                "provider": spec.name,
                "key": "present" if has_key else "missing",
                "model": result.get("model", "-"),
                "status": status,
                "detail": detail,
            }
        )
    return rows, failures


# ---------------------------------------------------------------------------
# End-to-end debate-judge smoke (only if all 3 keys are present)
# ---------------------------------------------------------------------------


def run_e2e_debate() -> bool:
    """Run one debate round on a tiny synthetic example. Returns True on success."""
    print()
    print(bold("End-to-end debate-judge smoke test"))
    print("-" * 50)

    x = (
        "62yo M, hx HTN, DM2. Presents with sudden-onset crushing chest pain "
        "radiating to L arm, diaphoresis. BP 150/90, HR 110."
    )
    a = (
        "Differential: 1) STEMI, 2) Aortic dissection, 3) PE. "
        "Plan: ECG, troponin, aspirin 325mg PO, transfer to ED."
    )
    criterion = DEFAULT_RUBRIC.criteria[0]  # ddx_accuracy

    debate = DebateJudge(
        defender=AnthropicClient(),
        prosecutor=GoogleClient(),
        judge=OpenAIClient(),
        defender_prompt="You are the defender. Argue the criterion '{criterion}' IS satisfied.",
        prosecutor_prompt="You are the prosecutor. Argue the criterion '{criterion}' is NOT satisfied.",
        judge_prompt="You are the judge for criterion '{criterion}'. Output JSON only.",
        max_tokens_advocate=128,
        max_tokens_judge=256,
    )

    try:
        score = debate.score_axis(x, a, criterion)
    except Exception as exc:  # noqa: BLE001
        print(red(f"FAIL: {exc}"))
        return False

    print(green(f"PASS: axis={criterion.axis_id} mean={score.mean:.3f} alpha={score.alpha} beta={score.beta}"))
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip providers whose API key env var is unset (don't count as failures).",
    )
    args = parser.parse_args(argv)

    print(bold("LLM provider health checks"))
    print("-" * 50)
    rows, failures = run_health_checks(skip_missing=args.skip_missing)
    print_table(rows)

    # Summary line
    total = len(rows)
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    print(
        f"\n{n_pass} pass, {n_fail} fail, {n_skip} skip "
        f"(of {total} provider{'s' if total != 1 else ''})"
    )

    # Run e2e only if every provider has a key
    all_keys = all(spec.has_key() for spec in PROVIDERS)
    if all_keys and failures == 0:
        ok = run_e2e_debate()
        if not ok:
            failures += 1
    elif not all_keys:
        print(
            yellow(
                "\nSkipping end-to-end debate test (requires all 3 keys: "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)."
            )
        )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
