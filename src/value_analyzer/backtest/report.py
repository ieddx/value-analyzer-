"""Plain-text and CSV report formatters for BacktestResult.

The backtest layer does not use ``rich`` — that belongs in report/.
These functions produce plain strings suitable for printing or writing to disk.
"""

from __future__ import annotations

import csv
import io
from datetime import date

from .models import BacktestResult, QuintileStats


def format_report(result: BacktestResult) -> str:
    """Return a multi-section plain-text backtest report string."""
    lines: list[str] = []
    w = 76  # column width

    def rule(char: str = "═") -> None:
        lines.append(char * w)

    def section(title: str) -> None:
        lines.append("")
        lines.append(title.upper())
        lines.append("─" * w)

    # ── Header ────────────────────────────────────────────────────────────
    rule()
    lines.append("VALUE ANALYZER — BACKTEST REPORT")
    lines.append(
        f"Run: {result.run_date}  |  "
        f"Dates: {result.as_of_dates[0]}–{result.as_of_dates[-1]}  |  "
        f"Universe: {len(result.universe)} tickers"
    )
    lines.append(
        f"Benchmark: {result.benchmark_ticker}  |  "
        f"Transaction cost: {result.transaction_cost_bps:.0f} bps round-trip"
    )
    rule()

    # ── Survivorship bias warning ─────────────────────────────────────────
    lines.append("")
    lines.append(result.survivorship_bias_note)

    # ── Execution summary ─────────────────────────────────────────────────
    section("Execution summary")
    lines.append(f"  Attempted:          {result.n_attempted}")
    lines.append(f"  Scored OK:          {result.n_scored}")
    lines.append(f"  Errors / no data:   {result.n_errors}")
    lines.append(f"  With 1y return:     {result.n_with_1y_return}")
    lines.append(f"  With 3y return:     {result.n_with_3y_return}")
    lines.append(f"  With 5y return:     {result.n_with_5y_return}")

    # ── Quintile returns ──────────────────────────────────────────────────
    section("Forward returns by score quintile (net of transaction costs)")
    lines.append(_quintile_table(result.quintile_stats))

    # ── Q1–Q5 spread and hit rate ─────────────────────────────────────────
    section("Q1 − Q5 spread and hit rate")
    lines.append(f"  {'Horizon':<10}  {'Spread':>8}  {'Hit rate':>10}")
    lines.append("  " + "-" * 32)
    for hz, spread, hr in [
        ("1-Year", result.q1_q5_spread_1y, result.hit_rate_1y),
        ("3-Year", result.q1_q5_spread_3y, result.hit_rate_3y),
        ("5-Year", result.q1_q5_spread_5y, result.hit_rate_5y),
    ]:
        spread_str = f"{spread * 100:+.1f}%" if spread is not None else "n/a"
        hr_str = f"{hr:.0%}" if hr is not None else "n/a"
        lines.append(f"  {hz:<10}  {spread_str:>8}  {hr_str:>10}")
    lines.append("")
    lines.append("  Hit rate = fraction of snapshot dates where Q1 mean > Q5 mean.")
    lines.append("  50% = indistinguishable from chance at the date level.")

    # ── Benchmark comparison ──────────────────────────────────────────────
    section(f"Versus benchmark ({result.benchmark_ticker} buy-and-hold)")
    _benchmark_table(lines, result)

    # ── Statistical significance ──────────────────────────────────────────
    section("Statistical significance (per-date Q1−Q5 spread t-test, H₀: spread=0)")
    for hz, t_stat, p_val in [
        ("1-Year", result.t_stat_1y, result.p_value_1y),
        ("3-Year", result.t_stat_3y, result.p_value_3y),
        ("5-Year", result.t_stat_5y, result.p_value_5y),
    ]:
        t_str = f"t={t_stat:+.2f}" if t_stat is not None else "t=n/a"
        p_str = f"p={p_val:.3f}" if p_val is not None else "p=n/a (install scipy)"
        lines.append(f"  {hz:<10} {t_str}  {p_str}")

    # ── Sample size note ──────────────────────────────────────────────────
    section("Sample size caveat")
    for chunk in _wrap(result.sample_size_note, w - 2):
        lines.append(f"  {chunk}")

    # ── Cost model ───────────────────────────────────────────────────────
    section("Cost model")
    for chunk in _wrap(result.cost_model_note, w - 2):
        lines.append(f"  {chunk}")

    # ── Conclusion ────────────────────────────────────────────────────────
    section("Conclusion")
    for chunk in _wrap(result.conclusion, w - 2):
        lines.append(f"  {chunk}")

    rule()
    return "\n".join(lines)


def to_csv(result: BacktestResult) -> str:
    """Return a CSV string with one row per (ticker, as_of_date) snapshot."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "ticker", "as_of_date", "composite_score", "weight_profile", "quintile",
        "fwd_return_1y", "fwd_return_3y", "fwd_return_5y",
        "net_return_1y", "net_return_3y", "net_return_5y",
        "benchmark_return_1y", "benchmark_return_3y", "benchmark_return_5y",
        "error",
    ])
    writer.writeheader()
    for s in result.snapshots:
        writer.writerow({
            "ticker": s.ticker,
            "as_of_date": s.as_of_date.isoformat(),
            "composite_score": _fmt(s.composite_score),
            "weight_profile": s.weight_profile or "",
            "quintile": s.quintile if s.quintile is not None else "",
            "fwd_return_1y": _pct(s.fwd_return_1y),
            "fwd_return_3y": _pct(s.fwd_return_3y),
            "fwd_return_5y": _pct(s.fwd_return_5y),
            "net_return_1y": _pct(s.net_return_1y),
            "net_return_3y": _pct(s.net_return_3y),
            "net_return_5y": _pct(s.net_return_5y),
            "benchmark_return_1y": _pct(s.benchmark_return_1y),
            "benchmark_return_3y": _pct(s.benchmark_return_3y),
            "benchmark_return_5y": _pct(s.benchmark_return_5y),
            "error": s.error or "",
        })
    return buf.getvalue()


# ── Private helpers ────────────────────────────────────────────────────────

def _quintile_table(stats: list[QuintileStats]) -> str:
    header = f"  {'':20s}  {'1-Year':>8}  {'3-Year':>8}  {'5-Year':>8}  {'N':>5}"
    rows = [header, "  " + "-" * (len(header) - 2)]
    for q in sorted(stats, key=lambda x: x.quintile):
        r1 = _pct(q.mean_net_return_1y) or "n/a"
        r3 = _pct(q.mean_net_return_3y) or "n/a"
        r5 = _pct(q.mean_net_return_5y) or "n/a"
        rows.append(f"  {q.label:<20s}  {r1:>8}  {r3:>8}  {r5:>8}  {q.n_obs:>5}")
    return "\n".join(rows)


def _benchmark_table(lines: list[str], result: BacktestResult) -> None:
    for hz, q1_ret, bm_ret, q1_vs_bm in [
        ("1-Year", _get_q1_ret(result, "1y"), result.benchmark_avg_1y, result.q1_vs_benchmark_1y),
        ("3-Year", _get_q1_ret(result, "3y"), result.benchmark_avg_3y, result.q1_vs_benchmark_3y),
        ("5-Year", _get_q1_ret(result, "5y"), result.benchmark_avg_5y, result.q1_vs_benchmark_5y),
    ]:
        q1_str = _pct(q1_ret) or "n/a"
        bm_str = _pct(bm_ret) or "n/a"
        diff_str = _pct(q1_vs_bm) or "n/a"
        lines.append(f"  {hz:<10}  Q1: {q1_str:>8}  {result.benchmark_ticker}: {bm_str:>8}  diff: {diff_str:>8}")


def _get_q1_ret(result: BacktestResult, hz: str) -> float | None:
    q1 = next((q for q in result.quintile_stats if q.quintile == 1), None)
    if q1 is None:
        return None
    return {"1y": q1.mean_net_return_1y, "3y": q1.mean_net_return_3y, "5y": q1.mean_net_return_5y}[hz]


def _pct(v: float | None) -> str:
    return f"{v * 100:+.1f}%" if v is not None else ""


def _fmt(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else ""


def _wrap(text: str, width: int) -> list[str]:
    """Very simple word-wrap."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
