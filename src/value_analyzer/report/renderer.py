"""Rich-based terminal report for a single CompositeScore.

Usage
-----
    from value_analyzer.report import render
    render(composite_score)

Rules enforced here
-------------------
- No bare print() — all output goes through rich Console.
- No directive language (buy / sell / hold / avoid / strong buy).
- Framing is always: "if an investor applied framework X, the result would be Y."
- DISCLAIMER_TEXT must appear in every rendered report.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from value_analyzer.news.models import NewsResult as _NewsResult

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

_BACKTEST_SUMMARY_PATH = Path.home() / ".value_analyzer" / "backtest_summary.json"

from value_analyzer.score.models import CompositeScore, SubScore
from value_analyzer.score.config import COMPLETENESS_CAUTION_THRESHOLD
from value_analyzer.peers.models import PeerComparison

DISCLAIMER_TEXT = (
    "This report is analysis only — it is NOT financial advice and NOT a buy or sell "
    "recommendation. All intrinsic-value estimates are the output of stated frameworks "
    "applied to public data. The investor must apply their own judgment."
)

_SCORE_COLORS = {
    "high":   "green",
    "mid":    "yellow",
    "low":    "red",
}


def _score_color(score: float) -> str:
    if score >= 65:
        return _SCORE_COLORS["high"]
    if score >= 40:
        return _SCORE_COLORS["mid"]
    return _SCORE_COLORS["low"]


def _score_bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def _fmt_x(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}×"


# ── Section builders ───────────────────────────────────────────────────────────

def _header_panel(cs: CompositeScore) -> Panel:
    cat = cs.category
    color = _score_color(cs.composite)
    bar = _score_bar(cs.composite)

    lines: list[str] = []
    lines.append(f"[bold]{cs.ticker}[/bold]   as of {cs.as_of_date}   profile: [italic]{cs.weight_profile}[/italic]")
    lines.append(
        f"Composite score: [{color}]{cs.composite:.1f}/100[/{color}]  [{color}]{bar}[/{color}]"
    )
    lines.append("")
    lines.append(
        f"Moat: [bold]{cat.moat_type.value}[/bold]  |  "
        f"Revenue: [bold]{cat.revenue_type.value}[/bold]  |  "
        f"Growth: [bold]{cat.growth_profile.value}[/bold]  |  "
        f"Capital: [bold]{cat.capital_intensity.value}[/bold]"
    )
    weights = cs.weights_used
    lines.append(
        f"Weights — moat {weights['moat']:.0%}  "
        f"health {weights['health']:.0%}  "
        f"valuation {weights['valuation']:.0%}  "
        f"management {weights['management']:.0%}"
    )

    # ── Data completeness indicator ────────────────────────────────────────
    real, total = cs.completeness_real, cs.completeness_total
    if total > 0:
        pct = real / total
        comp_color = (
            "green" if pct >= 0.85
            else "yellow" if pct >= COMPLETENESS_CAUTION_THRESHOLD
            else "red"
        )
        lines.append(
            f"Data completeness: [{comp_color}]{real}/{total} inputs ({pct:.0%})[/{comp_color}]"
        )

        # Per-pillar detail for notably incomplete pillars (< 60%)
        low_pillars = [
            f"{sub.name} {sub.real_inputs}/{sub.total_inputs}"
            for sub in (cs.moat, cs.health, cs.valuation, cs.management)
            if sub.total_inputs > 0 and sub.real_inputs / sub.total_inputs < 0.60
        ]
        if low_pillars:
            lines.append(f"  Incomplete pillars: {', '.join(low_pillars)}")

        # Low-confidence caution when completeness is below threshold OR dispersion fires
        caution_parts: list[str] = []
        if pct < COMPLETENESS_CAUTION_THRESHOLD:
            caution_parts.append(
                f"data completeness {pct:.0%} < {COMPLETENESS_CAUTION_THRESHOLD:.0%} threshold"
            )
        if cs.iv_dispersion_flag is not None:
            caution_parts.append("valuation methods disagree significantly")
        if caution_parts:
            lines.append(
                f"[bold red]⚠ Low-confidence analysis: {'; '.join(caution_parts)} — "
                "treat composite score and IV estimates as indicative only.[/bold red]"
            )

    return Panel("\n".join(lines), title="[bold]Value Analyzer[/bold]", border_style="blue")


def _classification_table(cs: CompositeScore) -> Table:
    cat = cs.category
    table = Table(title="Classification", show_header=True, header_style="bold cyan")
    table.add_column("Dimension", style="bold", min_width=16)
    table.add_column("Result", min_width=18)
    table.add_column("Confidence", justify="center", min_width=10)
    table.add_column("Rationale")

    for dim, trace in cat.traces.items():
        conf_pct = f"{trace.confidence:.0%}"
        conf_color = "green" if trace.confidence >= 0.7 else ("yellow" if trace.confidence >= 0.4 else "red")
        table.add_row(
            dim,
            trace.result,
            f"[{conf_color}]{conf_pct}[/{conf_color}]",
            trace.rationale,
        )
    return table


def _subscore_panel(sub: SubScore, weight: float) -> Panel:
    color = _score_color(sub.score)
    bar = _score_bar(sub.score, width=16)
    header = (
        f"[{color}]{sub.score:.1f}/100[/{color}]  [{color}]{bar}[/{color}]  "
        f"(weight {weight:.0%})"
    )
    lines: list[str] = [header, ""]

    if sub.reasons:
        lines.append("[bold]Components:[/bold]")
        for r in sub.reasons:
            lines.append(f"  {r}")

    if sub.flags:
        lines.append("")
        lines.append("[bold]Notes / data quality:[/bold]")
        for f in sub.flags:
            lines.append(f"  [yellow]{f}[/yellow]")

    return Panel(
        "\n".join(lines),
        title=f"[bold]{sub.name.title()}[/bold]",
        border_style=color,
    )


def _valuation_iv_table(cs: CompositeScore) -> Table | None:
    """Extract IV-estimate lines from the valuation flags and render as a table."""
    iv_lines = [
        f for f in cs.valuation.flags
        if "IV estimate" in f or "Average IV" in f or "Current price" in f
    ]
    if not iv_lines:
        return None

    table = Table(title="Intrinsic-Value Estimates", show_header=True, header_style="bold cyan")
    table.add_column("Item")
    table.add_column("Value", justify="right")

    for line in iv_lines:
        # Strip the leading ⚠ prefix the Scorer adds
        clean = line.lstrip("⚠ ").strip()
        # Split "Label: $123.45 — ..." into label / value columns
        if ": " in clean:
            label, rest = clean.split(": ", 1)
            table.add_row(label, rest)
        else:
            table.add_row(clean, "")

    return table


def _peer_panel(pc: PeerComparison) -> Panel:
    lines: list[str] = []
    lines.append(
        f"[italic]{pc.context_note or 'Same-category peers from value investor 13F filings.'}[/italic]"
    )
    lines.append(f"Peer count: {pc.peer_count}   Profile: {pc.weight_profile}")
    if pc.peer_tickers:
        lines.append(f"Tickers: {', '.join(pc.peer_tickers[:12])}"
                     + (" …" if len(pc.peer_tickers) > 12 else ""))
    lines.append("")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Subject", justify="right")
    table.add_column("Peer median", justify="right")
    table.add_column("Peer P25–P75", justify="right")

    pe_range = (
        f"{_fmt_x(pc.peer_pe_p25)} – {_fmt_x(pc.peer_pe_p75)}"
        if pc.peer_pe_p25 is not None and pc.peer_pe_p75 is not None
        else "n/a"
    )
    table.add_row("P/E", _fmt_x(pc.subject_pe), _fmt_x(pc.peer_pe_median), pe_range)
    table.add_row("P/FCF", _fmt_x(pc.subject_pfcf), _fmt_x(pc.peer_pfcf_median), "")
    table.add_row("Gross margin", "—", _fmt_pct(pc.peer_gross_margin_median), "")
    table.add_row("ROIC", "—", _fmt_pct(pc.peer_roic_median), "")

    buf = io.StringIO()
    tmp = Console(file=buf, highlight=False)
    tmp.print(table)
    lines.append(buf.getvalue().rstrip())
    lines.append("")
    lines.append(
        "[dim]Reference context only — not a scoring input and not a valuation target. "
        "These are stocks that great value investors held in this category.[/dim]"
    )

    return Panel("\n".join(lines), title="[bold]Peer Comparison (same-category)[/bold]", border_style="cyan")


def _bull_bear_panel(cs: CompositeScore) -> Panel:
    """Assemble bull and bear cases from sub-score flags and reasons."""
    bull: list[str] = []
    bear: list[str] = []

    for sub in (cs.moat, cs.health, cs.valuation, cs.management):
        for r in sub.reasons:
            # Heuristic: reasons with high points awarded are positives
            try:
                pts_str = r.split("/")[0].lstrip("[+")
                pts = float(pts_str)
                max_str = r.split("/")[1].split("]")[0]
                max_pts = float(max_str)
            except (ValueError, IndexError):
                continue
            text = r.split("] ", 1)[1] if "] " in r else r
            if max_pts > 0 and pts / max_pts >= 0.7:
                bull.append(f"[green]+[/green] [{sub.name}] {text}")
            elif max_pts > 0 and pts / max_pts <= 0.3:
                bear.append(f"[red]−[/red] [{sub.name}] {text}")

        # Flags with warning language → bear
        for f in sub.flags:
            clean = f.lstrip("⚠ ").strip()
            keywords = ("premium", "expensive", "high expectations", "above estimated",
                        "exceeds", "distress", "dilut", "declining", "risk", "missing")
            if any(kw in clean.lower() for kw in keywords):
                bear.append(f"[red]−[/red] [{sub.name}] {clean}")

    lines: list[str] = []
    if bull:
        lines.append("[bold green]Bull case[/bold green]")
        for b in bull[:6]:
            lines.append(f"  {b}")
    if bear:
        if lines:
            lines.append("")
        lines.append("[bold red]Bear case[/bold red]")
        for b in bear[:6]:
            lines.append(f"  {b}")
    if not lines:
        lines.append("Insufficient flag data to assemble bull / bear case.")

    return Panel("\n".join(lines), title="[bold]Bull / Bear Summary[/bold]", border_style="white")


def _sizing_context_panel(cs: CompositeScore) -> Panel:
    """
    Position-sizing educational context.  Strictly analytical — no directives.
    """
    # Extract MoS from valuation flags
    mos_str = "unknown"
    for f in cs.valuation.flags:
        if "Margin of safety:" in f:
            # e.g. "⚠ Average IV estimate: ... | Margin of safety: +12.3%."
            try:
                mos_str = f.split("Margin of safety:")[1].strip().rstrip(".")
            except IndexError:
                pass
            break

    lines: list[str] = [
        "[bold]Position-sizing context (educational frameworks only)[/bold]",
        "",
        "Value investing literature offers several frameworks for thinking about "
        "how conviction and margin of safety relate to position sizing.  "
        "The numbers below are illustrative outputs of those frameworks — "
        "they are [bold]not[/bold] a recommendation of any specific allocation.",
        "",
        f"  Composite score:    {cs.composite:.1f}/100   (higher = more framework support)",
        f"  Implied margin of safety: {mos_str}",
        f"  Weight profile:     {cs.weight_profile}",
        "",
        "Graham / Buffett tradition: a margin of safety > 25% is generally considered "
        "a meaningful cushion for a stable business; higher uncertainty warrants a wider "
        "margin.  Position size is typically scaled with conviction: 1–3% equal-weight "
        "for exploratory ideas, 5–10% for high-conviction holdings in a concentrated "
        "portfolio.",
        "",
        "Kelly criterion (simplified): if the probability-weighted upside/downside ratio "
        "favours the investment, a partial-Kelly approach (e.g. ¼ Kelly) limits "
        "the position to avoid ruin from model error.",
        "",
        "These are frameworks for an investor's own analysis — each investor's risk "
        "tolerance, portfolio context, and tax situation differ.  Apply independent "
        "judgment.",
    ]

    return Panel("\n".join(lines), title="[bold]Position-Sizing Context[/bold]", border_style="dim")


def _ai_commentary_panel(commentary: str | None, ai_attempted: bool) -> Panel | None:
    """Return a visually distinct AI commentary panel, or None if not applicable."""
    if not ai_attempted:
        return None
    if commentary is None:
        return Panel(
            "[dim]AI commentary unavailable — ANTHROPIC_API_KEY not set or call failed. "
            "The report above is complete without it.[/dim]",
            title="[bold]AI Commentary[/bold]",
            border_style="dim",
        )
    return Panel(
        commentary,
        title="[bold magenta]AI Commentary[/bold magenta]",
        subtitle="[dim]Interpretation only — every figure comes from the structured analysis above[/dim]",
        border_style="magenta",
    )


def _disclaimer_panel() -> Panel:
    return Panel(
        f"[bold yellow]{DISCLAIMER_TEXT}[/bold yellow]",
        title="[bold red]Disclaimer[/bold red]",
        border_style="red",
    )


def _load_backtest_summary() -> dict | None:
    """Read ~/.value_analyzer/backtest_summary.json.

    Returns the parsed dict, or None if the file is absent or malformed.
    No import of the backtest layer — uses stdlib json only.
    """
    try:
        if not _BACKTEST_SUMMARY_PATH.exists():
            return None
        data = json.loads(_BACKTEST_SUMMARY_PATH.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("could not read backtest summary: %s", exc)
        return None


def _backtest_context_line() -> str:
    """Render the backtest-context line from the stored summary file.

    Reads ~/.value_analyzer/backtest_summary.json written by the backtest engine.
    Falls back to "not yet run" when the file is absent or unreadable.
    Never imports or triggers the backtest layer.
    """
    _CAVEAT = (
        "A validated backtest is evidence, not a guarantee — "
        "past edges decay and markets change."
    )

    summary = _load_backtest_summary()

    if summary is None:
        return (
            "[dim]Backtest not yet run — score is unvalidated. "
            "Run  value-analyzer --backtest  to generate out-of-sample context. "
            f"{_CAVEAT}[/dim]"
        )

    # Pull fields, tolerating missing keys in old/partial files
    run_date   = summary.get("run_date", "unknown date")
    date_range = summary.get("date_range", "unknown range")
    n_scored   = summary.get("n_scored")
    edge       = summary.get("q1_vs_benchmark_1y")   # float or None
    t_stat     = summary.get("t_stat_1y")             # float or None
    p_value    = summary.get("p_value_1y")            # float or None
    benchmark  = summary.get("benchmark_ticker", "index")

    # Edge vs benchmark
    if edge is not None:
        edge_str = f"top-quintile edge vs {benchmark} {edge * 100:+.1f}% after costs"
    else:
        edge_str = "top-quintile edge vs index: n/a"

    # Sample size
    n_str = f"n={n_scored}" if n_scored is not None else "n=unknown"

    # T-stat
    if t_stat is not None:
        t_str = f"t={t_stat:.2f}"
        if p_value is not None:
            t_str += f" (p={p_value:.2f})"
    else:
        t_str = "t=n/a"

    # Robustness judgement — explicit "not robust" when t < 2 or data thin
    robust = (
        t_stat is not None and abs(t_stat) >= 2.0
        and n_scored is not None and n_scored >= 30
    )
    if robust:
        robust_note = "edge above t=2 threshold, but small-sample caution still applies"
    else:
        robust_note = "edge NOT statistically robust — small sample or low t-stat"

    return (
        f"[dim]Backtest (run {run_date}): {edge_str}, {t_str}, "
        f"{n_str} over {date_range}. "
        f"{robust_note.capitalize()}. "
        f"{_CAVEAT}[/dim]"
    )


# ── Evaluation & framework context ────────────────────────────────────────────

def _evaluation_panel(cs: CompositeScore) -> Panel:
    """Synthesis of the composite result: confidence assessment + framework context.

    Part 1 — what the scores collectively say and how much to trust them.
    Part 2 — educational framework context; never a directive for this specific stock.
    """
    # ── Extract supporting data ────────────────────────────────────────────────
    real, total = cs.completeness_real, cs.completeness_total
    comp_pct = real / total if total > 0 else 1.0

    # Margin of safety from valuation flags
    mos_val: float | None = None
    mos_str = "not computed"
    for f in cs.valuation.flags:
        if "Margin of safety:" in f:
            try:
                raw = f.split("Margin of safety:")[1].strip().rstrip(".")
                mos_str = raw
                mos_val = float(raw.replace("%", "").replace("+", "")) / 100
            except (ValueError, IndexError):
                pass
            break

    # IV range from dispersion flag
    iv_range_str: str | None = None
    if cs.iv_dispersion_flag:
        m = re.search(r'\$([0-9.]+)[–\-]\$([0-9.]+)', cs.iv_dispersion_flag)
        if m:
            iv_range_str = f"${m.group(1)}–${m.group(2)}"

    # ── Confidence tier ────────────────────────────────────────────────────────
    confidence_issues: list[str] = []

    if total > 0 and comp_pct < COMPLETENESS_CAUTION_THRESHOLD:
        confidence_issues.append(
            f"data completeness {comp_pct:.0%} — {total - real} of {total} scoring "
            "inputs fell back to missing-data floors"
        )
    elif total > 0 and comp_pct < 0.85:
        confidence_issues.append(
            f"data completeness {comp_pct:.0%} — some inputs estimated from floors"
        )

    if cs.iv_dispersion_flag:
        spread_note = f" (range {iv_range_str})" if iv_range_str else ""
        confidence_issues.append(
            f"valuation methods disagree significantly{spread_note} — "
            "the average IV is unreliable as a single target; review individual method outputs"
        )

    if comp_pct < COMPLETENESS_CAUTION_THRESHOLD:
        tier, tier_color = "LOW", "red"
    elif not confidence_issues:
        tier, tier_color = "HIGH", "green"
    else:
        tier, tier_color = "MEDIUM", "yellow"

    # ── Quality and valuation reads ────────────────────────────────────────────
    quality_avg = (cs.moat.score + cs.health.score) / 2
    if quality_avg >= 70:
        quality_read = "high-quality business with strong fundamentals"
    elif quality_avg >= 50:
        quality_read = "above-average business quality with some limitations"
    elif quality_avg >= 35:
        quality_read = "mixed business-quality signals"
    else:
        quality_read = "below-average business-quality indicators"

    if mos_val is not None:
        if mos_val >= 0.25:
            val_read = f"meaningful margin of safety ({mos_str} below framework IV)"
        elif mos_val >= 0.10:
            val_read = f"moderate margin of safety ({mos_str} below framework IV)"
        elif mos_val >= 0:
            val_read = f"thin margin of safety ({mos_str} below framework IV)"
        else:
            val_read = f"priced above framework IV ({mos_str})"
    else:
        if cs.valuation.score >= 65:
            val_read = "appears attractively valued relative to framework IV estimates"
        elif cs.valuation.score >= 45:
            val_read = "roughly fairly valued relative to framework IV estimates"
        else:
            val_read = "appears fully valued or above framework IV estimates"

    # Strongest / weakest pillar
    subs = [
        (cs.moat, "moat"), (cs.health, "health"),
        (cs.valuation, "valuation"), (cs.management, "management"),
    ]
    subs_sorted = sorted(subs, key=lambda x: x[0].score)
    weakest_name, weakest_score = subs_sorted[0][1], subs_sorted[0][0].score
    strongest_name, strongest_score = subs_sorted[-1][1], subs_sorted[-1][0].score

    profile_note = {
        "compounder": "weighted toward moat quality and management",
        "stable":     "balanced weighting across all dimensions",
        "cyclical":   "weighted toward valuation and financial health",
        "declining":  "weighted toward financial survival and price",
    }.get(cs.weight_profile, "equally weighted across dimensions")

    # ── Build panel text ───────────────────────────────────────────────────────
    lines: list[str] = []

    # ── Part 1 ────────────────────────────────────────────────────────────────
    lines.append(
        "[bold]Part 1 — Evaluation of this analysis[/bold]"
    )
    lines.append("")

    lines.append(
        f"The composite of [bold]{cs.composite:.1f}/100[/bold] ({profile_note}) "
        f"reflects a {quality_read}. Valuation: {val_read}. "
        f"Strongest pillar: {strongest_name} ({strongest_score:.0f}/100); "
        f"weakest: {weakest_name} ({weakest_score:.0f}/100)."
    )
    lines.append("")

    lines.append(
        f"[bold]Confidence in this assessment: [{tier_color}]{tier}[/{tier_color}][/bold]"
    )
    if not confidence_issues:
        lines.append(
            f"  All {total} scoring inputs resolved from real filed data and valuation "
            "methods are in reasonable agreement. The score reflects actual business "
            "metrics rather than estimated floors."
        )
    else:
        for issue in confidence_issues:
            lines.append(f"  [{tier_color}]·[/{tier_color}] {issue[:1].upper()}{issue[1:]}.")
        lines.append(
            "  Treat the composite score and IV estimates as indicative. "
            "A wider margin of safety than usual is appropriate when confidence is not HIGH."
        )
    lines.append("")

    # Backtest calibration note — reads stored summary for dynamic data, falls
    # back to static language when no backtest has been run.
    summary = _load_backtest_summary()
    if summary:
        run_date = summary.get("run_date", "prior run")
        n_scored = summary.get("n_scored", "n/a")
        t1 = summary.get("t_stat_1y")
        t1_str = f"t={t1:.2f}" if t1 is not None else "t=n/a"
        lines.append(
            f"[dim]Backtest calibration (run {run_date}, {n_scored} scored pairs): "
            "this framework showed a Q1–Q5 spread near zero at 1-year horizons "
            f"({t1_str}, not statistically significant at α=0.10) but roughly +41% "
            "Q1-over-Q5 at 5-year horizons (hit rate 88%, t≈2.1, p≈0.07 on 8 df — "
            "borderline, small sample). "
            "The composite is a long-horizon analytical signal; it has no demonstrated "
            "predictive value at horizons under ~3 years on this dataset.[/dim]"
        )
    else:
        lines.append(
            "[dim]Backtest calibration: no validated backtest on file. "
            "When a backtest is available, this framework has shown signal mainly at "
            "5-year horizons and none at 1-year, on a small 43-ticker sample — "
            "borderline statistical significance. The composite is a long-horizon "
            "analytical signal, not a short-term or proven indicator.[/dim]"
        )
    lines.append("")

    # ── Part 2 ────────────────────────────────────────────────────────────────
    lines.append(
        "[bold]Part 2 — Framework context (educational; not an instruction for this stock)[/bold]"
    )
    lines.append("")

    lines.append(
        "[bold]Margin of safety:[/bold] "
        "The Graham tradition holds that paying materially less than intrinsic value "
        "provides a buffer against estimation error and adverse developments. "
        "An investor applying this framework would look for a gap of at least 25% below "
        "IV for a stable, high-quality business — and a wider gap (40%+) for an uncertain "
        "or cyclical one. "
        f"When confidence is [{tier_color}]{tier}[/{tier_color}], the required cushion "
        "grows: a wider spread between price and IV is needed to compensate for "
        "the added uncertainty in the estimate itself."
    )
    lines.append("")

    lines.append(
        "[bold]Conviction and position sizing:[/bold] "
        "Value investors approach sizing differently by school of thought. "
        "A concentrated approach (Buffett/Munger) treats highest-conviction ideas as "
        "potentially deserving larger weights (5–10% of a portfolio); exploratory or "
        "lower-conviction ideas receive smaller weights (1–3%). "
        "A mechanical equal-weight approach (Greenblatt) avoids overconfidence in any "
        "single estimate by spreading positions equally across a basket. "
        "An investor applying conviction-scaling would typically require higher composite "
        "scores and/or wider margins of safety before committing larger allocations — "
        "and would downgrade conviction when confidence is not HIGH."
    )
    lines.append("")

    lines.append(
        "[bold]Partial-Kelly as an overbetting guard:[/bold] "
        "The Kelly criterion derives a theoretically optimal bet fraction from estimated "
        "edge and win probability. Because any IV estimate carries model uncertainty, "
        "practitioners typically apply a fraction of full Kelly (¼ to ½) as insurance "
        "against being confidently wrong. "
        "An investor using partial-Kelly would treat the confidence tier as an input to "
        "their own edge estimate — lower confidence implies lower effective Kelly fraction."
    )
    lines.append("")

    lines.append(
        "[bold]Time horizon and diversification:[/bold] "
        "This framework's demonstrated properties are at 5-year horizons. "
        "An investor whose time horizon is shorter than that would be using the tool "
        "outside the window where it has shown any historical signal. "
        "Even at appropriate horizons, holding 10–20 diversified positions limits the "
        "damage from any single analytical or data error — and caps the cost of the "
        "inevitable cases where even a well-reasoned analysis proves incorrect."
    )
    lines.append("")

    lines.append(
        "[dim]These framework outputs are illustrative — not instructions for this stock. "
        "The composite score and IV estimates reflect stated analytical frameworks applied "
        "to public data under specific assumptions (WACC, growth rate, P/E multiples). "
        "They describe what the frameworks say about the business, not what the price "
        "will do. Risk tolerance, tax situation, existing portfolio, liquidity needs, and "
        "personal financial circumstances differ for every investor. "
        "The analytical decision and its financial consequences are the investor's own.[/dim]"
    )

    return Panel(
        "\n".join(lines),
        title="[bold]Evaluation & Framework Context[/bold]",
        border_style="cyan",
    )


# ── News section ──────────────────────────────────────────────────────────────

def _news_panel(news: "_NewsResult") -> Panel:
    """Render a panel showing recent news headlines."""
    lines: list[str] = []
    lines.append(
        "[dim italic]External context — does not affect the composite score "
        "or any valuation math.[/dim italic]"
    )
    lines.append(
        "Material events (capital raises, M&A, guidance changes, leadership changes) "
        "may post-date the latest SEC filing this analysis is based on. "
        "Weigh news against the filed fundamentals above using your own judgment."
    )
    lines.append("")

    for item in news.items[:8]:
        lines.append(
            f"[dim]{item.published_at}[/dim]  [bold]{item.source}[/bold]  {item.headline}"
        )

    lines.append("")
    lines.append(f"[dim]Provider: {news.provider}  |  Fetched: {news.fetched_at}[/dim]")

    return Panel("\n".join(lines), title="[bold]Recent News[/bold]", border_style="dim")


def _news_unavailable_line(news: "_NewsResult") -> str:
    """Return a dim line when news.error is set."""
    return f"[dim]Recent news: unavailable ({news.error})[/dim]"


# ── Public API ─────────────────────────────────────────────────────────────────

def render(
    cs: CompositeScore,
    *,
    console: Console | None = None,
    ai_commentary: str | None = None,
    ai_attempted: bool = False,
    news: "_NewsResult | None" = None,
) -> None:
    """Render *cs* to the terminal (or the supplied *console*)."""
    con = console or Console()

    con.print(_header_panel(cs))
    con.print()

    if cs.category.traces:
        con.print(_classification_table(cs))
        con.print()

    con.print(Rule("[bold]Sub-Scores[/bold]", style="blue"))
    con.print()

    weights = cs.weights_used
    for sub, key in (
        (cs.moat, "moat"),
        (cs.health, "health"),
        (cs.valuation, "valuation"),
        (cs.management, "management"),
    ):
        con.print(_subscore_panel(sub, weights.get(key, 0.25)))
        con.print()

    iv_table = _valuation_iv_table(cs)
    if iv_table is not None:
        con.print(iv_table)
        con.print()

    if cs.peer_comparison is not None and cs.peer_comparison.peer_count > 0:
        con.print(_peer_panel(cs.peer_comparison))
        con.print()

    con.print(_bull_bear_panel(cs))
    con.print()

    con.print(_sizing_context_panel(cs))
    con.print()

    con.print(_backtest_context_line())
    con.print()

    con.print(_evaluation_panel(cs))
    con.print()

    if news is not None:
        if news.available:
            con.print(_news_panel(news))
            con.print()
        else:
            con.print(_news_unavailable_line(news))
            con.print()

    ai_panel = _ai_commentary_panel(ai_commentary, ai_attempted)
    if ai_panel is not None:
        con.print(ai_panel)
        con.print()

    con.print(_disclaimer_panel())
    # Print the disclaimer text verbatim on a plain line so programmatic
    # checks can always find the exact string regardless of terminal width.
    con.print(DISCLAIMER_TEXT, markup=False, highlight=False)


def render_markdown(
    cs: CompositeScore,
    *,
    ai_commentary: str | None = None,
    ai_attempted: bool = False,
    news: "_NewsResult | None" = None,
) -> str:
    """Return a plain-text markdown representation of *cs*.

    Useful for piping to a file:  ``python -m value_analyzer.cli KO > ko.md``
    """
    buf = io.StringIO()
    con = Console(file=buf, highlight=False, markup=False, width=300)
    render(cs, console=con, ai_commentary=ai_commentary, ai_attempted=ai_attempted, news=news)
    return buf.getvalue()
