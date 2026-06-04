"""Optional AI commentary layer.

Interprets the already-computed CompositeScore; never supplies numbers itself.

API key
-------
Set the environment variable ``ANTHROPIC_API_KEY`` before running.  If the key
is absent or the call fails the function returns ``None`` and the report
renders normally without commentary.

Model
-----
claude-opus-4-8 — current Anthropic flagship, verified against SDK docs.
Adaptive thinking enabled; streaming used to prevent request timeouts.

Rules enforced via system prompt
---------------------------------
* Interpret only — every number must already appear in the structured data
  passed by the caller.  The model must not invent figures.
* No directive language (buy/sell/hold/avoid).
* Analysis framing: "if an investor applied framework X, the result would be Y."
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import TYPE_CHECKING

from value_analyzer.score.models import CompositeScore

if TYPE_CHECKING:
    from value_analyzer.news.models import NewsResult

logger = logging.getLogger(__name__)

# ── Stable system prompt (cached) ─────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an educational value-investing analysis assistant.

Your role is to provide clear, plain-English interpretation of quantitative
analysis that has already been computed for you.  You MUST follow these rules:

1. INTERPRET, do not SOURCE.
   Every number you reference must come from the structured data provided by
   the caller.  You must NEVER invent, estimate, or look up figures not
   present in the input.  If a figure is missing, say so explicitly.

2. No directive language.
   Do not use "buy", "sell", "hold", "avoid", "strong buy", or any equivalent
   phrasing.  Framing must always be: "if an investor applied framework X, the
   result would be Y."  This is educational analysis, not personalised advice.

3. Be concise and honest.
   Aim for 3–5 short paragraphs.  Highlight what the numbers reveal, flag
   where the analysis is limited or data is missing, and note any tension
   between sub-scores (e.g. a high moat score but poor valuation).

4. Respect uncertainty.
   Intrinsic-value estimates rest on stated assumptions (WACC, terminal
   growth).  Acknowledge that the output is framework-dependent and that
   reasonable investors may use different assumptions.
"""

# ── Prompt builder ─────────────────────────────────────────────────────────────

def _news_section(news: "NewsResult") -> str:
    """Return the markdown news headlines block to append to the user message."""
    lines: list[str] = [
        f"### Recent news headlines (last 30 days, via {news.provider})",
    ]
    for item in news.items:
        lines.append(f"{item.published_at}  {item.source}: {item.headline}")
    lines += [
        "",
        "**Task for news section**: Identify any apparent MATERIAL events among the "
        "headlines above (equity offering or dilution, M&A activity, major guidance change, "
        "significant litigation, leadership change) that could affect the investment thesis "
        "but would not yet appear in the filed fundamentals analyzed above. If no material "
        "event is apparent, say so briefly. Interpret only what the headlines state — "
        "do NOT invent or speculate beyond them. Do NOT issue buy, sell, or hold directives.",
    ]
    return "\n".join(lines)


def _build_user_message(cs: CompositeScore, news: "NewsResult | None" = None) -> str:
    cat = cs.category
    lines: list[str] = [
        f"## Analysis summary for {cs.ticker} — as of {cs.as_of_date}",
        "",
        f"**Composite score**: {cs.composite:.1f}/100  "
        f"(weight profile: {cs.weight_profile})",
        "",
        "### Classification",
        f"- Moat type: {cat.moat_type.value}",
        f"- Revenue type: {cat.revenue_type.value}",
        f"- Growth profile: {cat.growth_profile.value}",
        f"- Capital intensity: {cat.capital_intensity.value}",
        "",
        "Classification rationale (from rule traces):",
    ]
    for dim, trace in cat.traces.items():
        lines.append(
            f"  - {dim}: {trace.result} "
            f"(confidence {trace.confidence:.0%}) — {trace.rationale}"
        )

    weights = cs.weights_used
    lines += [
        "",
        "### Sub-scores",
        f"Weights: moat {weights['moat']:.0%} | health {weights['health']:.0%} | "
        f"valuation {weights['valuation']:.0%} | management {weights['management']:.0%}",
        "",
    ]

    for sub in (cs.moat, cs.health, cs.valuation, cs.management):
        lines.append(f"**{sub.name.title()}**: {sub.score:.1f}/100")
        for r in sub.reasons:
            lines.append(f"  - {r}")
        for f in sub.flags:
            lines.append(f"  - {f}")
        lines.append("")

    if cs.peer_comparison is not None and cs.peer_comparison.peer_count > 0:
        pc = cs.peer_comparison
        lines += [
            "### Peer comparison (same-category, from value investor 13F filings)",
            f"Profile: {pc.weight_profile} | Peer count: {pc.peer_count}",
        ]
        if pc.subject_pe is not None:
            lines.append(f"Subject P/E: {pc.subject_pe:.1f}× | "
                         f"Peer P/E median: {pc.peer_pe_median:.1f}×" if pc.peer_pe_median else
                         f"Subject P/E: {pc.subject_pe:.1f}×")
        if pc.subject_pfcf is not None:
            lines.append(f"Subject P/FCF: {pc.subject_pfcf:.1f}×")
        if pc.peer_roic_median is not None:
            lines.append(f"Peer ROIC median: {pc.peer_roic_median:.1%}")
        lines.append("")

    lines += [
        "### Task",
        "Write a concise qualitative interpretation of the above analysis.",
        "Reference only the numbers shown above — do not introduce any new figures.",
        "Follow the rules in the system prompt: no directive language, analysis not advice.",
    ]

    if news is not None and news.available:
        lines.append("")
        lines.append(_news_section(news))

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_commentary(cs: CompositeScore, news: "NewsResult | None" = None) -> str | None:
    """Return a qualitative interpretation of *cs*, or ``None`` on any failure.

    Requires ``ANTHROPIC_API_KEY`` in the environment.  If the key is absent or
    the API call fails for any reason the function logs a warning and returns
    ``None`` — the caller's report continues unaffected.

    Numbers always come from *cs*; the model is instructed not to invent figures.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set — skipping AI commentary")
        return None

    try:
        import anthropic  # import inside function so the module loads without SDK
    except ImportError:
        logger.warning("anthropic package not installed — skipping AI commentary")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Stream to avoid HTTP timeouts on longer outputs.
        # Prompt caching on the stable system prompt: cache_control on the
        # system block means repeated calls for different tickers reuse the
        # cached system prefix, paying only for the per-ticker user message.
        with client.messages.stream(
            model="claude-opus-4-8",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": _build_user_message(cs, news=news),
            }],
        ) as stream:
            message = stream.get_final_message()

        # Extract the text blocks only (skip thinking blocks)
        parts = [
            block.text
            for block in message.content
            if block.type == "text"
        ]
        return "\n".join(parts).strip() or None

    except Exception as exc:
        logger.warning("AI commentary failed (%s: %s) — continuing without it",
                       type(exc).__name__, exc)
        return None
