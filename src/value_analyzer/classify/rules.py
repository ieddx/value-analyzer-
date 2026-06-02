"""Explicit, readable classification rules.

DESIGN INTENT
─────────────
Every classification decision is expressed as a standalone function that returns
a (result, confidence, rationale) tuple.  Rationale is plain English explaining
which metric values fired which threshold.

To override a rule: subclass or monkey-patch the function, or just edit the
threshold constants at the top of this file.  Constants are named and documented
so you can find them by searching for the metric name.

THRESHOLD CONVENTIONS
─────────────────────
Thresholds are chosen to reflect typical ranges for large-cap US equities:
  - CAPEX: based on S&P 500 deciles for capex/revenue by sector
  - MARGINS: based on observed ranges for brand/commodity/software businesses
  - CAGR: 8% is roughly the long-run S&P 500 nominal growth rate
  - CV: 0.25 is where revenue swings become meaningfully cyclical

All thresholds are floats in natural units (e.g., 0.05 = 5%).
"""

from __future__ import annotations

from .models import (
    CapitalIntensity,
    GrowthProfile,
    MoatType,
    Metrics,
    RevenueType,
    RuleTrace,
    SicHint,
)

# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS  —  edit here to adjust classification sensitivity
# ══════════════════════════════════════════════════════════════════════════════

# Capital intensity (capex as % of revenue)
CAPEX_ASSET_LIGHT = 0.05    # < 5%  → asset_light  (software, beverage brands, insurance)
CAPEX_ASSET_HEAVY = 0.10    # > 10% → asset_heavy  (utilities, airlines, railroads, telco)
# Between 5–10% → moderate by default, can be overridden by SIC hint

# Revenue type
CYCLICAL_CV_THRESHOLD = 0.25   # revenue growth CV > 25% → cyclical signal
LOW_CV_THRESHOLD = 0.12        # revenue growth CV < 12% → stability signal
RECURRING_GM_FLOOR = 0.45      # gross margin > 45% + low CV → recurring label
# Below RECURRING_GM_FLOOR with any CV → transactional

# Moat
BRAND_GM_FLOOR = 0.50          # gross margin > 50% = pricing power = brand candidate
HIGH_GM_FLOOR = 0.65           # > 65% = switching-cost candidate (software-like economics)
ROIC_MOAT_CONFIRM = 0.12       # ROIC > 12% confirms a moat is real, not statistical noise
# A high ROIC without a margin signal → cost_advantage if SIC agrees, else none

# Growth profile
COMPOUNDER_CAGR = 0.08         # revenue CAGR > 8% → compounder
STABLE_CAGR_MIN = -0.02        # CAGR between −2% and +8% → stable
# Below −2% → declining
MIN_YEARS_FOR_CAGR = 3         # require ≥3 years of data to trust a CAGR


# ══════════════════════════════════════════════════════════════════════════════
# RULE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def classify_capital_intensity(
    m: Metrics, sic: SicHint
) -> tuple[CapitalIntensity, float, str]:
    """Classify capital intensity from capex/revenue ratio plus SIC hint.

    Primary signal: capex/revenue average over available years.
    SIC hint upgrades "moderate" to "asset_heavy" for industries where capex
    is structural (airlines, utilities, rails, telco) even if measured capex
    is borderline — these industries carry large maintenance capex burdens that
    sometimes appear lumpy in the data.

    Returns (classification, confidence 0–1, rationale string).
    """
    c = m.capex_pct_revenue

    if c is None:
        # No capex data — fall back to SIC hint if available
        if sic.capital_intensity:
            return (
                sic.capital_intensity,
                0.40,
                f"No capex data; using SIC {sic.sic_code} ({sic.sic_description}) hint.",
            )
        return (
            CapitalIntensity.moderate,
            0.20,
            "No capex data and no SIC hint; defaulting to moderate.",
        )

    pct_str = f"{c:.1%}"

    data_sparse = m.years_of_data < 3

    if c < CAPEX_ASSET_LIGHT:
        # Commodity-price booms can inflate revenue while the underlying mill/fleet
        # capex stays flat, making capex/revenue look deceptively low.  If SIC
        # strongly indicates heavy capital AND data is sparse, trust the SIC.
        if sic.capital_intensity == CapitalIntensity.asset_heavy and data_sparse:
            result = CapitalIntensity.asset_heavy
            conf = 0.50
            rationale = (
                f"capex/revenue = {pct_str} looks asset-light, but only "
                f"{m.years_of_data} year(s) of revenue data available "
                f"(commodity boom may inflate denominator). "
                f"SIC {sic.sic_code} ({sic.sic_description}) indicates structurally "
                "heavy capital industry — trusting SIC hint over single-year ratio."
            )
        else:
            result = CapitalIntensity.asset_light
            conf = min(0.95, 0.60 + (CAPEX_ASSET_LIGHT - c) / CAPEX_ASSET_LIGHT * 0.35)
            rationale = (
                f"capex/revenue = {pct_str} < {CAPEX_ASSET_LIGHT:.0%} threshold. "
                "Low physical reinvestment relative to revenue."
            )
    elif c > CAPEX_ASSET_HEAVY:
        result = CapitalIntensity.asset_heavy
        conf = min(0.95, 0.60 + (c - CAPEX_ASSET_HEAVY) / CAPEX_ASSET_HEAVY * 0.35)
        rationale = (
            f"capex/revenue = {pct_str} > {CAPEX_ASSET_HEAVY:.0%} threshold. "
            "High physical reinvestment required to maintain the business."
        )
    else:
        # Borderline: 5–10%. SIC hint can upgrade to asset_heavy.
        if sic.capital_intensity == CapitalIntensity.asset_heavy:
            result = CapitalIntensity.asset_heavy
            conf = 0.60
            rationale = (
                f"capex/revenue = {pct_str} is borderline (5–10%). "
                f"SIC {sic.sic_code} ({sic.sic_description}) indicates structurally "
                "heavy capex industry; classifying as asset_heavy."
            )
        elif sic.capital_intensity == CapitalIntensity.asset_light:
            result = CapitalIntensity.asset_light
            conf = 0.55
            rationale = (
                f"capex/revenue = {pct_str} is borderline (5–10%). "
                f"SIC {sic.sic_code} ({sic.sic_description}) suggests asset-light model; "
                "classifying as asset_light."
            )
        else:
            result = CapitalIntensity.moderate
            conf = 0.65
            rationale = (
                f"capex/revenue = {pct_str} falls in the moderate range (5–10%). "
                "No SIC override available."
            )

    return result, round(conf, 2), rationale


def classify_revenue_type(
    m: Metrics, sic: SicHint
) -> tuple[RevenueType, float, str]:
    """Classify revenue type using revenue volatility and gross margin as signals.

    Revenue growth coefficient of variation (CV) measures cyclicality:
    high CV = revenue swings driven by commodity prices or credit cycles.
    Gross margin floor separates recurring-revenue businesses (sticky pricing)
    from transactional ones (volume-driven).

    SIC hint overrides when CV is unavailable or ambiguous (< 3 years of data).
    """
    cv = m.revenue_growth_cv
    gm = m.gross_margin_avg

    # ── Cyclical check (highest priority) ─────────────────────────────────
    if cv is not None and cv >= CYCLICAL_CV_THRESHOLD:
        # Override: commodity businesses have thin margins.  A company with
        # gross margins > RECURRING_GM_FLOOR and a non-cyclical SIC is likely
        # experiencing structural revenue changes (refranchising, SaaS transition,
        # COVID shock) rather than true commodity-price cyclicality.  Do not
        # misclassify consumer staples or software companies as cyclical just
        # because they restructured their revenue model.
        margin_override = (
            gm is not None
            and gm > RECURRING_GM_FLOOR
            and sic.revenue_type != RevenueType.cyclical_commodity
        )
        if not margin_override:
            conf = min(0.90, 0.55 + (cv - CYCLICAL_CV_THRESHOLD) * 1.5)
            return (
                RevenueType.cyclical_commodity,
                round(conf, 2),
                f"Revenue growth CV = {cv:.2f} ≥ {CYCLICAL_CV_THRESHOLD} threshold. "
                "Revenue fluctuates materially year-to-year, consistent with commodity "
                "pricing exposure or credit-cycle sensitivity.",
            )
        # Margin override fired — fall through to recurring/transactional checks below.

    if sic.revenue_type == RevenueType.cyclical_commodity and (cv is None or cv > 0.15):
        # SIC strongly implies cyclical and data doesn't contradict it.
        return (
            RevenueType.cyclical_commodity,
            0.60,
            f"SIC {sic.sic_code} ({sic.sic_description}) implies commodity/cyclical revenue. "
            f"Revenue growth CV = {f'{cv:.2f}' if cv is not None else 'N/A'} does not contradict this.",
        )

    # ── Recurring check ───────────────────────────────────────────────────
    low_cv = cv is not None and cv < LOW_CV_THRESHOLD
    high_gm = gm is not None and gm > RECURRING_GM_FLOOR
    sic_recurring = sic.revenue_type == RevenueType.recurring

    if high_gm and (low_cv or sic_recurring):
        conf = 0.75
        reasons = []
        if high_gm:
            reasons.append(f"gross margin = {gm:.1%} > {RECURRING_GM_FLOOR:.0%}")
        if low_cv:
            reasons.append(f"revenue growth CV = {cv:.2f} < {LOW_CV_THRESHOLD} (stable)")
        if sic_recurring:
            reasons.append(
                f"SIC {sic.sic_code} ({sic.sic_description}) suggests recurring demand"
            )
        return (
            RevenueType.recurring,
            conf,
            "Stable, predictable revenue: " + "; ".join(reasons) + ".",
        )

    if sic_recurring and cv is None:
        return (
            RevenueType.recurring,
            0.45,
            f"SIC {sic.sic_code} implies recurring demand; insufficient data to confirm "
            "from financial metrics alone.",
        )

    # ── Default: transactional ────────────────────────────────────────────
    rationale_parts = []
    if cv is not None:
        rationale_parts.append(f"revenue growth CV = {cv:.2f}")
    if gm is not None:
        rationale_parts.append(f"gross margin = {gm:.1%}")
    rationale = (
        "No strong recurring or cyclical signal. "
        + ("; ".join(rationale_parts) + "." if rationale_parts else "Insufficient data.")
    )
    return RevenueType.transactional, 0.50, rationale


def classify_moat(
    m: Metrics, sic: SicHint
) -> tuple[MoatType, float, str]:
    """Classify competitive moat type from gross margin and SIC hint.

    Gross margin is the primary signal because it captures pricing power:
    - Very high margins (> 65%) suggest customers can't easily substitute
      (switching cost) or pay a premium for differentiation (brand/network).
    - High margins (50–65%) in consumer-facing industries = brand.
    - Moderate margins + cost-advantage SIC + ROIC > 12% = cost_advantage.
    - Low margins + no SIC moat signal = none.

    ROIC > ROIC_MOAT_CONFIRM is used as a secondary confirmation: a claimed
    moat that doesn't produce above-average returns on capital is suspect.
    """
    gm = m.gross_margin_avg
    roic = m.roic_avg

    roic_confirms = roic is not None and roic > ROIC_MOAT_CONFIRM

    # ── Switching cost (highest margin, often software/enterprise) ─────────
    if gm is not None and gm > HIGH_GM_FLOOR:
        if sic.moat_type == MoatType.switching_cost or roic_confirms:
            conf = 0.80 if roic_confirms else 0.65
            return (
                MoatType.switching_cost,
                conf,
                f"Gross margin = {gm:.1%} > {HIGH_GM_FLOOR:.0%} threshold. "
                + (f"ROIC = {roic:.1%} confirms above-average returns. " if roic_confirms else "")
                + (f"SIC {sic.sic_code} ({sic.sic_description}) corroborates. " if sic.moat_type == MoatType.switching_cost else ""),
            )

    # ── Brand moat ────────────────────────────────────────────────────────
    if gm is not None and gm > BRAND_GM_FLOOR:
        sic_brand = sic.moat_type == MoatType.brand
        conf = 0.80 if (sic_brand and roic_confirms) else (0.65 if sic_brand or roic_confirms else 0.55)
        reasons = [f"gross margin = {gm:.1%} > {BRAND_GM_FLOOR:.0%}"]
        if sic_brand:
            reasons.append(f"SIC {sic.sic_code} ({sic.sic_description}) = brand industry")
        if roic_confirms:
            reasons.append(f"ROIC = {roic:.1%} > {ROIC_MOAT_CONFIRM:.0%} confirms pricing power")
        return (
            MoatType.brand,
            conf,
            "; ".join(reasons) + ".",
        )

    # ── Network moat ──────────────────────────────────────────────────────
    if sic.moat_type == MoatType.network and gm is not None and gm > 0.35:
        return (
            MoatType.network,
            0.65,
            f"SIC {sic.sic_code} ({sic.sic_description}) indicates network-effects industry. "
            f"Gross margin = {gm:.1%} is consistent with network-moat economics.",
        )

    # ── Cost advantage ────────────────────────────────────────────────────
    if sic.moat_type == MoatType.cost_advantage and roic_confirms:
        return (
            MoatType.cost_advantage,
            0.65,
            f"SIC {sic.sic_code} ({sic.sic_description}) = cost-advantage industry. "
            f"ROIC = {roic:.1%} > {ROIC_MOAT_CONFIRM:.0%} confirms structural cost edge.",
        )

    if sic.moat_type == MoatType.cost_advantage:
        return (
            MoatType.cost_advantage,
            0.45,
            f"SIC {sic.sic_code} ({sic.sic_description}) = cost-advantage industry. "
            + (f"ROIC = {roic:.1%} does not confirm this (below {ROIC_MOAT_CONFIRM:.0%} threshold)." if roic is not None else "ROIC unavailable to confirm."),
        )

    # ── No evident moat ───────────────────────────────────────────────────
    reasons = []
    if gm is not None:
        reasons.append(f"gross margin = {gm:.1%} (below brand floor of {BRAND_GM_FLOOR:.0%})")
    if roic is not None:
        reasons.append(f"ROIC = {roic:.1%}")
    return (
        MoatType.none,
        0.60,
        "No moat signals detected. "
        + ("; ".join(reasons) + "." if reasons else "Insufficient data."),
    )


def classify_growth_profile(
    m: Metrics, sic: SicHint
) -> tuple[GrowthProfile, float, str]:
    """Classify growth profile from revenue CAGR.

    Requires at least MIN_YEARS_FOR_CAGR years of data; otherwise confidence
    is capped at 0.40 regardless of the result.

    SIC hint does not override growth profile — this must be measured from data.
    """
    cagr = m.revenue_cagr
    years = m.years_of_data
    data_limited = years < MIN_YEARS_FOR_CAGR

    if cagr is None:
        return (
            GrowthProfile.stable,
            0.15,
            "Revenue CAGR could not be computed (insufficient data). Defaulting to stable.",
        )

    cagr_str = f"{cagr:+.1%}"
    conf_cap = 0.40 if data_limited else 1.0

    if cagr > COMPOUNDER_CAGR:
        conf = min(0.90, 0.60 + (cagr - COMPOUNDER_CAGR) * 3.0)
        return (
            GrowthProfile.compounder,
            round(min(conf, conf_cap), 2),
            f"Revenue CAGR = {cagr_str} > {COMPOUNDER_CAGR:.0%} compounder threshold "
            f"over {years} year(s) of data."
            + (" (limited data — treat with caution)" if data_limited else ""),
        )

    if cagr >= STABLE_CAGR_MIN:
        conf = 0.70 - abs(cagr) * 2  # confidence highest near middle of stable range
        return (
            GrowthProfile.stable,
            round(min(max(conf, 0.40), conf_cap), 2),
            f"Revenue CAGR = {cagr_str} falls in stable range "
            f"({STABLE_CAGR_MIN:.0%} to {COMPOUNDER_CAGR:.0%}) "
            f"over {years} year(s) of data."
            + (" (limited data)" if data_limited else ""),
        )

    conf = min(0.85, 0.55 + abs(cagr - STABLE_CAGR_MIN) * 4.0)
    return (
        GrowthProfile.declining,
        round(min(conf, conf_cap), 2),
        f"Revenue CAGR = {cagr_str} < {STABLE_CAGR_MIN:.0%} declining threshold "
        f"over {years} year(s) of data."
        + (" (limited data)" if data_limited else ""),
    )


# ── Orchestrator ──────────────────────────────────────────────────────────

def apply_all_rules(m: Metrics, sic: SicHint) -> dict[str, RuleTrace]:
    """Run all four rules and return a traces dict keyed by dimension name."""
    traces: dict[str, RuleTrace] = {}

    for dim, fn, enum_cls in [
        ("capital_intensity", classify_capital_intensity, CapitalIntensity),
        ("revenue_type", classify_revenue_type, RevenueType),
        ("moat_type", classify_moat, MoatType),
        ("growth_profile", classify_growth_profile, GrowthProfile),
    ]:
        result, conf, rationale = fn(m, sic)  # type: ignore[operator]
        traces[dim] = RuleTrace(
            rule_name=fn.__name__,
            result=result.value,
            confidence=conf,
            rationale=rationale,
        )

    return traces
