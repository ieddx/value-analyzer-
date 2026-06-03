"""Valuation score (0–100).

Components and max points
─────────────────────────
  pe_vs_history           20 pts  — current P/E vs this stock's own 10y median P/E
  pfcf_vs_history         20 pts  — current P/FCF vs own 10y median P/FCF
  margin_of_safety        35 pts  — intrinsic-value estimate vs current price
  reverse_dcf_growth      25 pts  — implied FCF growth rate the price embeds
  ─────────────────────────────────────────────────────────────────────
  Total max               100 pts

ASSUMPTIONS (all stated explicitly — see config.py for values)
──────────────────────────────────────────────────────────────
  WACC             = 9%    Weighted-average cost of capital
  Terminal growth  = 2.5%  Long-run FCF growth rate (≈ nominal GDP)
  Intrinsic value  = average of three approaches:
      1. No-growth earnings power:  EPSnorm / WACC
      2. Normalized-P/E reversion:  median_10y_PE × EPSnorm
      3. Graham Number:             sqrt(22.5 × EPS × BVPS)

  The reverse DCF uses the Gordon Growth Model (single-stage perpetuity):
      g_implied = WACC - FCF_per_share / price
  This is conservative (understates value for high-growth companies) and
  simple enough to audit by hand.

OUTPUT IS ANALYSIS — the margin-of-safety and implied-growth numbers are
context for an investor's own judgment, not buy/sell advice.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from value_analyzer.classify.models import Metrics, MoatType, Category
from value_analyzer.peers.models import CategoryPeerStats

from .config import (
    PEER_PE,
    VAL_IMPLIED_HIGH,
    VAL_IMPLIED_LOW,
    VAL_IMPLIED_OK,
    VAL_MOS_ADEQUATE,
    VAL_MOS_GOOD,
    VAL_PE_DISCOUNT,
    VAL_PE_FAIR,
    VAL_PE_PREMIUM,
    WACC,
    TERMINAL_GROWTH,
    VAL_IV_DISPERSION_RATIO,
)
from ._helpers import Scorer, annual_series, latest, price_at, safe_div
from .models import SubScore


def score_valuation(
    fund: pd.DataFrame,
    prices: pd.DataFrame,
    metrics: Metrics,
    category: Category,
    peer_stats: CategoryPeerStats | None = None,
) -> SubScore:
    """Compute the valuation sub-score.

    Parameters
    ----------
    fund:
        Point-in-time-filtered fundamentals.
    prices:
        Point-in-time-filtered price history.
    metrics:
        Pre-computed Metrics from the classify layer.
    category:
        Used to select the peer P/E fallback when no peer registry is available.
    peer_stats:
        Same-category peer aggregate stats from the peer registry.  When
        provided, the peer context flag uses actual same-category peers rather
        than the static PEER_PE averages.
    """
    s = Scorer("valuation")

    s.flag(f"Valuation assumptions: WACC = {WACC:.0%}, terminal growth = {TERMINAL_GROWTH:.1%}.")
    s.flag("All intrinsic-value estimates are analytical frameworks, not price targets.")

    # ── Current price and per-share figures ───────────────────────────────
    if prices.empty:
        s.flag("No price data available — valuation scoring skipped.")
        s.add(25, 100, "No price data; awarding neutral floor across all components.",
              data_available=False)
        return s.build()

    current_price = float(prices["close"].iloc[-1])
    price_date = prices.index[-1].date()

    eps_series = annual_series(fund, "eps_diluted")
    shares_series = annual_series(fund, "shares_outstanding")
    if shares_series.empty:
        shares_series = annual_series(fund, "shares_diluted")

    op_cf_series = annual_series(fund, "operating_cf")
    capex_series = annual_series(fund, "capex")
    equity_series = annual_series(fund, "equity")

    # Most-recent annual figures
    eps_latest = float(eps_series.iloc[-1]) if not eps_series.empty else None
    shares_latest = float(shares_series.iloc[-1]) if not shares_series.empty else None
    eq_latest = float(equity_series.iloc[-1]) if not equity_series.empty else None

    # FCF per share
    common = op_cf_series.index.intersection(capex_series.index)
    fcf_latest: float | None = None
    fcf_ps: float | None = None
    if not common.empty and shares_latest:
        fcf_latest = float(op_cf_series.loc[common].iloc[-1] - capex_series.abs().loc[common].iloc[-1])
        fcf_ps = fcf_latest / shares_latest if shares_latest > 0 else None

    # Book value per share
    bvps: float | None = safe_div(eq_latest, shares_latest) if eq_latest and shares_latest else None

    # Normalised EPS: 3-year average (more stable than trailing 1-year)
    eps_norm: float | None = None
    if len(eps_series) >= 3:
        eps_norm = float(eps_series.iloc[-3:].mean())
    elif not eps_series.empty:
        eps_norm = float(eps_series.iloc[-1])

    # ── 1. P/E vs own 10-year history (max 20) ────────────────────────────
    pe_current = safe_div(current_price, eps_latest) if eps_latest and eps_latest > 0 else None
    pe_history = _build_pe_history(eps_series, prices)
    pe_median = float(np.nanmedian(list(pe_history.values()))) if pe_history else None

    if pe_current is None or pe_median is None:
        s.flag("P/E history unavailable — skipping P/E vs history component.")
        s.add(8, 20, "P/E data unavailable; awarding neutral floor.", data_available=False)
    else:
        ratio = pe_current / pe_median
        s.flag(f"Current P/E = {pe_current:.1f}× | 10y median P/E = {pe_median:.1f}× | "
               f"ratio = {ratio:.2f}.")
        if ratio < VAL_PE_DISCOUNT:
            s.add(20, 20, f"Current P/E ({pe_current:.1f}×) is {ratio:.0%} of its 10y median "
                           f"({pe_median:.1f}×) — trading at a meaningful discount to own history.")
        elif ratio < VAL_PE_FAIR:
            pts = 10 + (VAL_PE_FAIR - ratio) / (VAL_PE_FAIR - VAL_PE_DISCOUNT) * 10
            s.add(pts, 20, f"Current P/E ({pe_current:.1f}×) near its 10y median "
                            f"({pe_median:.1f}×) — roughly fairly valued vs own history.")
        elif ratio < VAL_PE_PREMIUM:
            pts = max(3, 10 * (VAL_PE_PREMIUM - ratio) / (VAL_PE_PREMIUM - VAL_PE_FAIR))
            s.add(pts, 20, f"Current P/E ({pe_current:.1f}×) at {ratio:.0%} of its median — "
                            "slight premium to own history.")
        else:
            s.add(0, 20, f"Current P/E ({pe_current:.1f}×) is {ratio:.0%} of its 10y median "
                          f"({pe_median:.1f}×) — trading at a significant premium to own history.")

    # ── 2. P/FCF vs own 10-year history (max 20) ──────────────────────────
    pfcf_current = safe_div(current_price, fcf_ps) if fcf_ps and fcf_ps > 0 else None
    pfcf_history = _build_pfcf_history(op_cf_series, capex_series, shares_series, prices)
    pfcf_median = float(np.nanmedian(list(pfcf_history.values()))) if pfcf_history else None

    if pfcf_current is None or pfcf_median is None:
        s.flag("P/FCF history unavailable — skipping P/FCF vs history component.")
        s.add(8, 20, "P/FCF data unavailable; awarding neutral floor.", data_available=False)
    else:
        ratio = pfcf_current / pfcf_median
        s.flag(f"Current P/FCF = {pfcf_current:.1f}× | 10y median P/FCF = {pfcf_median:.1f}× | "
               f"ratio = {ratio:.2f}.")
        if ratio < VAL_PE_DISCOUNT:
            s.add(20, 20, f"Current P/FCF ({pfcf_current:.1f}×) is {ratio:.0%} of its median — "
                           "cash flow is cheap vs own history.")
        elif ratio < VAL_PE_FAIR:
            pts = 10 + (VAL_PE_FAIR - ratio) / (VAL_PE_FAIR - VAL_PE_DISCOUNT) * 10
            s.add(pts, 20, f"Current P/FCF ({pfcf_current:.1f}×) near its 10y median "
                            f"({pfcf_median:.1f}×) — fair value on cash-flow basis.")
        elif ratio < VAL_PE_PREMIUM:
            pts = max(3, 10 * (VAL_PE_PREMIUM - ratio) / (VAL_PE_PREMIUM - VAL_PE_FAIR))
            s.add(pts, 20, f"Current P/FCF ({pfcf_current:.1f}×) at slight premium "
                            f"to median ({pfcf_median:.1f}×).")
        else:
            s.add(0, 20, f"Current P/FCF ({pfcf_current:.1f}×) is {ratio:.0%} of median — "
                          "expensive relative to own cash-flow history.")

    # ── 3. Margin of safety — intrinsic value estimate (max 35) ──────────
    iv_estimates: list[tuple[str, float]] = []

    # Methods A–C require positive normalised EPS.
    # Method A: No-growth earnings power = normalised EPS / WACC
    if eps_norm and eps_norm > 0:
        iv_a = eps_norm / WACC
        iv_estimates.append(("No-growth earnings power (EPS_norm / WACC)", iv_a))

    # Method B: Normalized P/E reversion = median_10y_PE × EPS_norm
    if pe_median and eps_norm and eps_norm > 0:
        iv_b = pe_median * eps_norm
        iv_estimates.append((f"Normalised P/E reversion ({pe_median:.1f}× × EPS_norm)", iv_b))

    # Method C: Graham Number = sqrt(22.5 × EPS × BVPS)
    if eps_norm and eps_norm > 0 and bvps and bvps > 0:
        iv_c = math.sqrt(22.5 * eps_norm * bvps)
        iv_estimates.append((f"Graham Number (√(22.5 × {eps_norm:.2f} × {bvps:.2f}))", iv_c))

    # ── FIX 1: when EPS methods produced nothing, explain why and try fallbacks ──
    if not iv_estimates:
        # Correct error message: distinguish absent data from negative earnings.
        if eps_norm is None:
            s.flag("IV estimates unavailable — EPS data absent.")
        else:
            s.flag(
                f"IV estimates unavailable via earnings methods — normalised EPS is "
                f"negative ({eps_norm:.2f}), likely non-cash impairment; "
                "earnings-based frameworks require positive earnings."
            )

        # ── FIX 2: P/B-reversion fallback (used when BVPS is available) ──────────
        # Computes median historical P/B ratio and applies it to current book value.
        # Only invoked when EPS-based methods produced no estimates.
        if bvps and bvps > 0:
            pb_history = _build_pb_history(equity_series, shares_series, prices)
            pb_median = float(np.nanmedian(list(pb_history.values()))) if pb_history else None
            if pb_median and pb_median > 0:
                iv_d = pb_median * bvps
                iv_estimates.append((
                    f"P/B reversion ({pb_median:.2f}× median P/B × BVPS ${bvps:.2f}; "
                    f"{len(pb_history)}-yr P/B history)",
                    iv_d,
                ))
                s.flag(
                    f"P/B fallback: median historical P/B = {pb_median:.2f}× "
                    f"(over {len(pb_history)} years) × BVPS ${bvps:.2f} = ${iv_d:.2f}. "
                    "Book-value multiples are sensitive to leverage, asset mix, and "
                    "intangible write-downs — treat as a rough floor, not a precise target."
                )

        # ── FIX 3: No-growth FCF earnings-power fallback ─────────────────────────
        # IV = FCF_per_share / WACC.  Conservative floor: zero perpetual FCF growth.
        # Only invoked when EPS-based methods produced no estimates.
        if fcf_ps and fcf_ps > 0:
            iv_e = fcf_ps / WACC
            iv_estimates.append((
                f"No-growth FCF earnings power "
                f"(FCF_ps ${fcf_ps:.2f} / WACC {WACC:.0%})",
                iv_e,
            ))
            s.flag(
                f"FCF fallback: FCF/share ${fcf_ps:.2f} ÷ WACC {WACC:.0%} = ${iv_e:.2f}. "
                "Conservative floor — assumes zero perpetual FCF growth. "
                "Appropriate when accounting earnings are distorted by non-cash charges "
                "such as goodwill impairment."
            )

    # ── Dispersion check — flag when IV methods disagree strongly ────────────
    if len(iv_estimates) >= 2:
        iv_vals = [v for _, v in iv_estimates]
        iv_hi, iv_lo = max(iv_vals), min(iv_vals)
        if iv_lo > 0 and iv_hi / iv_lo > VAL_IV_DISPERSION_RATIO:
            s.flag(
                f"IV_DISPERSION: Valuation methods disagree significantly "
                f"(${iv_lo:.2f}–${iv_hi:.2f}, {iv_hi / iv_lo:.1f}× spread) — "
                "treat the average IV as low-confidence. "
                "Review individual method outputs rather than relying on the mean."
            )

    if not iv_estimates:
        # All methods exhausted — award floor score.
        s.add(10, 35, "IV estimates unavailable; awarding conservative floor.",
              data_available=False)
    else:
        iv_avg = float(np.mean([v for _, v in iv_estimates]))
        mos = (iv_avg - current_price) / current_price

        for label, iv in iv_estimates:
            s.flag(f"IV estimate ({label}): ${iv:.2f} — "
                   f"{'discount' if iv > current_price else 'premium'} to current price ${current_price:.2f}.")
        s.flag(f"Average IV estimate: ${iv_avg:.2f} | Current price: ${current_price:.2f} | "
               f"Margin of safety: {mos:+.1%}.")

        if mos >= VAL_MOS_GOOD:
            s.add(35, 35, f"Margin of safety = {mos:+.1%} (avg IV ${iv_avg:.2f} vs price ${current_price:.2f}) — "
                           f"meaningful discount to estimated intrinsic value.")
        elif mos >= VAL_MOS_ADEQUATE:
            pts = 18 + (mos - VAL_MOS_ADEQUATE) / (VAL_MOS_GOOD - VAL_MOS_ADEQUATE) * 17
            s.add(pts, 35, f"Margin of safety = {mos:+.1%} — some cushion; "
                            "upside depends on how close reality matches the model assumptions.")
        elif mos >= 0:
            pts = 8 + (mos / VAL_MOS_ADEQUATE) * 10
            s.add(pts, 35, f"Margin of safety = {mos:+.1%} — thin cushion; "
                            "priced near estimated intrinsic value.")
        else:
            pts = max(0, 8 + mos * 30)  # mos is negative here
            s.add(pts, 35, f"Margin of safety = {mos:+.1%} — stock appears to be trading "
                            "above the estimated intrinsic value under these assumptions.")

    # ── 4. Reverse-DCF implied growth rate (max 25) ───────────────────────
    # Gordon Growth Model: Price = FCF_ps × (1+g) / (r-g)
    # Solving for g: g = (r × P - FCF_ps) / (P + FCF_ps)
    if fcf_ps and fcf_ps > 0 and current_price > 0:
        g_implied = (WACC * current_price - fcf_ps) / (current_price + fcf_ps)

        # Sanity: if implied g > r, the formula breaks down (price extremely high)
        if g_implied >= WACC:
            s.add(0, 25, f"Reverse-DCF implied growth rate ≥ WACC ({WACC:.0%}) — "
                          "price implies perpetual growth at or above the discount rate, "
                          "which is mathematically unsustainable.")
            s.flag(f"Implied FCF growth ≥ WACC; price may be extremely elevated.")
        else:
            s.flag(f"Reverse-DCF (Gordon Growth): price ${current_price:.2f} implies FCF "
                   f"must grow at {g_implied:.1%}/yr in perpetuity (WACC = {WACC:.0%}).")

            if g_implied < 0:
                s.add(25, 25, f"Implied FCF growth = {g_implied:.1%}/yr — market is pricing "
                               "in declining cash flows; if the business stabilises, significant "
                               "upside may exist.")
            elif g_implied < VAL_IMPLIED_LOW:
                s.add(22, 25, f"Implied FCF growth = {g_implied:.1%}/yr < {VAL_IMPLIED_LOW:.0%} — "
                               "low bar; the business does not need exceptional growth "
                               "to justify the current price.")
            elif g_implied < VAL_IMPLIED_OK:
                pts = 14 + (VAL_IMPLIED_OK - g_implied) / (VAL_IMPLIED_OK - VAL_IMPLIED_LOW) * 8
                s.add(pts, 25, f"Implied FCF growth = {g_implied:.1%}/yr — "
                                "reasonable expectations embedded in the price.")
            elif g_implied < VAL_IMPLIED_HIGH:
                pts = max(4, 14 * (VAL_IMPLIED_HIGH - g_implied) / (VAL_IMPLIED_HIGH - VAL_IMPLIED_OK))
                s.add(pts, 25, f"Implied FCF growth = {g_implied:.1%}/yr — "
                                "demanding expectations; growth must be sustained to justify price.")
            else:
                s.add(0, 25, f"Implied FCF growth = {g_implied:.1%}/yr > {VAL_IMPLIED_HIGH:.0%} — "
                               "very high expectations baked in; significant execution risk.")
                s.flag(f"Reverse-DCF implies {g_implied:.1%}/yr FCF growth in perpetuity — "
                       "verify this is achievable given business fundamentals.")

        # Peer P/E context — same-category peers when available, static fallback otherwise
        if peer_stats is not None and pe_current is not None:
            _flag_same_category_peers(s, peer_stats, pe_current)
        elif pe_current is not None:
            _flag_static_peer_pe(s, category, pe_current)

    else:
        fcf_reason = "FCF per share unavailable" if not fcf_ps else f"FCF per share = ${fcf_ps:.2f} (≤ 0)"
        s.flag(f"Reverse-DCF skipped: {fcf_reason}.")
        s.add(8, 25, "Reverse-DCF unavailable; awarding conservative floor.", data_available=False)

    return s.build()


# ── Private helpers ────────────────────────────────────────────────────────

def _flag_same_category_peers(
    s: "Scorer",
    peer_stats: CategoryPeerStats,
    pe_current: float,
) -> None:
    n = len(peer_stats.peer_tickers)
    profile = peer_stats.weight_profile
    if peer_stats.pe_median is not None:
        range_str = (
            f" (P25 {peer_stats.pe_p25:.1f}× – P75 {peer_stats.pe_p75:.1f}×)"
            if peer_stats.pe_p25 is not None and peer_stats.pe_p75 is not None
            else ""
        )
        s.flag(
            f"Same-category peer context ({n} {profile} stocks from value investor "
            f"portfolios): P/E median {peer_stats.pe_median:.1f}×{range_str}; "
            f"current P/E {pe_current:.1f}×. "
            "Reference only — not a valuation target."
        )
    else:
        s.flag(
            f"Same-category peer registry present ({n} {profile} stocks) "
            "but P/E data unavailable for peers."
        )


def _flag_static_peer_pe(s: "Scorer", category, pe_current: float) -> None:
    moat_key = category.moat_type.value
    peer_pe = PEER_PE.get(moat_key, PEER_PE.get("none", 15.0))
    if category.revenue_type.value == "cyclical_commodity":
        peer_pe = PEER_PE["cyclical"]
    s.flag(
        f"Peer context (approx. long-run typical P/E for '{moat_key}' moat businesses: "
        f"{peer_pe:.0f}×; current P/E = {pe_current:.1f}×). "
        "No same-category peer registry — using static reference. "
        "Run build_peer_registry() to enable same-category peers."
    )


def _build_pe_history(
    eps_series: pd.Series, prices: pd.DataFrame
) -> dict[int, float]:
    """Return {year: P/E} for years where we have both EPS and a year-end price."""
    result: dict[int, float] = {}
    if eps_series.empty or prices.empty:
        return result
    for year, eps in eps_series.items():
        if eps <= 0:
            continue
        ye_price = price_at(prices, pd.Timestamp(f"{year}-12-31"))
        if ye_price:
            result[int(year)] = ye_price / eps
    return result


def _build_pb_history(
    equity_series: pd.Series,
    shares_series: pd.Series,
    prices: pd.DataFrame,
) -> dict[int, float]:
    """Return {year: P/B} for years where equity, shares, and a year-end price are available.

    Used as a fallback IV input when normalised EPS is negative.
    """
    result: dict[int, float] = {}
    if equity_series.empty or shares_series.empty or prices.empty:
        return result
    common = equity_series.index.intersection(shares_series.index)
    for year in common:
        eq = float(equity_series.loc[year])
        sh = float(shares_series.loc[year])
        if sh <= 0 or eq <= 0:
            continue
        bvps_yr = eq / sh
        ye_price = price_at(prices, pd.Timestamp(f"{year}-12-31"))
        if ye_price and bvps_yr > 0:
            result[int(year)] = ye_price / bvps_yr
    return result


def _build_pfcf_history(
    op_cf: pd.Series,
    capex: pd.Series,
    shares: pd.Series,
    prices: pd.DataFrame,
) -> dict[int, float]:
    """Return {year: P/FCF} for years where all inputs are available."""
    result: dict[int, float] = {}
    if op_cf.empty or capex.empty or shares.empty or prices.empty:
        return result
    common = op_cf.index.intersection(capex.index).intersection(shares.index)
    for year in common:
        fcf = float(op_cf.loc[year]) - abs(float(capex.loc[year]))
        sh = float(shares.loc[year])
        if sh <= 0 or fcf <= 0:
            continue
        fcf_ps = fcf / sh
        ye_price = price_at(prices, pd.Timestamp(f"{year}-12-31"))
        if ye_price:
            result[int(year)] = ye_price / fcf_ps
    return result
