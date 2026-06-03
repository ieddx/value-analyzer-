"""Score layer configuration — all assumptions live here, nowhere else.

DESIGN INTENT
─────────────
Every number used in scoring has a name and a comment.  If you disagree with
a threshold, change it here; you do not need to touch the scorer files.

All monetary assumptions (WACC, terminal growth) are stated explicitly and
will be printed in the valuation output so the reader can apply their own
judgment.  This is not a black box.

CATEGORY WEIGHTS
────────────────
Weights express the relative importance of each dimension for a given
business type.  They must sum to 1.0 per profile.

  compounder  — moat quality and management dominate; buy at fair price.
  stable      — balanced; valuation matters more than for compounders.
  cyclical    — valuation and financial health dominate; moat is transient.
  declining   — survival (health) and price (valuation) are paramount.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY WEIGHT PROFILES  (must sum to 1.0 per profile)
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "compounder": {
        "moat":       0.35,   # durable advantage is the investment thesis
        "health":     0.15,   # compounders usually have strong balance sheets
        "valuation":  0.30,   # still need a reasonable entry price
        "management": 0.20,   # capital allocation makes or breaks compounding
    },
    "stable": {
        "moat":       0.25,
        "health":     0.25,
        "valuation":  0.30,   # price matters more for lower-growth businesses
        "management": 0.20,
    },
    "cyclical": {
        "moat":       0.15,   # cyclical moats are weaker/transient
        "health":     0.30,   # surviving the trough is everything
        "valuation":  0.40,   # buying the cycle right is the whole game
        "management": 0.15,
    },
    "declining": {
        "moat":       0.10,   # moat is eroding by definition
        "health":     0.40,   # can the business service debt through decline?
        "valuation":  0.40,   # only justified if price is deeply discounted
        "management": 0.10,
    },
    "default": {
        "moat":       0.25,
        "health":     0.25,
        "valuation":  0.25,
        "management": 0.25,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# VALUATION ASSUMPTIONS  (stated explicitly — override these as you see fit)
# ══════════════════════════════════════════════════════════════════════════════

WACC = 0.09
# 9% discount rate.  Represents a blend of:
#   equity cost (~10–11% for S&P 500 historical returns) and
#   debt cost (~5–6% pre-tax, after-tax ~4%).
# A higher WACC makes the stock look cheaper (lower implied growth required).
# Increase to 11% for more speculative / leveraged businesses.

TERMINAL_GROWTH = 0.025
# 2.5% perpetual growth rate — roughly nominal US GDP.
# Used in the reverse-DCF stage-2 terminal value.
# A company cannot grow faster than the economy forever.

PEER_PE: dict[str, float] = {
    # Rough category typical trailing P/E — used for relative context only.
    # These are long-run averages, NOT current market levels.
    "brand":           22.0,
    "network":         28.0,
    "switching_cost":  28.0,
    "cost_advantage":  16.0,
    "none":            15.0,
    "cyclical":        12.0,   # commodity/cyclical businesses
}

# ══════════════════════════════════════════════════════════════════════════════
# MOAT SCORING THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

MOAT_GM_EXCELLENT = 0.60   # > 60% gross margin — brand-level pricing power
MOAT_GM_GOOD      = 0.40   # > 40% — meaningful product differentiation
MOAT_GM_FAIR      = 0.20   # > 20% — thin but positive
# < 20% gross margin: commodity pricing, no structural advantage

MOAT_GM_CV_STABLE = 0.06   # gross-margin CV (std/avg) < 6% = very stable pricing
MOAT_GM_CV_OK     = 0.12   # < 12% = acceptable stability

MOAT_ROIC_EXCELLENT = 0.15  # > 15% ROIC = clear economic moat
MOAT_ROIC_GOOD      = 0.10  # > 10% = good returns above cost of capital
MOAT_ROIC_HURDLE    = 0.08  # > 8% = proxy cost of capital; below = value destruction

MOAT_ROIC_CONSISTENCY = 0.06  # ROIC std < 6% = consistently above hurdle

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH SCORING THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

HEALTH_DE_SAFE     = 0.50   # D/E < 0.5 — conservative balance sheet
HEALTH_DE_OK       = 1.50   # D/E < 1.5 — manageable leverage
HEALTH_DE_HIGH     = 3.00   # D/E > 3.0 — elevated risk

HEALTH_COVERAGE_SAFE = 10   # interest coverage > 10× — very comfortable
HEALTH_COVERAGE_OK   = 5    # > 5× — adequate
HEALTH_COVERAGE_LOW  = 3    # > 3× — watch carefully; below = distress risk

HEALTH_FCF_GOOD = 0.15      # FCF margin > 15% — strong cash generation
HEALTH_FCF_OK   = 0.07      # > 7% — adequate

# ══════════════════════════════════════════════════════════════════════════════
# VALUATION SCORING THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

VAL_PE_DISCOUNT    = 0.80   # current P/E < 80% of 10y median = trading at discount
VAL_PE_FAIR        = 1.10   # current P/E < 110% of median = roughly fair
VAL_PE_PREMIUM     = 1.30   # current P/E > 130% = meaningful premium

VAL_MOS_GOOD       = 0.25   # margin of safety > 25% = meaningful cushion
VAL_MOS_ADEQUATE   = 0.10   # > 10% = some cushion

VAL_IMPLIED_LOW    = 0.04   # reverse-DCF implied growth < 4% = priced conservatively
VAL_IMPLIED_OK     = 0.08   # < 8% = reasonable expectations
VAL_IMPLIED_HIGH   = 0.12   # > 12% = demanding expectations

# ══════════════════════════════════════════════════════════════════════════════
# MANAGEMENT SCORING THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

MGMT_BUYBACK_STRONG = -0.02  # share count CAGR < −2%/yr = active returning capital
MGMT_BUYBACK_OK     =  0.00  # flat or slight reduction = disciplined
MGMT_DILUTION_WARN  =  0.02  # > 2%/yr growth = watch (SBC, equity issuance)
MGMT_DILUTION_BAD   =  0.04  # > 4%/yr growth = material dilution

MGMT_ROE_EXCELLENT  = 0.20   # > 20% — exceptional capital allocation
MGMT_ROE_GOOD       = 0.12   # > 12% — strong
MGMT_ROE_OK         = 0.08   # > 8% — adequate

# ══════════════════════════════════════════════════════════════════════════════
# DATA COMPLETENESS AND CONFIDENCE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

COMPLETENESS_CAUTION_THRESHOLD = 0.70
# When real_inputs / total_inputs falls below this, the report header shows a
# low-confidence caution.  0.70 means ≥30% of scoring decisions fell back to
# missing-data floors.

VAL_IV_DISPERSION_RATIO = 2.5
# When the highest IV estimate exceeds the lowest by more than this multiple,
# the valuation section flags the disagreement.  A 2.5× spread means the
# "average IV" is unreliable as a single target; the investor should look at
# the individual method outputs rather than the mean.
