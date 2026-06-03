"""Peers layer — reference set of value investor holdings, classified by category.

Public API
----------
build_peer_registry(as_of_date)  — offline build; classifies seed tickers, writes cache
get_peer_stats(profile, date)    — fast load from cache; returns CategoryPeerStats or None
build_peer_comparison(...)       — creates PeerComparison for the report layer
"""

from .models import CategoryPeerStats, PeerComparison, PeerSnapshot
from .registry import build_peer_comparison, build_peer_registry, get_peer_stats

__all__ = [
    "build_peer_registry",
    "get_peer_stats",
    "build_peer_comparison",
    "PeerSnapshot",
    "CategoryPeerStats",
    "PeerComparison",
]
