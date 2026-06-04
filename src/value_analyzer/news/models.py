"""Pydantic models for the news layer."""
from __future__ import annotations
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    headline: str
    source: str
    published_at: date
    url: str = ""
    summary: str = ""


class NewsResult(BaseModel):
    ticker: str
    fetched_at: date
    provider: str
    items: list[NewsItem] = Field(default_factory=list)
    error: Optional[str] = None   # set when fetch failed/key missing

    @property
    def available(self) -> bool:
        return self.error is None and len(self.items) > 0
