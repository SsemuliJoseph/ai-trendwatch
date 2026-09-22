"""Domain schemas (TrendItem). Pure data, zero I/O.
Leaf layer — imports NOTHING from other app packages.
"""

from app.models.trend_item import SourceType, TrendItem, canonicalize_url

__all__ = ["SourceType", "TrendItem", "canonicalize_url"]
