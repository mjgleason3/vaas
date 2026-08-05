"""VAAS: a lightweight visual memory and attention layer for agents."""

from .models import AttentionSignal, EntityMatch, MediaRecord, SearchHit
from .pipeline import VAAS

__all__ = ["VAAS", "AttentionSignal", "EntityMatch", "MediaRecord", "SearchHit"]
__version__ = "0.1.0"
