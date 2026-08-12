"""Trust: is this record right, is it new, and did the pipeline notice a change?

Three questions the original pipeline could not answer about a stored row, and
the three modules here answer one each. Provenance — where it came from and
which tier produced it — lives on :class:`~models.Provenance`, attached to every
record at write time.
"""

from .dedupe import Deduplicator, DuplicateVerdict, hamming, simhash
from .drift import DriftAlert, DriftMonitor
from .validation import Rule, ValidationResult, validate_record

__all__ = [
    "Deduplicator",
    "DriftAlert",
    "DriftMonitor",
    "DuplicateVerdict",
    "Rule",
    "ValidationResult",
    "hamming",
    "simhash",
    "validate_record",
]
