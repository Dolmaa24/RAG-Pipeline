"""The extraction cascade and its tiers.

Ordered cheapest first: a content-hash cache, the publisher's own structured
data, a learned per-domain selector spec, and only then a model.
"""

from .cache import ExtractionCache, cache_key
from .cascade import ExtractionCascade, get_cascade
from .schema import coerce_to_schema, compile_schema, fill_rate, schema_hash, validate
from .selectors import SelectorRule, SelectorSpec, SpecStore, apply_spec, learn_spec
from .structured import harvest, map_to_schema

__all__ = [
    "ExtractionCache",
    "ExtractionCascade",
    "SelectorRule",
    "SelectorSpec",
    "SpecStore",
    "apply_spec",
    "cache_key",
    "coerce_to_schema",
    "compile_schema",
    "fill_rate",
    "get_cascade",
    "harvest",
    "learn_spec",
    "map_to_schema",
    "schema_hash",
    "validate",
]
