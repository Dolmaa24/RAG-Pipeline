"""Compare extraction models on the jobs this pipeline actually gives them.

Not a leaderboard. Four checks, each one a failure this pipeline has really
hit with llama3.2:3b:

1. **Entity recall** — are the entities the text names actually found?
2. **Edge direction** — ``Northwind ACQUIRED Fabrikam`` and the reverse are not
   the same fact, and a graph that gets it backwards answers questions wrongly
   with complete confidence.
3. **Alias judgement** — is ``Apple Inc.`` the same entity as ``Apple``? Suffix
   stripping handles that case now, so the model is asked the harder ones it is
   still the only screen for.
4. **Cypher validity** — can it write a traversal query the database will
   actually accept? Checked against Kuzu's own parser, not a regex: an earlier
   version of this benchmark used only the static guards and scored a model 2/2
   on queries that Kuzu then rejected outright.

Run:  PYTHONPATH=. ./venv/bin/python -m bench.graph_models
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from pipeline.extract.llm.ollama import OllamaBackend


# --------------------------------------------------------------------------- #
# The corpus, with the answers written down
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    name: str
    text: str
    #: Entities the text plainly names. Matched case-insensitively on substring,
    #: so "Doug Field" counts whether or not the model added a title.
    entities: list[str]
    #: (source, relation-ish, target) the text states, in that direction.
    #: The relation is matched loosely — what is being scored is the direction.
    edges: list[tuple[str, str, str]] = field(default_factory=list)


CASES = [
    Case(
        name="acquisition",
        text=(
            "Northwind Traders acquired Fabrikam Ltd in 2025 for twelve million "
            "dollars. Marcus Webb, the chief financial officer of Northwind, led "
            "the transaction."
        ),
        entities=["Northwind Traders", "Fabrikam", "Marcus Webb"],
        edges=[("Northwind", "acquir", "Fabrikam")],
    ),
    Case(
        name="career",
        text=(
            "Project Titan is an autonomous vehicle initiative started by Apple "
            "in 2014. Doug Field was hired to lead Project Titan in 2018. Before "
            "joining Apple, Doug Field worked at Tesla as Senior VP of "
            "Engineering, reporting to Elon Musk."
        ),
        entities=["Project Titan", "Apple", "Doug Field", "Tesla", "Elon Musk"],
        edges=[
            ("Apple", "start", "Project Titan"),
            ("Doug Field", "work", "Tesla"),
        ],
    ),
    Case(
        name="employment",
        text=(
            "Priya Raman, VP of People Operations, introduced the remote work "
            "policy at Northwind Traders in 2025. She reports to the chief "
            "executive, Alan Doyle."
        ),
        entities=["Priya Raman", "Northwind Traders", "Alan Doyle"],
        edges=[("Priya Raman", "report", "Alan Doyle")],
    ),
]

#: Traversal prompts, with the seed sets they arrive with. The seeds matter as
#: much as the question: the same model writes a query Kuzu accepts for one set
#: and one it rejects for another, so a single prompt overstates the rate.
CYPHER_CASES = [
    ("What was Doug Field's trajectory before joining Ford?", ["Doug Field", "Northwind Traders"]),
    ("Which companies did Northwind Traders acquire?", ["Doug Field", "Northwind Traders"]),
    ("What was Doug Field's trajectory before joining Ford?", ["Tesla", "Doug Field", "Elon Musk"]),
    ("Who reports to Alan Doyle?", ["Alan Doyle"]),
    ("How is Marcus Webb connected to Fabrikam?", ["Marcus Webb", "Fabrikam Ltd"]),
]

#: Pairs the suffix rule cannot decide, so the model is the only screen.
ALIAS_CASES = [
    # (name a, name b, are they the same thing?)
    ("Apple Computer", "Apple", "the technology company", "the technology company", True),
    ("IBM", "International Business Machines", "the computing company", "the computing company", True),
    ("Apple Records", "Apple", "the record label founded by the Beatles", "the technology company", False),
    ("Jordan", "Michael Jordan", "a country in the Middle East", "a basketball player", False),
]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _found(needle: str, haystack: list[str]) -> bool:
    low = needle.lower()
    return any(low in item.lower() or item.lower() in low for item in haystack)


def _edge_matches(edge: tuple[str, str, str], relationships: list) -> Optional[bool]:
    """True if present and correctly directed, False if reversed, None if absent."""
    source, relation, target = (part.lower() for part in edge)
    for rel in relationships:
        s = (rel.source or "").lower()
        t = (rel.target or "").lower()
        r = (rel.relation or "").lower()
        if relation not in r:
            continue
        if source in s and target in t:
            return True
        if source in t and target in s:
            return False
    return None


@dataclass
class Score:
    model: str
    entities_found: int = 0
    entities_total: int = 0
    edges_right: int = 0
    edges_reversed: int = 0
    edges_missing: int = 0
    #: Edges beyond the ones the corpus asks about. Not all are wrong — the text
    #: states more than the two or three facts written down per case — but
    #: fabrication shows up here and nowhere else, so it is worth seeing.
    edges_extra: int = 0
    alias_right: int = 0
    alias_total: int = 0
    cypher_valid: int = 0
    cypher_total: int = 0
    seconds: float = 0.0
    tokens_out: int = 0

    def report(self) -> str:
        entity_rate = self.entities_found / max(1, self.entities_total)
        alias_rate = self.alias_right / max(1, self.alias_total)
        edges_seen = self.edges_right + self.edges_reversed
        return (
            f"{self.model:<18} "
            f"entities {self.entities_found}/{self.entities_total} ({entity_rate:.0%})  "
            f"edges ok {self.edges_right} rev {self.edges_reversed} "
            f"miss {self.edges_missing} extra {self.edges_extra}  "
            f"alias {self.alias_right}/{self.alias_total} ({alias_rate:.0%})  "
            f"cypher {self.cypher_valid}/{self.cypher_total}  "
            f"{self.seconds:.1f}s total"
        )


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def evaluate(model: str) -> Score:
    from pipeline.graph.cache import GraphCache
    from pipeline.graph.cypher import CypherAgent, is_read_only, is_well_formed
    from pipeline.graph.extractor import GraphExtractor
    from pipeline.graph.resolution import EntityResolver
    from pipeline.graph.schema import Entity

    backend = OllamaBackend(model=model)
    score = Score(model=model)
    started = time.perf_counter()

    # A cache would defeat the point of timing this.
    extractor = GraphExtractor(backend=backend, cache=_NullCache())

    for case in CASES:
        result = extractor.extract(case.text, source_url=f"bench://{case.name}")
        names = [e.name for e in result.entities]

        score.entities_total += len(case.entities)
        score.entities_found += sum(1 for want in case.entities if _found(want, names))

        for edge in case.edges:
            verdict = _edge_matches(edge, result.relationships)
            if verdict is True:
                score.edges_right += 1
            elif verdict is False:
                score.edges_reversed += 1
            else:
                score.edges_missing += 1

        score.edges_extra += max(0, len(result.relationships) - len(case.edges))

    resolver = EntityResolver(backend=backend, embedder=_NullEmbedder())
    for a, b, desc_a, desc_b, same in ALIAS_CASES:
        score.alias_total += 1
        verdict = resolver._same_entity(
            Entity(name=a, type="Organization", description=desc_a),
            {"name": b, "type": "Organization", "description": desc_b},
        )
        if verdict == same:
            score.alias_right += 1

    agent = CypherAgent(backend=backend)
    with _scratch_graph() as connection:
        for question, seeds in CYPHER_CASES:
            score.cypher_total += 1
            query = agent.generate(question, list(seeds), max_hops=2)
            if not query or not is_read_only(query) or not is_well_formed(query):
                continue
            try:
                connection.execute(query)
            except Exception:
                continue  # the guards passed; the database did not
            score.cypher_valid += 1

    score.seconds = time.perf_counter() - started
    return score


@contextmanager
def _scratch_graph():
    """An empty graph with the real schema, to parse candidate queries against."""
    from pipeline.graph.store import GraphStore

    directory = tempfile.mkdtemp()
    store = GraphStore(db_path=f"{directory}/kuzu")
    try:
        yield store.read_connection
    finally:
        store.close()
        shutil.rmtree(directory, ignore_errors=True)


class _NullCache:
    def get(self, *a, **k):
        return None

    def put(self, *a, **k):
        return None


class _NullEmbedder:
    """Forces every alias pair past the similarity screen to the model."""

    model_name = "null"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["llama3.2:3b", "qwen2.5:3b"])
    args = parser.parse_args()

    print(f"{len(CASES)} documents, {len(ALIAS_CASES)} alias pairs, {len(CYPHER_CASES)} Cypher prompts\n")
    scores = []
    for model in args.models or ["llama3.2:3b", "qwen2.5:3b"]:
        print(f"running {model} ...", flush=True)
        scores.append(evaluate(model))

    print()
    for score in scores:
        print("  " + score.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
