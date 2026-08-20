"""GLiNER entity extraction: dedup, descriptions, and the fallback."""

from __future__ import annotations

import pytest

from pipeline.graph import ner

TEXT = (
    "Project Titan is an autonomous vehicle initiative started by Apple in 2014. "
    "Doug Field was hired to lead Project Titan in 2018. "
    "Before joining Apple, Doug Field worked at Tesla."
)


class _FakeGliner:
    """Returns spans the way GLiNER does — one per mention, not per name."""

    def __init__(self, spans=None):
        self._spans = spans
        self.calls = 0

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls += 1
        if self._spans is not None:
            return self._spans
        return [
            {"text": "Project Titan", "label": "project", "score": 0.98,
             "start": text.index("Project Titan")},
            {"text": "Apple", "label": "organization", "score": 0.96,
             "start": text.index("Apple")},
            {"text": "Doug Field", "label": "person", "score": 0.99,
             "start": text.index("Doug Field")},
            # A second, higher-scoring mention of a name already seen.
            {"text": "Doug Field", "label": "person", "score": 1.00,
             "start": text.rindex("Doug Field")},
            {"text": "Tesla", "label": "organization", "score": 0.90,
             "start": text.index("Tesla")},
        ]


def test_each_name_appears_once(monkeypatch):
    """GLiNER returns a span per mention; the graph wants a node per name."""
    entities = ner.extract_entities(TEXT, model=_FakeGliner())
    names = [e.name for e in entities]
    assert names.count("Doug Field") == 1
    assert set(names) == {"Project Titan", "Apple", "Doug Field", "Tesla"}


def test_labels_map_to_graph_types():
    entities = {e.name: e.type for e in ner.extract_entities(TEXT, model=_FakeGliner())}
    assert entities["Doug Field"] == "Person"
    assert entities["Apple"] == "Organization"
    assert entities["Project Titan"] == "Project"


def test_an_unrecognised_label_becomes_unknown():
    spans = [{"text": "Thing", "label": "sasquatch", "score": 0.9, "start": 0}]
    [entity] = ner.extract_entities(TEXT, model=_FakeGliner(spans))
    assert entity.type == "Unknown"


def test_the_description_is_a_real_sentence_from_the_text():
    """Quoted, not paraphrased — it cannot hallucinate and it cites its source."""
    entities = {e.name: e.description for e in ner.extract_entities(TEXT, model=_FakeGliner())}
    assert entities["Project Titan"].startswith("Project Titan is an autonomous vehicle")
    assert entities["Project Titan"] in TEXT.replace("  ", " ")
    assert entities["Tesla"].startswith("Before joining Apple")


def test_the_description_comes_from_the_first_mention():
    """Where a name was introduced says more than where it recurs."""
    entities = {e.name: e.description for e in ner.extract_entities(TEXT, model=_FakeGliner())}
    assert entities["Doug Field"].startswith("Doug Field was hired")


def test_provenance_is_attached():
    [entity] = ner.extract_entities(
        TEXT,
        model=_FakeGliner([{"text": "Apple", "label": "organization", "score": 0.9, "start": 0}]),
        source_url="https://example.com/a",
        content_hash="abc",
    )
    assert entity.source_url == "https://example.com/a"
    assert entity.content_hash == "abc"


def test_empty_text_makes_no_call():
    model = _FakeGliner()
    assert ner.extract_entities("   ", model=model) == []
    assert model.calls == 0


def test_blank_span_text_is_skipped():
    spans = [{"text": "  ", "label": "person", "score": 0.9, "start": 0}]
    assert ner.extract_entities(TEXT, model=_FakeGliner(spans)) == []


# --------------------------------------------------------------------------- #
# The extractor's use of it
# --------------------------------------------------------------------------- #


def test_the_model_is_only_asked_for_relationships(fake_backend, monkeypatch):
    """GLiNER supplies entities, so the model emits edges and nothing else."""
    from config import config
    from pipeline.graph.cache import GraphCache
    from pipeline.graph.extractor import GraphExtractor

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "gliner")
    monkeypatch.setattr(ner, "get_model", lambda *a, **k: _FakeGliner())

    backend = fake_backend(
        {
            "relationships": [
                {"source": "Doug Field", "target": "Tesla", "relation": "WORKED_AT",
                 "description": ""}
            ]
        }
    )
    result = GraphExtractor(backend=backend, cache=GraphCache()).extract(TEXT)

    assert {e.name for e in result.entities} == {
        "Project Titan", "Apple", "Doug Field", "Tesla"
    }
    assert (result.relationships[0].source, result.relationships[0].target) == (
        "Doug Field",
        "Tesla",
    )
    # The prompt carried the entity list rather than asking for one.
    assert backend.calls == 1


def test_a_failed_relationship_call_keeps_the_entities(fake_backend, monkeypatch):
    """The entities are real; a later document may connect them."""
    from config import config
    from pipeline.graph.cache import GraphCache
    from pipeline.graph.extractor import GraphExtractor

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "gliner")
    monkeypatch.setattr(ner, "get_model", lambda *a, **k: _FakeGliner())

    backend = fake_backend(raises=RuntimeError("rate limited"))
    result = GraphExtractor(backend=backend, cache=GraphCache()).extract(TEXT)

    assert result.entities
    assert result.relationships == []


def test_missing_gliner_falls_back_to_the_model(fake_backend, monkeypatch):
    """An optional dependency must not fail a job."""
    from config import config
    from errors import MissingDependency
    from pipeline.graph.cache import GraphCache
    from pipeline.graph.extractor import GraphExtractor

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "gliner")

    def _absent(*a, **k):
        raise MissingDependency("gliner", "GLiNER entity extraction")

    monkeypatch.setattr(ner, "extract_entities", _absent)

    backend = fake_backend(
        {
            "entities": [{"name": "Apple", "type": "Organization", "description": "a firm"}],
            "relationships": [],
        }
    )
    result = GraphExtractor(backend=backend, cache=GraphCache()).extract(TEXT)
    assert [e.name for e in result.entities] == ["Apple"]


@pytest.mark.slow
def test_the_real_model_finds_the_entities():
    """Downloads ~200 MB on first run."""
    entities = ner.extract_entities(TEXT)
    names = {e.name for e in entities}
    assert {"Project Titan", "Apple", "Doug Field", "Tesla"} <= names
