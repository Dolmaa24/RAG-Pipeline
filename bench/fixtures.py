"""The corpus bench/agents.py expects, built from scratch.

A benchmark that only runs against whatever happens to be indexed measures the
corpus as much as the code. These two documents are small, contain facts of
every shape the cases ask about — a relationship, a figure, a location, a
person — and between them span two documents so a question can require both.

Indexing is done in-process rather than through the queue: no workers needed,
and a failure is a traceback here rather than a task that quietly did not run.

Run:  PYTHONPATH=. ./venv/bin/python -m bench.fixtures
      PYTHONPATH=. ./venv/bin/python -m bench.fixtures --graph
"""

from __future__ import annotations

import argparse

DOCUMENTS: dict[str, str] = {
    "bench://quarterly-report": (
        "Quarterly Report\n\n"
        "Acme Corporation acquired Beta Industries in March 2026.\n"
        "Revenue for the quarter was 42.5 million dollars.\n"
    ),
    "bench://board-memo": (
        "Northwind Logistics - Board Memo\n\n"
        "Northwind Logistics acquired Fabrikam Freight in June 2026.\n"
        "The purchase price was 310 million euros.\n"
        "Fabrikam Freight is headquartered in Rotterdam.\n"
        "Priya Raman was appointed chief executive of the combined group.\n"
    ),
}


def seed(*, build_graph: bool) -> None:
    from pipeline.index import index_text

    for source, text in DOCUMENTS.items():
        report = index_text(text, source=source, extra_metadata={"department": "bench"})
        print(f"indexed {source}: {report.chunks} chunk(s)")

        if build_graph:
            from pipeline.graph.builder import build_graph as build

            result = build(text, source_url=source)
            print(
                f"  graph: {result.entities_extracted} entities, "
                f"{result.relationships} relationships"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", action="store_true",
        help="also extract entities and relationships (slow; needs a model)",
    )
    args = parser.parse_args()
    seed(build_graph=args.graph)
    print("\nNow run: PYTHONPATH=. ./venv/bin/python -m bench.agents")


if __name__ == "__main__":
    main()
