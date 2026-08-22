"""Specialists, as tool subsets with a prompt each.

A star, not a mesh: specialists return to the supervisor and never talk to each
other. Two reasons, one measured and one practical. Fewer tools per decision is
strictly better for a small model — the gate scored 92% across twelve tools, and
every tool a specialist does not need is one more plausible wrong answer. And a
star produces a trace a person can read top to bottom, where a mesh produces a
transcript of two 3B models agreeing with each other.

A role is data, not a class hierarchy. It names the tools it may use, the prompt
that tells it what it is for, and what the supervisor should know when choosing
between them. Everything else — the loop, the budget, the registry — is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.agents.budget import Budget
from pipeline.agents.tools import Effect, catalog, describe_all


@dataclass(frozen=True, slots=True)
class Role:
    name: str
    #: What the supervisor reads when deciding whether this is the right one.
    purpose: str
    tools: tuple[str, ...]
    system: str
    #: The effect without which this role cannot do its job. Every role holds
    #: some read tools — acquisition polls tasks and checks the corpus — so
    #: "has any permitted tool" is not the test: an acquisition specialist that
    #: may only poll and profile is not an acquisition specialist.
    requires: Effect = Effect.READ

    def catalog(self, allowed: list[Effect]) -> list[dict]:
        """This role's tools, minus any the run's budget does not permit."""
        permitted = {spec.name for spec in catalog(allowed)}
        by_name = {tool["name"]: tool for tool in describe_all(allowed)}
        return [by_name[name] for name in self.tools if name in permitted and name in by_name]


CORPUS = Role(
    name="corpus",
    purpose=(
        "Search what is already indexed, and follow relationships in the "
        "knowledge graph. Everything the system already knows."
    ),
    tools=(
        "corpus_profile",
        "search_corpus",
        "answer_from_corpus",
        "graph_neighbors",
        "graph_path",
        "fetch_chunk",
    ),
    system="""\
You find evidence in an indexed corpus. You do not answer from your own
knowledge — only from what the tools return.

Work in this order. If you do not know what the corpus holds, call
corpus_profile first. To find passages, use search_corpus. To find how two
things relate, use the graph tools. When a search comes back thin, do not repeat
it: widen it instead — drop a filter, or try the words the documents would use
rather than the words the question used.

When a graph result names something you had not searched for, that is a lead:
search for it. Most questions worth asking need two steps, not one.

Stop when you have the evidence, and say what you found.\
""",
)


ACQUISITION = Role(
    name="acquisition",
    purpose=(
        "Fetch something the corpus does not have and add it, when a URL is "
        "known or discoverable. Slow, and reaches the outside world."
    ),
    tools=(
        "corpus_profile",
        "detect_url",
        "discover_sitemap",
        "extract_url",
        "index_document",
        "poll_task",
        "crawl_site",
    ),
    requires=Effect.WRITE,
    system="""\
You fetch things the corpus is missing.

Check detect_url before extracting: it tells you whether a URL is reachable,
what kind of resource it is, and whether it is already indexed. Do not extract
something already indexed.

Extraction is queued, not instant. extract_url returns a task id — poll it with
poll_task until it reports finished, and only then is the content searchable.

Prefer extract_url on a specific page. crawl_site walks someone else's whole
site and takes minutes; use it only when asked to.\
""",
)


ROLES: dict[str, Role] = {role.name: role for role in (CORPUS, ACQUISITION)}


def available(budget: Budget) -> list[Role]:
    """The roles a run with this budget can actually use.

    A role whose tools are all beyond the budget's effects is not offered as a
    choice that will then fail — it is simply not one of the options, the same
    principle the registry and the MCP server both follow.
    """
    allowed = budget.effects()
    return [
        role
        for role in ROLES.values()
        if role.requires in allowed and role.catalog(allowed)
    ]


def default_role(budget: Budget) -> Role:
    """Where a run starts. Always the corpus: the cheapest thing that might
    already hold the answer, and the only role available by default."""
    return CORPUS


__all__ = ["ACQUISITION", "CORPUS", "ROLES", "Role", "available", "default_role"]
