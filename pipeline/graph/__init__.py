"""The knowledge graph: extraction, resolution, storage and traversal.

Ported from ``Dolmaa24/GraphRAG``. Import-light on purpose — nothing here pulls
Kuzu, LanceDB or a model until something actually builds or queries a graph.
"""

from .schema import Entity, KnowledgeGraphExtraction, Relationship, Triple

__all__ = ["Entity", "KnowledgeGraphExtraction", "Relationship", "Triple"]
