"""Embedding: chunks in, vectors out.

Kept deliberately import-light — pulling this package must not pull
sentence-transformers, because the io worker imports the task module and has no
business loading a transformer. The models are constructed inside the factories.
"""

from .orchestrator import DocumentEmbedder

__all__ = ["DocumentEmbedder"]
