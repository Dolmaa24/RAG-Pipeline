"""The agent layer: tools an agent may call, and eventually the loop that calls them.

Nothing here is imported by the extraction or retrieval pipeline. The dependency
runs one way — agents know about the pipeline, the pipeline does not know about
agents — so the system keeps working exactly as it does today if this package is
never loaded.
"""
