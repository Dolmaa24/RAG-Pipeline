"""Streamlit dashboard.
"""

from __future__ import annotations

import json
import time

import requests
import streamlit as st

from config import config

API = config.API_BASE_URL.rstrip("/")
POLL_INTERVAL = 1.5
POLL_TIMEOUT = 3600

st.set_page_config(page_title="Web Scraper", layout="wide")
st.title("Web Scraper")
st.caption("Point it at anything — a page, a PDF, a spreadsheet, a feed, an archive, a podcast, a live stream.")

def api_get(path: str, **params):
    try:
        response = requests.get(f"{API}{path}", params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API} ({exc})")
        return None


def api_post(path: str, payload: dict, timeout: int = 120):
    """POST and return the body, or None with the reason already surfaced."""
    try:
        response = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:200]
        st.error(f"{exc.response.status_code}: {detail}")
        return None
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API} ({exc})")
        return None

# Sidebar
with st.sidebar:
    st.header("System")
    health = api_get("/health")
    if health:
        workers = health.get("workers_online", 0)
        st.metric("Workers online", workers)
        if workers == 0:
            st.warning("No Celery workers. Jobs will queue but never run.")
        depth = health.get("queue_depth") or {}
        if depth:
            st.caption("Queue depth")
            for queue, count in depth.items():
                st.text(f"  {queue}: {count}")

    st.divider()
    st.header("Search index")
    stats = api_get("/api/v1/index/stats")
    if stats and stats.get("available"):
        st.metric("Indexed chunks", f"{stats.get('chunks', 0):,}")
    elif stats:
        st.metric("Indexed chunks", 0)
        st.caption("Nothing indexed yet. Run an extraction with indexing on.")

    graph = api_get("/api/v1/graph/entities", limit=1)
    if graph and graph.get("available"):
        counts = graph.get("counts") or {}
        st.metric("Graph entities", f"{counts.get('entities', 0):,}")
        st.caption(f"{counts.get('relationships', 0):,} relationships")
    else:
        st.metric("Graph entities", 0)
        st.caption("Build one with 'Knowledge graph' on.")

# Main Interface
extract_tab, search_tab = st.tabs(["Extract", "Search"])

with extract_tab:
    left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "png", "jpg", "jpeg", "csv", "xlsx", "txt"])
    url = st.text_input("Or enter a URL", "https://quotes.toscrape.com/")
    
    prompt = st.text_area("What to extract", "Extract the main items and their key fields.", height=120)
    schema_text = st.text_area(
        "Schema",
        json.dumps({"title": "string", "summary": "string"}, indent=2),
        height=300,
    )

    with st.expander("Options"):
        crawl_depth = st.number_input("Max crawl depth (0 for single page)", 0, 10, 0)
        crawl_budget = st.number_input("Max URLs (for crawls)", 1, 50000, 200)

        st.caption("Retrieval")
        do_index = st.checkbox(
            "Index for search", value=True,
            help="Chunk, embed and store the text so the Search tab can find it.",
        )
        do_graph = st.checkbox(
            "Build knowledge graph", value=False,
            help="Extract entities and relationships. One model call per chunk, "
                 "so this is much slower than indexing alone.",
        )
        st.caption("Provenance — filters you can search on later")
        meta_department = st.text_input("Department", "", placeholder="finance")
        meta_region = st.text_input("Region", "", placeholder="EMEA")
        meta_permission = st.text_input("Permission level", "", placeholder="internal")

    start = st.button("Start Processing", use_container_width=True, type="primary")

with right:
    st.subheader("Progress")

    if start:
        try:
            schema = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            st.error(f"Schema is not valid JSON: {exc}")
            st.stop()

        target_url = None
        if uploaded_file is not None:
            with st.spinner("Uploading file..."):
                try:
                    response = requests.post(f"{API}/api/v1/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}, timeout=30)
                    response.raise_for_status()
                    target_url = response.json().get("url")
                except Exception as exc:
                    st.error(f"Failed to upload file: {exc}")
                    st.stop()
        elif url:
            target_url = url
        else:
            st.warning("Please upload a file or enter a URL.")
            st.stop()

        if crawl_depth > 0:
            # Run Crawl
            payload = {
                "start_url": target_url,
                "prompt": prompt,
                "schema_template": schema,
                "collect_extensions": [],
                "max_depth": int(crawl_depth),
                "max_pages": int(crawl_budget),
                "same_site": True,
            }
            try:
                response = requests.post(f"{API}/api/v1/crawl", json=payload, timeout=20)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()
            
            started = response.json()
            crawl_id = started["crawl_id"]
            st.info(started["plan"])
            
            status_box = st.empty()
            metrics_box = st.empty()
            
            began = time.time()
            while time.time() - began < POLL_TIMEOUT:
                state = api_get(f"/api/v1/crawls/{crawl_id}")
                if state is None:
                    break
                
                with metrics_box.container():
                    a, b, c, d = st.columns(4)
                    a.metric("Claimed", state.get("claimed", 0))
                    b.metric("Fetched", state.get("fetched", 0))
                    c.metric("Collected", state.get("collected", 0))
                    d.metric("Failed", state.get("failed", 0))

                if state.get("done"):
                    status_box.success("Crawl finished.")
                    break
                status_box.info(f"Crawling... {state.get('in_flight', 0)} in flight")
                time.sleep(POLL_INTERVAL)
            
            if st.button("Stop this crawl"):
                requests.delete(f"{API}/api/v1/crawls/{crawl_id}", timeout=10)
                st.warning("Stop requested.")
        else:
            # Run Extract
            try:
                response = requests.post(
                    f"{API}/api/v1/extract",
                    json={
                        "url": target_url,
                        "prompt": prompt,
                        "schema_template": schema,
                        "force_dynamic": False,
                        "local_only": False,
                        "follow_children": True,
                        "allowed_tiers": None,
                        "index": bool(do_index),
                        "build_graph": bool(do_graph),
                        "metadata": {
                            k: v for k, v in (
                                ("department", meta_department.strip()),
                                ("region", meta_region.strip()),
                                ("permission_level", meta_permission.strip()),
                            ) if v
                        } or None,
                    },
                    timeout=20,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()

            queued = response.json()
            task_id = queued["task_id"]
            st.info(f"Queued task: `{task_id}`")

            status_box = st.empty()
            progress = st.progress(0.0)
            started = time.time()

            while time.time() - started < POLL_TIMEOUT:
                state = api_get(f"/api/v1/tasks/{task_id}")
                if state is None:
                    break
                status = state.get("status")

                if status in ("PENDING", "RECEIVED"):
                    status_box.info("Waiting for a worker...")
                elif status == "PROGRESS":
                    status_box.info(state.get("stage", "Working..."))
                    progress.progress(min(0.9, (time.time() - started) / 60))
                elif status == "RETRY":
                    status_box.warning("Retrying...")
                elif status == "SUCCESS":
                    progress.progress(1.0)
                    result = state["result"]
                    status_box.success("Done")

                    a, b, c = st.columns(3)
                    a.metric("Kind", result.get("kind", "?"))
                    b.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                    c.metric("Total", f"{result.get('timings_ms', {}).get('total', 0) / 1000:.1f}s")

                    if result.get("validation_failures"):
                        st.warning("Validation flagged this record:")
                        for failure in result["validation_failures"]:
                            st.text(f"  • {failure}")

                    st.json(result.get("extracted_data") or {})

                    if result.get("children"):
                        st.subheader(f"{len(result['children'])} nested item(s)")
                        for child in result["children"]:
                            with st.expander(child["url"].split("!/")[-1]):
                                st.json(child.get("extracted_data") or {"error": child.get("error")})

                    with st.expander("Run report"):
                        st.json(result)

                    break
                elif status == "FAILURE":
                    progress.empty()
                    status_box.error(state.get("error", "Task failed"))
                    break

                time.sleep(POLL_INTERVAL)


# --------------------------------------------------------------------------- #
# Search — the other half of the pipeline
# --------------------------------------------------------------------------- #

with search_tab:
    st.subheader("Ask the corpus")
    st.caption(
        "Ask a question and get an answer that cites the documents it came from, "
        "or turn that off and read the passages yourself."
    )

    query = st.text_input(
        "Question", "", placeholder="What did the Q1 report say about revenue?",
        label_visibility="collapsed",
    )

    controls, filters = st.columns([1, 1])

    with controls:
        with st.expander("How to search", expanded=False):
            limit = st.slider("Results", 1, 25, 5)
            fusion = st.radio(
                "Fusion", ["rrf", "alpha"], horizontal=True,
                help="RRF combines by rank, so the two legs' unrelated score "
                     "scales cannot distort it. Alpha weights them directly.",
            )
            alpha = st.slider(
                "Dense weight", 0.0, 1.0, 0.7, 0.05,
                disabled=(fusion != "alpha"),
                help="1.0 is pure vector search, 0.0 is pure BM25.",
            )
            use_graph = st.checkbox("Include knowledge graph", value=True)
            rerank = st.checkbox(
                "Rerank with a cross-encoder", value=False,
                help="More accurate, and loads another model.",
            )
            rewrite = st.checkbox(
                "Rewrite the question", value=True,
                help="One model call splits the question, extracts filters and "
                     "generalises it. Skipped automatically for simple questions.",
            )

    with filters:
        with st.expander("Narrow it down", expanded=False):
            st.caption("Filters run before the search, so a narrow filter still fills the results.")
            index_stats = api_get("/api/v1/index/stats") or {}
            available = index_stats.get("filters") or {}

            def pick(field: str, label: str):
                options = available.get(field) or []
                if not options:
                    return []
                return st.multiselect(label, options, default=[])

            f_doc_type = pick("doc_type", "Document type")
            f_department = pick("department", "Department")
            f_author = pick("author", "Author")
            f_region = pick("region", "Region")
            f_permission = pick("permission_level", "Permission level")
            f_language = pick("language", "Language")
            date_from = st.text_input("From (YYYY-MM-DD)", "")
            date_to = st.text_input("To (YYYY-MM-DD)", "")

            if not available:
                st.caption("No filter values yet — they appear once something is indexed.")

    mode = st.radio(
        "Mode", ["Answer the question", "Just show me the passages"],
        horizontal=True, label_visibility="collapsed",
    )
    wants_answer = mode.startswith("Answer")

    go = st.button(
        "Ask" if wants_answer else "Search", type="primary", use_container_width=True
    )

    if go and not query.strip():
        st.warning("Type a question first.")
    elif go:
        payload = {
            "query": query,
            "limit": int(limit),
            "fusion": fusion,
            "alpha": float(alpha),
            "use_graph": bool(use_graph),
            "rerank": bool(rerank),
            "rewrite": bool(rewrite),
        }
        chosen = {
            "doc_type": f_doc_type, "department": f_department, "author": f_author,
            "region": f_region, "permission_level": f_permission, "language": f_language,
        }
        applied = {k: v for k, v in chosen.items() if v}
        if date_from.strip():
            applied["date_from"] = date_from.strip()
        if date_to.strip():
            applied["date_to"] = date_to.strip()
        if applied:
            payload["filters"] = applied

        reply = None
        if wants_answer:
            with st.spinner("Retrieving, then answering..."):
                reply = api_post("/api/v1/answer", payload, timeout=600)
            result = (reply or {}).get("retrieval")
        else:
            with st.spinner("Searching..."):
                result = api_post("/api/v1/search", payload)

        if reply is not None:
            if reply.get("sufficient"):
                st.success(reply.get("answer") or "")
            elif reply.get("answer"):
                st.warning(reply.get("answer"))
                st.caption(
                    "Marked as not fully answered by the indexed documents — "
                    "read the sources below before relying on it."
                )
            for warning in reply.get("warnings") or []:
                st.info(warning)

            cited = reply.get("cited") or []
            sources = reply.get("sources") or []
            if sources:
                st.caption(
                    f"Answered from {len(sources)} source(s)"
                    + (f", citing {', '.join(f'[{n}]' for n in cited)}" if cited else "")
                )
                with st.expander("What the answer was allowed to read"):
                    for source in sources:
                        marker = "cited" if source["number"] in cited else "not cited"
                        st.markdown(
                            f"**[{source['number']}]** `{source['kind']}` · "
                            f"{source.get('origin') or 'unknown'} · _{marker}_"
                        )
                        st.text(source["text"][:600])

        if result is not None:
            chunks = result.get("chunks") or []
            triples = result.get("triples") or []
            timings = result.get("timings_ms") or {}
            plan = result.get("plan") or {}

            shown = (reply or {}).get("timings_ms") or timings
            a, b, c = st.columns(3)
            a.metric("Passages", len(chunks))
            b.metric("Graph facts", len(triples))
            c.metric("Took", f"{shown.get('total', 0) / 1000:.2f}s")

            for warning in result.get("warnings") or []:
                st.warning(warning)

            if plan.get("rewritten"):
                with st.expander("How the question was read"):
                    if plan.get("sub_queries"):
                        st.write("**Split into:**")
                        for sub in plan["sub_queries"]:
                            st.write(f"- {sub}")
                    if plan.get("step_back"):
                        st.write(f"**Broader form:** {plan['step_back']}")
                    if plan.get("entities"):
                        st.write(f"**Entities:** {', '.join(plan['entities'])}")
                    if plan.get("filters"):
                        st.write("**Filters inferred from the question:**")
                        st.json(plan["filters"])

            if not chunks and not triples:
                st.info(
                    "Nothing matched. Either nothing is indexed yet — check the "
                    "sidebar — or the filters excluded everything."
                )
            elif chunks:
                st.subheader("Passages the answer drew on")

            for position, chunk in enumerate(chunks, start=1):
                meta = chunk.get("metadata") or {}
                found = ", ".join(chunk.get("found_by") or [])
                source = meta.get("source") or "unknown source"
                with st.container(border=True):
                    head, badge = st.columns([5, 1])
                    head.markdown(f"**{position}. {source}**")
                    badge.markdown(f"`{chunk.get('score', 0):.4f}`")
                    st.write(chunk.get("document", ""))

                    bits = [f"found by {found}"]
                    for field in ("doc_type", "department", "author", "date", "language"):
                        if meta.get(field):
                            bits.append(f"{field}: {meta[field]}")
                    if meta.get("page_no", -1) not in (-1, None):
                        bits.append(f"page {meta['page_no']}")
                    st.caption(" · ".join(bits))

            if triples:
                st.subheader("From the knowledge graph")
                if result.get("graph_seeds"):
                    st.caption("Seeded from: " + ", ".join(result["graph_seeds"]))
                for triple in triples:
                    st.markdown(f"- `{triple.get('text', '')}`")

            with st.expander("Timings and raw response"):
                st.json({"timings_ms": timings, "plan": plan})
                st.json(result)
