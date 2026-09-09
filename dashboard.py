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


def api_detail(exc: requests.HTTPError) -> str:
    """The reason the API gave, rather than the generic status line."""
    try:
        detail = exc.response.json().get("detail", "")
    except Exception:
        detail = exc.response.text[:200]
    return f"{exc.response.status_code}: {detail}"


def api_post(path: str, payload: dict, timeout: int = 120):
    """POST and return the body, or None with the reason already surfaced."""
    try:
        response = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        st.error(api_detail(exc))
        return None
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API} ({exc})")
        return None

def api_delete(path: str, timeout: int = 30):
    try:
        response = requests.delete(f"{API}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        st.error(api_detail(exc))
        return None
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API} ({exc})")
        return None


def poll_build(task_id: str, status_box) -> dict | None:
    """Watch a build. Longer-running than an investigation and noisier, so the
    progress line names the worker rather than only the stage."""
    began = time.time()
    while time.time() - began < POLL_TIMEOUT:
        state = api_get(f"/api/v1/builds/{task_id}")
        if state is None:
            return None
        if state.get("done"):
            if state.get("error"):
                st.error(state["error"])
            return state.get("result")

        step = state.get("progress") or {}
        stage = step.get("stage", state.get("status", "queued"))
        worker = step.get("worker", "")
        position = (
            f" ({step['index']} of {step['of']})"
            if step.get("index") and step.get("of")
            else ""
        )
        status_box.info(
            f"{stage}"
            + (f" · {worker}{position}" if worker else "")
            + f" · {int(time.time() - began)}s"
        )
        time.sleep(POLL_INTERVAL)
    return None


def render_build(payload: dict) -> None:
    """One finished build: what each worker wrote, and whether it runs."""
    files = payload.get("files") or []
    if not files:
        st.error(payload.get("stopped") or "Nothing was written.")
    else:
        st.success(f"{len(files)} file(s) in {payload.get('project', 'the workspace')}")

    tests = payload.get("tests")
    if tests is None:
        st.info(
            "The generated code was not run. Tick *Run the generated tests* to "
            "execute it — that is a separate permission from writing it."
        )
    elif tests.get("timed_out"):
        st.warning("The tests were killed for running too long.")
    elif tests.get("ok"):
        st.success(f"Tests passed in {tests.get('seconds', 0)}s.")
    else:
        st.warning(f"Tests failed (exit {tests.get('exit_code')}).")

    st.markdown("#### What each worker wrote")
    for step in payload.get("steps") or []:
        wrote = ", ".join(step.get("files") or []) or "nothing"
        line = f"**{step['name']}** — {wrote}"
        if step.get("note"):
            line += f"  \n<span style='color:#c60'>{step['note']}</span>"
        if step.get("purpose"):
            line += f"  \n<span style='color:#888'>{step['purpose']}</span>"
        st.markdown(line, unsafe_allow_html=True)

    if tests and tests.get("output"):
        st.markdown("#### Test output")
        st.code(tests["output"][:4000], language="text")

    if files:
        st.markdown("#### Files")
        for name in files:
            st.markdown(f"`{name}`")

    for warning in payload.get("warnings") or []:
        st.caption(f"warning: {warning}")


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

    # The graph is derived data with no link back to its documents, so removing
    # a document cannot remove what it contributed — an entity two documents
    # mention is one node. All of it is the only honest granularity, and
    # without this the graph goes on answering after its sources are gone.
    graph_rows = (graph or {}).get("counts", {}) if graph else {}
    if graph and graph.get("available") and graph_rows.get("entities"):
        with st.expander("Reset the knowledge graph"):
            st.caption(
                "Removes every entity and relationship, and the entity index "
                "that seeds traversal. Rebuild by extracting the documents "
                "again with 'Knowledge graph' on."
            )
            if st.button("Clear the graph", key="clear_graph_go"):
                try:
                    response = requests.delete(f"{API}/api/v1/graph", timeout=60)
                    response.raise_for_status()
                    gone = response.json()
                except requests.HTTPError as exc:
                    st.error(api_detail(exc))
                except requests.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")
                else:
                    st.success(
                        f"Cleared {gone.get('entities', 0)} entities and "
                        f"{gone.get('relationships', 0)} relationships."
                    )
                    st.rerun()

    # Deleting the uploaded file removes the file, not the text: the upload is
    # only needed while it is being extracted, and the corpus keeps its own
    # copy. Without this, a document goes on answering questions after the user
    # believes they have removed it.
    if stats and stats.get("chunks"):
        with st.expander("Remove a document"):
            indexed = stats.get("filters", {}).get("source", [])
            if not indexed:
                st.caption("Nothing to remove.")
            else:
                chosen = st.selectbox(
                    "Source", indexed, key="forget_source",
                    format_func=lambda s: s if len(s) < 46 else "…" + s[-45:],
                )
                st.caption(
                    "Removes its passages from search. Knowledge-graph entities "
                    "stay: they are merged across documents, so they cannot be "
                    "attributed to one."
                )
                if st.button("Remove from corpus", key="forget_go"):
                    try:
                        response = requests.delete(
                            f"{API}/api/v1/index/source",
                            params={"source": chosen}, timeout=30,
                        )
                        response.raise_for_status()
                        gone = response.json().get("chunks_removed", 0)
                    except requests.HTTPError as exc:
                        st.error(api_detail(exc))
                    except requests.RequestException as exc:
                        st.error(f"Could not reach the API: {exc}")
                    else:
                        st.success(f"Removed {gone} passage(s).")
                        st.rerun()


def poll_investigation(task_id: str, status_box) -> dict | None:
    """Watch one queued investigation until it finishes or the wait runs out.

    Shared by the Investigate and Task tabs. Both queue the same task and poll
    the same endpoint, and a progress display that drifts between two tabs is a
    display nobody trusts.
    """
    began = time.time()
    while time.time() - began < POLL_TIMEOUT:
        state = api_get(f"/api/v1/investigations/{task_id}")
        if state is None:
            return None
        if state.get("done"):
            if state.get("error"):
                st.error(state["error"])
            return state.get("result")

        step = state.get("progress") or {}
        stage = step.get("stage", state.get("status", "queued"))
        detail = step.get("role") or step.get("tools") or ""
        status_box.info(
            f"{stage}"
            + (f" · round {step['round']}" if step.get("round") else "")
            + (f" · {detail}" if detail else "")
            + f" · {int(time.time() - began)}s"
        )
        time.sleep(POLL_INTERVAL)
    return None


def render_investigation(payload: dict) -> None:
    """One finished investigation: the answer, what was checked, and the trace."""

    if payload.get("sufficient"):
        st.success("Answered from the corpus.")
    else:
        st.warning(
            "The corpus did not fully cover this. The answer below is "
            "what could be supported; treat the rest as unverified."
        )

    st.markdown(payload.get("answer") or "_No answer was produced._")

    check = payload.get("verification") or {}
    if check.get("skipped"):
        st.caption(f"Not checked: {check.get('reason', 'unknown')}")
    elif check:
        flagged = len(check.get("unsupported") or [])
        if flagged:
            st.caption(
                f"{flagged} of {check['checked']} claims are marked "
                "[unsupported] — the cited passages did not clearly "
                "carry them."
            )
            # The rate is stated with the flags, not buried in docs. A
            # reader who finds one correct sentence in two marked stops
            # reading the marks, which costs the real catches their
            # value.
            if check.get("caveat"):
                st.caption(f":grey[{check['caveat']}]")
        else:
            st.caption(
                f"All {check['checked']} sentences are supported by the "
                "passages cited."
            )

    a, b, c, d = st.columns(4)
    a.metric("Rounds", payload.get("rounds", 0))
    b.metric("Sources", len(payload.get("sources") or []))
    c.metric("Seconds", payload.get("seconds", 0))
    d.metric("Ended", payload.get("stopped", ""))

    # Shown, not hidden behind an expander. For a system whose selling
    # point is the reasoning, the reasoning is the interesting part of
    # the screen.
    st.markdown("#### What it did")
    for step in payload.get("trace") or []:
        if step.get("kind") == "specialist":
            tools = ", ".join(step.get("tools") or []) or "no tools"
            st.markdown(
                f"**{step['role']}** — {tools}  \n"
                f"<span style='color:#888'>ended: {step.get('stopped','')}</span>",
                unsafe_allow_html=True,
            )
            for inner in step.get("steps") or []:
                if inner.get("kind") == "tool":
                    ok = "ok" if inner.get("ok") else "failed"
                    st.code(
                        f"{inner['tool']}({json.dumps(inner.get('arguments', {}))})"
                        f"  -> {ok}, {inner.get('duration_ms', 0):.0f}ms\n"
                        f"{(inner.get('observation') or '')[:400]}",
                        language="text",
                    )
        else:
            st.markdown(
                f"**synthesis** — {step.get('sources', 0)} sources, "
                f"sufficient: {step.get('sufficient')}, "
                f"{step.get('seconds', 0)}s"
            )

    if payload.get("sources"):
        st.markdown("#### Sources")
        for source in payload["sources"]:
            st.markdown(
                f"**[{source.get('number')}]** `{source.get('origin', '')}`"
            )
            st.caption((source.get("text") or "")[:300])

    if payload.get("warnings"):
        for warning in payload["warnings"]:
            st.caption(f"warning: {warning}")


# Main Interface
extract_tab, search_tab, investigate_tab, task_tab, build_tab = st.tabs(
    ["Extract", "Search", "Investigate", "Task", "Build"]
)

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
        do_children = st.checkbox(
            "Recurse into contained documents", value=True,
            help="Archive members, feed entries and attachments — the parts a "
                 "container arrives with. Unrelated to crawl depth, which walks "
                 "links between pages.",
        )

        st.caption("Retrieval")
        do_index = st.checkbox(
            "Index for search", value=True,
            help="Chunk, embed and store the text so the Search tab can find it.",
        )
        do_graph = st.checkbox(
            "Build knowledge graph", value=config.GRAPH_ENABLED,
            help="Extract entities and relationships. One model call per chunk, "
                 "so this is much slower than indexing alone. Defaults to "
                 f"GRAPH_ENABLED, currently {config.GRAPH_ENABLED}.",
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
                    response = requests.post(
                        f"{API}/api/v1/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=30,
                    )
                    response.raise_for_status()
                    target_url = response.json().get("url")
                except requests.HTTPError as exc:
                    st.error(f"Failed to upload file: {api_detail(exc)}")
                    st.stop()
                except requests.RequestException as exc:
                    st.error(f"Failed to upload file: {exc}")
                    st.stop()
        elif url:
            target_url = url
        else:
            st.warning("Please upload a file or enter a URL.")
            st.stop()

        # A crawl walks links between pages. An uploaded file has no site to
        # walk, so the depth setting is simply not applicable to it.
        if crawl_depth > 0 and uploaded_file is not None:
            st.caption("Crawl depth does not apply to an uploaded file — extracting it on its own.")
            crawl_depth = 0

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
                        "follow_children": bool(do_children),
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

    # Said before the search runs, not after. A question asking for a set is
    # one this tab answers partially — measured at 1/3 against the loop's 3/3 —
    # and finding that out from a thin answer teaches nothing.
    if query.strip():
        advice = api_get("/api/v1/route", question=query)
        if advice and advice.get("path") == "investigate":
            st.info(
                "This question asks for a set of things. Search answers those "
                "partially — the **Investigate** tab lists them properly, and "
                "takes about half a minute."
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


with investigate_tab:
    st.subheader("Investigate")
    st.caption(
        "Search once, judge whether that was enough, and go back for what was "
        "missing. Slower than Search — half a minute or more — and the reason "
        "to use it is a question with two parts."
    )

    question = st.text_input(
        "Question", "",
        placeholder="What did Acme acquire, and what was the quarterly revenue?",
        key="investigate_question", label_visibility="collapsed",
    )

    with st.expander("What it may do", expanded=False):
        rounds = st.slider(
            "Rounds", 1, 5, 2,
            help="One round is a specialist plus a synthesis call. The second "
                 "is where a two-part answer comes from; a third rarely adds "
                 "evidence the second did not.",
        )
        do_verify = st.checkbox(
            "Check the answer against its sources", value=True,
            help="Catches about 82% of unsupported claims, and wrongly flags "
                 "about 29% of supported ones. It marks sentences rather than "
                 "removing them.",
        )
        st.caption("Reaching outside the corpus")
        allow_network = st.checkbox(
            "May fetch a URL in the question", value=False,
            help="Only used after the corpus has been tried and come up short.",
        )
        allow_write = st.checkbox(
            "May add what it fetched to the corpus", value=False,
            help="Fetching without indexing changes nothing, so acquisition "
                 "needs both.",
        )

    if question.strip():
        advice = api_get("/api/v1/route", question=question)
        if advice and advice.get("path") == "answer":
            st.caption(
                "Measured, the loop answers this kind of question no better "
                "than **Search** does, and takes four times as long. It is "
                "worth it for questions asking which or how many."
            )

    if st.button("Investigate", type="primary", use_container_width=True, key="go_investigate"):
        if not question.strip():
            st.warning("Ask a question first.")
            st.stop()

        queued = api_post(
            "/api/v1/investigate",
            {
                "question": question,
                "allow_network": bool(allow_network),
                "allow_write": bool(allow_write),
                "max_rounds": int(rounds),
                "verify": bool(do_verify),
            },
            timeout=20,
        )
        if queued is None:
            st.stop()

        status_box = st.empty()
        payload = poll_investigation(queued["task_id"], status_box)

        if payload:
            status_box.empty()
            render_investigation(payload)


with task_tab:
    st.subheader("Task")
    st.caption(
        "Say what you want done. The domain is worked out from the words — "
        "no model call — and the matching skill supplies the specialist's "
        "prompt and its tools."
    )

    intent = st.text_input(
        "Intent", "",
        placeholder="Which policies cover physiotherapy?",
        key="task_intent", label_visibility="collapsed",
    )

    catalogue = api_get("/api/v1/skills") or {}
    names = [skill["name"] for skill in catalogue.get("skills", [])]

    if not names:
        st.info(
            "No skills are loaded. Each one is a folder under `skills/` "
            "holding a `SKILL.md`."
        )

    chosen = st.selectbox(
        "Skill",
        ["(work it out from the intent)", *names],
        help="Override the routing when the match is wrong.",
    )
    override = None if chosen.startswith("(") else chosen

    # The routing shown before the run, not after. A decision the operator
    # cannot see is one they cannot correct, and correcting it is what the
    # selectbox above is for.
    if intent.strip() and override is None:
        preview = api_get("/api/v1/skills/match", intent=intent)
        if preview:
            if preview.get("skill"):
                st.info(
                    f"**{preview['skill']}** — {preview['why']} "
                    f"(confidence {preview['confidence']})"
                )
            else:
                st.caption(
                    "No skill matched, so the generic corpus specialist will "
                    "run — which is what every question got before skills "
                    "existed. Naming one above is how you force a domain."
                )
                # Offered here because here is where the gap shows up. The
                # draft is written to a folder the loader ignores; approving
                # it in the Build tab is what makes it real.
                if st.button("Draft a skill for this", key="draft_skill"):
                    drafted = api_post(
                        "/api/v1/skills/draft", {"intent": intent}, timeout=180
                    )
                    if drafted:
                        st.success(
                            f"Drafted **{drafted['name']}**. Nothing is live yet — "
                            "read it in the Build tab and approve it there."
                        )
                        st.code(drafted["text"], language="markdown")
            runners = preview.get("runners_up") or []
            if runners:
                st.caption(
                    "also considered: "
                    + ", ".join(f"{r['skill']} ({r['score']})" for r in runners)
                )

    with st.expander("What it may do", expanded=False):
        task_rounds = st.slider("Rounds", 1, 5, 2, key="task_rounds")
        task_verify = st.checkbox(
            "Check the answer against its sources", value=True, key="task_verify"
        )
        task_network = st.checkbox(
            "May fetch a URL in the intent", value=False, key="task_network"
        )
        task_write = st.checkbox(
            "May add what it fetched to the corpus", value=False, key="task_write",
            help="A skill cannot grant this. It is granted here or not at all.",
        )

    if st.button("Run", type="primary", use_container_width=True, key="go_task"):
        if not intent.strip():
            st.warning("Say what you want done first.")
            st.stop()

        queued = api_post(
            "/api/v1/task",
            {
                "intent": intent,
                "skill": override,
                "allow_network": bool(task_network),
                "allow_write": bool(task_write),
                "max_rounds": int(task_rounds),
                "verify": bool(task_verify),
            },
            timeout=20,
        )
        if queued is None:
            st.stop()

        agent = queued.get("agent") or {}
        st.markdown(
            f"**Agent:** `{agent.get('role', 'corpus')}` — "
            + (", ".join(agent.get("tools") or []) or "the default tools")
        )

        status_box = st.empty()
        payload = poll_investigation(queued["task_id"], status_box)

        if payload:
            status_box.empty()
            render_investigation(payload)


with build_tab:
    st.subheader("Build")
    st.caption(
        "A skill that declares a roster can be built: each worker writes its "
        "part of a project into a sandbox. What comes out is a scaffold to "
        "read, not a finished application."
    )

    drafts = (api_get("/api/v1/skills/drafts") or {}).get("drafts") or []
    if drafts:
        st.markdown("#### Drafts waiting to be read")
        st.caption(
            "The body of a skill file becomes an agent's system prompt. Read it "
            "before approving — that review is the only thing between a model's "
            "proposal and an agent's instructions."
        )
        which = st.selectbox("Draft", drafts, key="which_draft")
        current = api_get(f"/api/v1/skills/drafts/{which}") or {}
        edited = st.text_area(
            "The file", current.get("text", ""), height=340, key="draft_text"
        )
        approve, discard = st.columns(2)
        if approve.button("Approve and install", type="primary", use_container_width=True):
            if api_post(f"/api/v1/skills/drafts/{which}/approve", {"text": edited}):
                st.success(f"Installed {which}.")
                st.rerun()
        if discard.button("Discard", use_container_width=True):
            if api_delete(f"/api/v1/skills/drafts/{which}"):
                st.rerun()
        st.divider()

    catalogue = (api_get("/api/v1/skills") or {}).get("skills") or []
    buildable = [skill for skill in catalogue if skill.get("buildable")]

    if not buildable:
        st.info(
            "No installed skill declares a roster. A skill becomes buildable "
            "when its SKILL.md has an `agents:` block — draft one from the Task "
            "tab, or add the block to a skill by hand."
        )
    else:
        chosen = st.selectbox(
            "Skill", [skill["name"] for skill in buildable], key="build_skill"
        )
        skill = next(s for s in buildable if s["name"] == chosen)

        st.markdown("**The roster**")
        for agent in skill.get("agents") or []:
            owns = ", ".join(agent.get("writes") or []) or "one module"
            st.markdown(
                f"- **{agent['name']}** — {agent['purpose']}  \n"
                f"  <span style='color:#888'>writes {owns}</span>",
                unsafe_allow_html=True,
            )

        what = st.text_input(
            "What to build",
            skill.get("description", ""),
            key="build_intent",
            help="Passed to every worker along with the shared contract.",
        )
        run_tests = st.checkbox(
            "Run the generated tests",
            value=False,
            key="build_execute",
            help=(
                "Executes code a model wrote, in a subprocess with no secrets "
                "in its environment and a hard timeout. That is mitigation, not "
                "isolation — leave it off to get the files and read them first."
            ),
        )
        if run_tests:
            st.warning(
                "This runs model-written code on this machine. The sandbox "
                "drops the environment and bounds CPU, memory and wall clock, "
                "but it is a subprocess, not a container."
            )

        if st.button("Build", type="primary", use_container_width=True, key="go_build"):
            queued = api_post(
                "/api/v1/build",
                {"skill": chosen, "intent": what, "allow_execute": bool(run_tests)},
                timeout=30,
            )
            if queued is None:
                st.stop()

            status_box = st.empty()
            payload = poll_build(queued["task_id"], status_box)
            if payload:
                status_box.empty()
                render_build(payload)
