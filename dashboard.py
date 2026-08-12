"""Streamlit dashboard.

Four tabs, and the one that matters most is "Cascade": it shows which tier
answered how often. That number is how you tell the pipeline is working, and
the first place to look on the day a site changes its markup and everything
starts falling through to the model again.

    ./venv/bin/streamlit run dashboard.py
"""

from __future__ import annotations

import json
import time

import requests
import streamlit as st

from config import config

API = config.API_BASE_URL.rstrip("/")
POLL_INTERVAL = 1.5
POLL_TIMEOUT = 3600  # a live capture can legitimately run for a long time

TIER_LABELS = {
    "cache": ("Tier 0 · cache", "Content unchanged since the last run."),
    "structured": ("Tier 1 · structured data", "Read the publisher's own JSON-LD or table."),
    "selector": ("Tier 2 · selector spec", "Replayed a spec learned once for this domain."),
    "llm": ("Tier 3 · model", "A model read the text."),
    "native": ("Native", "The handler produced fields directly."),
}

st.set_page_config(page_title="Universal Extractor", page_icon="⚡", layout="wide")
st.title("⚡ Universal Extraction Pipeline")
st.caption(
    "Point it at anything — a page, a PDF, a spreadsheet, a feed, an archive, a "
    "podcast, a live stream. Type detection is by magic bytes; a model is only "
    "reached when the cheaper tiers cannot answer."
)


def api_get(path: str, **params):
    try:
        response = requests.get(f"{API}{path}", params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API} — is uvicorn running? ({exc})")
        return None


# --------------------------------------------------------------------------- #
# Sidebar: is anything actually running?
# --------------------------------------------------------------------------- #

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

        st.caption("Dependencies")
        st.text(f"  redis   {'ok' if health.get('redis') else 'DOWN'}")
        st.text(f"  mongo   {health.get('mongo')}")
        llm = health.get("llm", {})
        st.text(f"  ollama  {'ok' if llm.get('ollama', {}).get('available') else 'unavailable'}")
        st.text(f"  groq    {'ok' if llm.get('groq', {}).get('available') else 'not configured'}")
        st.text(f"  robots  {'enforced' if health.get('robots_enforced') else 'IGNORED'}")

run_tab, crawl_tab, cascade_tab, specs_tab, history_tab = st.tabs(
    ["🚀 Extract", "🕸️ Crawl a site", "📊 Cascade", "🎯 Learned selectors", "🗃️ History"]
)

# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #

with run_tab:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input")
        url = st.text_input("URL", "https://quotes.toscrape.com/")

        if url:
            preview = api_get("/api/v1/detect", url=url)
            if preview:
                if preview.get("allowed"):
                    st.caption(
                        f"Would acquire via **{preview['acquisition']}** — {preview['reason']}"
                    )
                else:
                    st.error(f"Blocked before fetching: {preview.get('policy_reason')}")

        prompt = st.text_area("What to extract", "Extract the main items and their key fields.")
        schema_text = st.text_area(
            "Schema",
            json.dumps({"title": "string", "summary": "string", "tags": "list of strings"}, indent=2),
            height=160,
            help='Field names to types. "string", "number", "list of strings", or a nested object.',
        )

        with st.expander("Options"):
            force_dynamic = st.checkbox("Force browser rendering (Playwright)")
            local_only = st.checkbox(
                "Local only — never send this content to a hosted model", value=False
            )
            follow_children = st.checkbox(
                "Recurse into archives, feeds and attachments", value=True
            )
            tier_choice = st.multiselect(
                "Cascade tiers",
                options=[0, 1, 2, 3],
                default=[0, 1, 2, 3],
                format_func=lambda t: f"{t} · " + ["cache", "structured data", "selectors", "model"][t],
            )

        start = st.button("Start extraction", use_container_width=True, type="primary")

    with right:
        st.subheader("Progress")

        if start:
            try:
                schema = json.loads(schema_text)
            except json.JSONDecodeError as exc:
                st.error(f"Schema is not valid JSON: {exc}")
                st.stop()

            try:
                response = requests.post(
                    f"{API}/api/v1/extract",
                    json={
                        "url": url,
                        "prompt": prompt,
                        "schema_template": schema,
                        "force_dynamic": force_dynamic,
                        "local_only": local_only,
                        "follow_children": follow_children,
                        "allowed_tiers": tier_choice or None,
                    },
                    timeout=20,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()

            if response.status_code >= 400:
                st.error(f"{response.status_code}: {response.text[:400]}")
                st.stop()

            queued = response.json()
            task_id = queued["task_id"]
            st.info(f"Queued on the **{queued['queue']}** queue · `{task_id}`")

            status_box = st.empty()
            progress = st.progress(0.0)
            started = time.time()

            while time.time() - started < POLL_TIMEOUT:
                state = api_get(f"/api/v1/tasks/{task_id}")
                if state is None:
                    break
                status = state.get("status")

                if status in ("PENDING", "RECEIVED"):
                    status_box.info("Waiting for a worker…")
                elif status == "PROGRESS":
                    status_box.info(state.get("stage", "Working…"))
                    progress.progress(min(0.9, (time.time() - started) / 60))
                elif status == "RETRY":
                    status_box.warning("Transient failure — retrying with backoff.")
                elif status == "SUCCESS":
                    progress.progress(1.0)
                    result = state["result"]
                    label, why = TIER_LABELS.get(result.get("method", ""), ("—", ""))
                    status_box.success(f"Done · {label}")
                    st.caption(why)

                    a, b, c = st.columns(3)
                    a.metric("Kind", result.get("kind", "?"))
                    b.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                    c.metric("Total", f"{result.get('timings_ms', {}).get('total', 0) / 1000:.1f}s")

                    if result.get("validation_failures"):
                        st.warning("Validation flagged this record:")
                        for failure in result["validation_failures"]:
                            st.text(f"  • {failure}")
                    for warning in result.get("warnings", []):
                        st.caption(f"⚠ {warning}")

                    st.json(result.get("extracted_data") or {})

                    if result.get("children"):
                        st.subheader(f"{len(result['children'])} nested item(s)")
                        for child in result["children"]:
                            with st.expander(child["url"].split("!/")[-1]):
                                st.json(child.get("extracted_data") or {"error": child.get("error")})

                    with st.expander("Run report"):
                        st.json(result.get("run", {}))
                    break
                elif status == "FAILURE":
                    progress.empty()
                    status_box.error(state.get("error", "Task failed"))
                    break

                time.sleep(POLL_INTERVAL)
            else:
                status_box.warning("Stopped polling. The job may still be running.")

# --------------------------------------------------------------------------- #
# Crawl
# --------------------------------------------------------------------------- #

with crawl_tab:
    st.subheader("Walk a site")
    st.caption(
        "Give it a starting URL and it follows links within the scope you set. "
        "Name a file type and pages become the map rather than the destination: "
        "they are walked for their links, and only matching files are extracted."
    )

    left, right = st.columns([1, 1])

    with left:
        crawl_url = st.text_input("Start URL", "https://books.toscrape.com/", key="crawl_url")
        crawl_prompt = st.text_area(
            "What to extract from each item", "Extract the title and any key figures.",
            key="crawl_prompt",
        )
        crawl_schema_text = st.text_area(
            "Schema",
            json.dumps({"title": "string", "summary": "string"}, indent=2),
            height=120, key="crawl_schema",
        )

        file_types = st.text_input(
            "File types to collect",
            "",
            help="Comma-separated, e.g. pdf or pdf,xlsx. Leave empty to extract every page.",
        )
        col_a, col_b = st.columns(2)
        crawl_depth = col_a.number_input("Max depth", 0, 10, 2)
        crawl_budget = col_b.number_input("Max URLs", 1, 50_000, 200)

        with st.expander("Scope"):
            same_site = st.checkbox("Stay on this site", value=True)
            include_pattern = st.text_input("Only collect URLs matching (regex)", "")
            exclude_pattern = st.text_input("Never visit URLs matching (regex)", "")

        start_crawl_btn = st.button("Start crawl", use_container_width=True, type="primary")

    with right:
        st.subheader("Progress")

        if start_crawl_btn:
            try:
                crawl_schema = json.loads(crawl_schema_text)
            except json.JSONDecodeError as exc:
                st.error(f"Schema is not valid JSON: {exc}")
                st.stop()

            payload = {
                "start_url": crawl_url,
                "prompt": crawl_prompt,
                "schema_template": crawl_schema,
                "collect_extensions": [
                    part.strip().lstrip(".") for part in file_types.split(",") if part.strip()
                ],
                "max_depth": int(crawl_depth),
                "max_pages": int(crawl_budget),
                "same_site": same_site,
                "include_patterns": [include_pattern] if include_pattern else [],
                "exclude_patterns": [exclude_pattern] if exclude_pattern else [],
            }

            try:
                response = requests.post(f"{API}/api/v1/crawl", json=payload, timeout=20)
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                st.stop()
            if response.status_code >= 400:
                st.error(f"{response.status_code}: {response.text[:400]}")
                st.stop()

            started = response.json()
            crawl_id = started["crawl_id"]
            st.info(started["plan"])
            st.caption(f"crawl id `{crawl_id}`")

            status_box = st.empty()
            metrics_box = st.empty()
            targets_box = st.empty()
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

                found = state.get("targets") or []
                if found:
                    with targets_box.container():
                        st.caption(f"{len(found)} file(s) found")
                        st.code("\n".join(found[:40]), language=None)

                if state.get("done"):
                    status_box.success(
                        f"{state.get('status')} in {state.get('elapsed_seconds')}s — "
                        f"{state.get('fetched')} fetched, {state.get('collected')} collected"
                    )
                    break

                status_box.info(
                    f"Crawling… {state.get('in_flight', 0)} in flight "
                    f"(budget {state.get('claimed', 0)}/{state.get('budget', 0)})"
                )
                time.sleep(POLL_INTERVAL)
            else:
                status_box.warning("Stopped polling. The crawl may still be running.")

            if st.button("Stop this crawl"):
                requests.delete(f"{API}/api/v1/crawls/{crawl_id}", timeout=10)
                st.warning("Stop requested — pages in flight will finish.")


# --------------------------------------------------------------------------- #
# Cascade
# --------------------------------------------------------------------------- #

with cascade_tab:
    st.subheader("Which tier answered")
    st.caption(
        "The share of records that never reached a model. This is the number that "
        "says whether the cascade is working."
    )
    days = st.slider("Window (days)", 1, 90, 7)
    stats = api_get("/api/v1/stats", days=days)

    if stats:
        total = stats.get("records", 0)
        avoidance = stats.get("llm_avoidance_rate")
        a, b = st.columns(2)
        a.metric("Records", total)
        b.metric("Model avoided", f"{avoidance:.0%}" if avoidance is not None else "—")

        by_method = stats.get("by_method") or {}
        if by_method:
            st.bar_chart({TIER_LABELS.get(k, (k, ""))[0]: v for k, v in by_method.items()})
        else:
            st.info("No records yet in this window.")

        with st.expander("Caches"):
            st.json({"extraction": stats.get("extraction_cache"), "http": stats.get("http_cache")})
        with st.expander("Per-host rate limiter"):
            st.json(stats.get("rate_limiter") or {"note": "no hosts contacted yet"})
        breakers = stats.get("circuit_breaker") or {}
        if breakers:
            st.subheader("Circuit breaker")
            for host, state in breakers.items():
                if state.get("permanent"):
                    st.error(f"{host} — blocked ({state.get('signal')})")
                else:
                    st.text(f"{host} — {state.get('state')} ({state.get('failures')} failures)")

# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #

with specs_tab:
    st.subheader("Selector specs learned per domain")
    st.caption(
        "Written once by a model on the first visit to a domain, then replayed for "
        "free. A spec whose fill rate collapses is retired and relearned."
    )
    specs = api_get("/api/v1/specs")
    if specs and specs.get("specs"):
        for spec in specs["specs"]:
            status = "retired" if spec.get("retired") else f"{spec.get('avg_fill_rate', 0):.0%} fill"
            header = f"{spec['domain']}{spec.get('path_prefix', '/')} — {status}, {spec.get('uses', 0)} uses"
            with st.expander(header):
                st.caption(f"Learned from {spec.get('learned_from')}")
                st.json(spec.get("rules") or {})
    else:
        st.info(
            "No specs yet. One is authored the first time a page is extracted from a "
            "domain whose markup carries no structured data."
        )

# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

with history_tab:
    st.subheader("Recent extractions")
    domain_filter = st.text_input("Filter by domain", "")
    records = api_get("/api/v1/records", limit=50, domain=domain_filter or None)

    if records and records.get("records"):
        for record in records["records"]:
            provenance = record.get("provenance") or {}
            method = provenance.get("method", "?")
            label = TIER_LABELS.get(method, (method, ""))[0]
            title = (record.get("metadata") or {}).get("title") or record.get("url", "")
            with st.expander(f"{title[:90]} — {label}"):
                st.caption(record.get("url"))
                st.json(record.get("extracted_data") or {})
                with st.expander("Provenance"):
                    st.json(provenance)
    else:
        st.info(
            "Nothing stored yet. Without MONGO_URI configured, results are written to "
            "output/extractions.jsonl instead."
        )
