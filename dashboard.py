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

# Main Interface
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
