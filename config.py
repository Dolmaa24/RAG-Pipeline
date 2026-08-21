"""Runtime configuration.

Every value can be overridden by an environment variable or a line in `.env`.
Nothing is hardcoded at a call site — in particular no credential ever is.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PIPELINE_VERSION: str = "3.0.0"

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    #: Sent on every request. An honest, contactable UA is what turns "some bot
    #: is hammering us" into an email rather than a firewall rule. Override
    #: CONTACT_URL with something a site owner can actually reach you at.
    BOT_NAME: str = "UniversalExtractor"
    CONTACT_URL: str = "https://example.invalid/bot"
    #: Used only when SPOOF_BROWSER_UA is on, which is off by default because
    #: pretending to be Chrome while behaving like a crawler is the behaviour
    #: robots.txt exists to make unnecessary.
    BROWSER_USER_AGENT: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    SPOOF_BROWSER_UA: bool = False

    # ------------------------------------------------------------------ #
    # Phase 1 — compliance and safety
    # ------------------------------------------------------------------ #
    RESPECT_ROBOTS: bool = True
    #: What to do when robots.txt itself is unreachable (5xx / network error).
    #: RFC 9309 says deny; "allow" is the common reading and the default here,
    #: because one flaky 502 should not halt a legitimate crawl.
    ROBOTS_ON_UNAVAILABLE: Literal["allow", "deny"] = "allow"
    ROBOTS_CACHE_TTL: float = 3600.0
    ROBOTS_TIMEOUT: float = 10.0

    REQUESTS_PER_SECOND: float = 1.0
    RATE_BURST: int = 3
    MIN_HOST_DELAY: float = 0.0
    MAX_CONCURRENCY_PER_HOST: int = 2
    #: Cap on a Crawl-delay we will actually honour by sleeping. A site asking
    #: for 300s per page is asking you not to crawl it; that becomes an error
    #: rather than a worker that sleeps for five minutes.
    MAX_CRAWL_DELAY: float = 30.0

    BREAKER_ENABLED: bool = True
    BREAKER_FAILURE_THRESHOLD: int = 8
    BREAKER_COOLDOWN: float = 300.0
    BREAKER_RECOVERY_SUCCESSES: int = 2

    DETECT_BLOCKS: bool = True
    MAX_REDIRECTS: int = 10
    #: SSRF guard. A user-supplied URL must not be able to make the worker
    #: fetch 169.254.169.254 or something on the loopback interface.
    ALLOW_PRIVATE_ADDRESSES: bool = False
    ALLOWED_SCHEMES: str = "http,https"
    HOST_ALLOWLIST: str = ""
    HOST_DENYLIST: str = ""

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    PROXY_URL: Optional[str] = None
    #: Allow TLS handshakes with servers that predate RFC 5746 secure
    #: renegotiation. OpenSSL 3 refuses those outright, which is why a site can
    #: load in curl (LibreSSL on macOS is lenient) yet fail here with
    #: "UNSAFE_LEGACY_RENEGOTIATION_DISABLED". Off by default: it re-opens the
    #: renegotiation MITM window (CVE-2009-3555) for whoever is on the path.
    #: Turn it on when a specific site you trust needs it — a fair number of
    #: government and university sites run TLS stacks that old.
    TLS_ALLOW_LEGACY_RENEGOTIATION: bool = False
    USE_BROWSER_FALLBACK: bool = True
    STATIC_TIMEOUT: float = 20.0
    BROWSER_TIMEOUT_MS: int = 30000
    MAX_RETRIES: int = 3
    MAX_CONTENT_BYTES: int = 64 * 1024 * 1024  # 64 MiB
    #: Bodies over this are streamed to a temp file instead of held in memory.
    STREAM_THRESHOLD_BYTES: int = 8 * 1024 * 1024

    # ------------------------------------------------------------------ #
    # Phase 2 — universal input
    # ------------------------------------------------------------------ #
    #: How deep to recurse into archives, emails, and feeds. 0 disables it.
    MAX_RECURSION_DEPTH: int = 2
    MAX_ARCHIVE_MEMBERS: int = 200
    MAX_FEED_ENTRIES: int = 100
    MAX_SITEMAP_URLS: int = 5000
    DOCUMENT_BACKEND: Literal["markitdown", "docling"] = "markitdown"
    OCR_ENABLED: bool = True
    #: Vision.framework needs no install and is fast on Apple Silicon;
    #: tesseract is the portable fallback.
    OCR_BACKEND: Literal["auto", "vision", "tesseract"] = "auto"

    LIVESTREAM_SEGMENT_SECONDS: int = 120
    LIVESTREAM_MAX_SEGMENTS: int = 60
    LIVESTREAM_MAX_MINUTES: int = 120

    # ------------------------------------------------------------------ #
    # Phase 3 — extraction cascade
    # ------------------------------------------------------------------ #
    ENABLE_TIER0_CACHE: bool = True
    ENABLE_TIER1_STRUCTURED: bool = True
    ENABLE_TIER2_SELECTORS: bool = True
    ENABLE_TIER3_LLM: bool = True
    #: A tier's result is accepted only if it filled at least this share of the
    #: requested fields. Below it, the cascade falls through to the next tier.
    MIN_FILL_RATE: float = 0.5
    #: A saved selector spec whose fill rate drops below this is treated as
    #: drifted and regenerated on the next visit.
    SPEC_DRIFT_FILL_RATE: float = 0.4
    SPEC_MAX_AGE_DAYS: int = 30

    LLM_BACKEND: Literal["auto", "ollama", "groq"] = "auto"
    #: Backend for calls a user is waiting on — query understanding, graph
    #: traversal. Those are small prompts at low volume, where a hosted model's
    #: ~500 tok/s against a local 35 tok/s is the whole latency budget. Bulk
    #: ingest stays on LLM_BACKEND, because a hosted free tier's tokens-per-
    #: minute cap makes it *slower* than local for one call per chunk.
    #: None means "same as LLM_BACKEND".
    LLM_INTERACTIVE_BACKEND: Optional[Literal["auto", "ollama", "groq"]] = None
    OLLAMA_HOST: str = "http://localhost:11434"
    #: qwen2.5:3b over llama3.2:3b on measurement, at the same ~2 GB and the
    #: same speed. Scored on this pipeline's own jobs (bench/graph_models.py),
    #: qwen against llama: relationship directions 3 right / 0 reversed against
    #: 2 right / 2 reversed, alias judgement 4/4 against 3/4, traversal Cypher
    #: Kuzu accepts 2/5 against 1/5. It finds one fewer entity (10/11 against
    #: 11/11), which is the right trade: a reversed edge is a confidently wrong
    #: fact, a missing entity is only an absent one.
    AI_MODEL_NAME: str = "qwen2.5:3b"
    AI_TIMEOUT: float = 300.0
    #: How long Ollama keeps the model resident after a call. Its own default is
    #: 5 minutes, so an intermittent pipeline pays a measured ~2.2s reload on
    #: every call after a gap.
    OLLAMA_KEEP_ALIVE: str = "30m"
    #: Context window, which the prompt and the generation share. Ollama's
    #: default is 4096 and this pipeline can exceed it: a full MAX_CHUNK_SIZE
    #: chunk measured 2919 prompt tokens, so with NUM_PREDICT at 4096 a long
    #: answer pushes the total past the window. Ollama then *shifts* the
    #: context, dropping the front of the prompt — which is where the
    #: instructions are, so the symptom is bad extraction rather than an error.
    #: The cost is KV cache: llama3.2:3b measured 2.0 GB resident at 4096 and
    #: 3.0 GB at 8192. Lower both together if that matters more than headroom.
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 4096
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TIMEOUT: float = 60.0
    #: Never send content to a hosted model. Forces the Ollama backend even
    #: when GROQ_BACKEND would be faster.
    LOCAL_ONLY: bool = False
    MAX_CHUNK_SIZE: int = 12000  # characters of text sent to the model
    LLM_MAX_ATTEMPTS: int = 2

    # ------------------------------------------------------------------ #
    # Phase 4 — performance
    # ------------------------------------------------------------------ #
    HTTP_CACHE_ENABLED: bool = True
    HTTP_CACHE_TTL_SECONDS: int = 86400
    EXTRACTION_CACHE_ENABLED: bool = True
    EXTRACTION_CACHE_TTL_DAYS: int = 30
    RAW_STORE_ENABLED: bool = True
    RAW_STORE_TTL_DAYS: int = 14

    WHISPER_BACKEND: Literal["auto", "mlx", "faster"] = "auto"
    WHISPER_MODEL_SIZE: str = "base"
    MLX_WHISPER_MODEL: str = "mlx-community/whisper-base-mlx"
    WHISPER_LANGUAGE: Optional[str] = None
    #: Load Whisper once per worker process instead of once per task.
    PRELOAD_MODELS: bool = True

    IO_QUEUE: str = "io"
    CPU_QUEUE: str = "cpu"

    # ------------------------------------------------------------------ #
    # Phase 5 — trust
    # ------------------------------------------------------------------ #
    VALIDATE_OUTPUT: bool = True
    #: Reject rather than store a record that fails validation. Off by default:
    #: a flagged row you can inspect beats a row that silently vanished.
    REJECT_ON_VALIDATION_FAILURE: bool = False
    DEDUPE_ENABLED: bool = True
    #: Hamming distance between 64-bit simhashes below which two documents are
    #: "the same page with a different ad". 3 is the widely used threshold.
    SIMHASH_MAX_DISTANCE: int = 3
    DRIFT_ENABLED: bool = True
    DRIFT_MIN_SAMPLES: int = 20
    DRIFT_FILL_RATE_DROP: float = 0.3

    # ------------------------------------------------------------------ #
    # Phase 6 — indexing for retrieval
    # ------------------------------------------------------------------ #
    #: On by default: retrieval is the point of indexing, and a feature that is
    #: off by default is a feature nobody has. The cost is a ~130 MB embedding
    #: model resident per cpu worker and a local vector table. Turn it off for a
    #: job that only wants structured JSON out.
    INDEX_ENABLED: bool = True
    #: "agentic" reads the document's shape and picks one of the others.
    INDEX_CHUNK_STRATEGY: Literal[
        "agentic", "fixed", "semantic", "hierarchical", "llm"
    ] = "agentic"
    #: Whether the agentic router may choose the model-assisted splitter.
    #: Off, and deliberately so: indexing is on by default, and a default-on
    #: feature must not silently start making a model call per document. It also
    #: misfires — a page of quotes and tag lists has short lines and reads as OCR
    #: damage, which cost 39s and 78 chunks for one HTML page. "llm" stays
    #: available by asking for it, which is where a scanned document wants it.
    INDEX_AGENTIC_ALLOW_LLM: bool = False
    #: Characters, not tokens. Distinct from MAX_CHUNK_SIZE, which is how much
    #: text the *extraction* stage sends to a model — a different question.
    INDEX_CHUNK_SIZE: int = 1000
    INDEX_CHUNK_OVERLAP: int = 200
    #: Ceiling on model calls for one document under the "llm" strategy. Without
    #: it a long PDF becomes an unbounded number of calls.
    INDEX_LLM_MAX_WINDOWS: int = 8
    INDEX_DENSE_PROVIDER: Literal["local_bge", "local_e5", "cohere", "voyage"] = "local_bge"
    #: "auto" means CPU. Metal has been measured killing both the prefork
    #: worker and the uvicorn process; BGE-small on CPU does 32 chunks in about
    #: a quarter of a second, so the GPU is not worth a crash. Set "mps"
    #: explicitly for a context you have verified — see pipeline/embed/dense.py.
    INDEX_EMBED_DEVICE: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    #: Presidio loads a spaCy NER model, so this costs hundreds of MB the first
    #: time it runs. Off by default, and the packages are not in requirements.
    INDEX_PII_REMOVAL: bool = False
    #: Cap on the text handed to one indexing task. Past this the tail is
    #: dropped and a warning recorded rather than the broker carrying a payload
    #: measured in megabytes.
    INDEX_MAX_TEXT_CHARS: int = 400_000

    LANCE_DB_DIR: str = "./lance_data"
    #: One table per embedding model. A mismatched write is refused.
    LANCE_TABLE_NAME: str = "chunks"
    #: Below this many rows an ANN index is not built. IVF_PQ has to train on
    #: the data, and under a few thousand vectors a flat scan is both faster
    #: and exact — an index here would cost accuracy and buy nothing.
    INDEX_ANN_MIN_ROWS: int = 5000
    #: IVF partitions and PQ sub-vectors. num_sub_vectors must divide the
    #: embedding dimension: 384 / 48 = 8 bytes per vector, a 192x reduction on
    #: the raw float32.
    INDEX_IVF_PARTITIONS: int = 256
    INDEX_PQ_SUB_VECTORS: int = 48

    # ------------------------------------------------------------------ #
    # Phase 7 — retrieval
    # ------------------------------------------------------------------ #
    #: Rewriting costs a model call. Queries shorter than this with no
    #: conjunction, comparative or date expression skip it entirely.
    RETRIEVE_REWRITE_ENABLED: bool = True
    RETRIEVE_SIMPLE_QUERY_WORDS: int = 8
    RETRIEVE_MAX_SUBQUERIES: int = 4
    RETRIEVE_PLAN_CACHE_SIZE: int = 256
    RETRIEVE_TOP_K: int = 10
    #: How many candidates each leg fetches before fusion. Wider than top_k,
    #: because fusion can only reorder what it was given.
    RETRIEVE_CANDIDATES: int = 50
    #: "rrf" ignores scores and uses rank, which is robust when the two legs
    #: score on different scales. "alpha" is a weighted sum of normalised
    #: scores, tunable when you know your corpus.
    RETRIEVE_FUSION: Literal["rrf", "alpha"] = "rrf"
    #: Weight on the dense leg under "alpha"; 1.0 is pure vector, 0.0 pure BM25.
    RETRIEVE_ALPHA: float = 0.7
    RETRIEVE_RRF_K: int = 60
    #: A cross-encoder is another model resident on an 8 GB host, so this is
    #: off unless asked for.
    RETRIEVE_RERANK: bool = False
    RETRIEVE_RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RETRIEVE_RERANK_CANDIDATES: int = 20
    #: IVF probes and exact-rescoring factor. Raising either trades latency for
    #: recall; refine_factor recovers most of what PQ quantisation costs.
    RETRIEVE_NPROBES: int = 20
    RETRIEVE_REFINE_FACTOR: int = 10

    # ------------------------------------------------------------------ #
    # Phase 8 — answering
    # ------------------------------------------------------------------ #
    #: How many retrieved passages are put in front of the model. More context
    #: is not more accuracy: past a handful the answer starts drifting toward
    #: whatever is longest rather than whatever is relevant.
    ANSWER_MAX_PASSAGES: int = 6
    ANSWER_MAX_TRIPLES: int = 20
    #: Characters of any single passage included. A whole page in one slot
    #: crowds out the other five.
    ANSWER_MAX_PASSAGE_CHARS: int = 1200

    # ------------------------------------------------------------------ #
    # Phase 7 — knowledge graph
    # ------------------------------------------------------------------ #
    #: On by default: the graph is half of what "hybrid + graph retrieval" means,
    #: and a graph nobody builds answers no questions. It costs one model call
    #: per MAX_CHUNK_SIZE window of a document, so it is the slowest part of
    #: ingest — turn it off for a job that only wants passages back.
    GRAPH_ENABLED: bool = True
    #: Cache graph extraction on the chunk's content hash, so re-ingesting an
    #: unchanged document costs a lookup instead of a model call per chunk.
    GRAPH_CACHE_ENABLED: bool = True
    #: Check every extracted relationship's direction against entity types and
    #: the word order of the sentence, and flip the ones that are backwards.
    #: A reversed edge is a confident falsehood nothing downstream can detect.
    GRAPH_VALIDATE_DIRECTION: bool = True
    #: How entities are found. "gliner" scores spans with a small encoder —
    #: measured at 166ms against the local 3B's ~10s on the same paragraph, and
    #: it finds entities the model misses. The model is still asked for the
    #: relationships, which GLiNER does not do. "llm" is the original path.
    GRAPH_ENTITY_BACKEND: Literal["llm", "gliner"] = "gliner"
    GRAPH_NER_MODEL: str = "urchade/gliner_small-v2.1"
    #: Zero-shot, so these are just words. Changing them changes what is found.
    GRAPH_NER_LABELS: str = "person,organization,location,product,project,technology,event"
    GRAPH_NER_THRESHOLD: float = 0.5
    KUZU_DB_PATH: str = "./kuzu_db"
    #: Ceiling on model calls for one document's graph. Long text goes through
    #: in MAX_CHUNK_SIZE windows rather than being truncated at the first one,
    #: which used to mean a long PDF produced a graph of its first few pages
    #: and said nothing about the rest.
    GRAPH_MAX_WINDOWS: int = 8
    GRAPH_MAX_HOPS: int = 2
    #: Cap on rows returned by a traversal. A well-connected node in a
    #: two-hop query fans out combinatorially.
    GRAPH_MAX_TRIPLES: int = 50
    GRAPH_SEED_ENTITIES: int = 3
    #: Cosine similarity above which two entity names are candidates for
    #: merging. The LLM verification pass is what actually decides.
    GRAPH_RESOLUTION_THRESHOLD: float = 0.70
    #: Let a model write the traversal query instead of using the fixed
    #: template. Off by default, on measurement with both local models: neither
    #: writes Cypher Kuzu reliably accepts (qwen2.5:3b 2/5, llama3.2:3b 1/5),
    #: so the template answers anyway — measured at 39ms against the agent's
    #: 1.9s for the identical six triples. Worth turning on with a strong hosted
    #: model, where it can express traversals the template cannot. It runs
    #: read-only either way, so the cost of it being wrong is latency.
    GRAPH_CYPHER_AGENT: bool = False

    # ------------------------------------------------------------------ #
    # Infrastructure
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://localhost:6379/0"
    MONGO_URI: Optional[str] = None
    MONGO_DB_NAME: str = "ai_scraping_pipeline"
    MONGO_COLLECTION: str = "extractions"
    MONGO_SPECS_COLLECTION: str = "selector_specs"
    MONGO_CACHE_COLLECTION: str = "extraction_cache"
    MONGO_HTTP_CACHE_COLLECTION: str = "http_cache"
    MONGO_RUNS_COLLECTION: str = "runs"
    MONGO_DEADLETTER_COLLECTION: str = "dead_letter"
    MONGO_FINGERPRINTS_COLLECTION: str = "fingerprints"
    GRIDFS_BUCKET: str = "raw_store"
    API_BASE_URL: str = "http://127.0.0.1:8000"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"

    # ------------------------------------------------------------------ #
    # Derived
    # ------------------------------------------------------------------ #

    @field_validator("MIN_FILL_RATE", "SPEC_DRIFT_FILL_RATE")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v

    @property
    def user_agent(self) -> str:
        if self.SPOOF_BROWSER_UA:
            return self.BROWSER_USER_AGENT
        return f"{self.BOT_NAME}/{self.PIPELINE_VERSION} (+{self.CONTACT_URL})"

    @property
    def robots_agent(self) -> str:
        """The token robots.txt rules are matched against."""
        return self.BOT_NAME if not self.SPOOF_BROWSER_UA else "*"

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.OLLAMA_HOST.rstrip('/')}/api/generate"

    @property
    def ollama_tags_url(self) -> str:
        return f"{self.OLLAMA_HOST.rstrip('/')}/api/tags"

    @property
    def allowed_schemes(self) -> frozenset[str]:
        return frozenset(s.strip().lower() for s in self.ALLOWED_SCHEMES.split(",") if s.strip())

    @property
    def host_allowlist(self) -> frozenset[str]:
        return frozenset(h.strip().lower() for h in self.HOST_ALLOWLIST.split(",") if h.strip())

    @property
    def host_denylist(self) -> frozenset[str]:
        return frozenset(h.strip().lower() for h in self.HOST_DENYLIST.split(",") if h.strip())

    @property
    def graph_ner_labels(self) -> tuple[str, ...]:
        return tuple(
            label.strip().lower()
            for label in self.GRAPH_NER_LABELS.split(",")
            if label.strip()
        )

    @property
    def groq_available(self) -> bool:
        return bool(self.GROQ_API_KEY) and not self.LOCAL_ONLY

    def prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def redacted(self) -> dict:
        """Settings dump safe to log or return from /health."""
        data = self.model_dump()
        for key in ("GROQ_API_KEY", "MONGO_URI", "PROXY_URL"):
            if data.get(key):
                data[key] = "***set***"
        return data


config = EngineConfig()
