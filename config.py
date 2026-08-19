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
    OLLAMA_HOST: str = "http://localhost:11434"
    AI_MODEL_NAME: str = "llama3.2:3b"
    AI_TIMEOUT: float = 300.0
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
    #: Off by default. Indexing loads a sentence-transformer and writes a vector
    #: store, which is real memory and real disk that a caller who only wants
    #: structured JSON should not be paying for.
    INDEX_ENABLED: bool = False
    #: "agentic" reads the document's shape and picks one of the others.
    INDEX_CHUNK_STRATEGY: Literal[
        "agentic", "fixed", "semantic", "hierarchical", "llm"
    ] = "agentic"
    #: Characters, not tokens. Distinct from MAX_CHUNK_SIZE, which is how much
    #: text the *extraction* stage sends to a model — a different question.
    INDEX_CHUNK_SIZE: int = 1000
    INDEX_CHUNK_OVERLAP: int = 200
    #: Ceiling on model calls for one document under the "llm" strategy. Without
    #: it a long PDF becomes an unbounded number of calls.
    INDEX_LLM_MAX_WINDOWS: int = 8
    INDEX_DENSE_PROVIDER: Literal["local_bge", "local_e5", "cohere", "voyage"] = "local_bge"
    #: "auto" uses CPU inside a prefork worker and lets torch choose elsewhere.
    #: Metal does not survive fork() — see pipeline/embed/dense.py.
    INDEX_EMBED_DEVICE: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    #: "tf" stores per-document term counts. Real BM25 needs corpus-level IDF,
    #: which nothing computes yet — see pipeline/embed/sparse.py.
    INDEX_SPARSE_PROVIDER: Literal["tf", "splade", "none"] = "tf"
    #: Presidio loads a spaCy NER model, so this costs hundreds of MB the first
    #: time it runs. Off by default, and the packages are not in requirements.
    INDEX_PII_REMOVAL: bool = False
    #: Cap on the text handed to one indexing task. Past this the tail is
    #: dropped and a warning recorded rather than the broker carrying a payload
    #: measured in megabytes.
    INDEX_MAX_TEXT_CHARS: int = 400_000

    CHROMA_PERSIST_DIR: str = "./chroma_data"
    #: One collection per embedding model. A mismatched write is refused.
    CHROMA_COLLECTION_NAME: str = "web_scraping_chunks"

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
