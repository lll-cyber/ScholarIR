"""Project-level config. SPAR knobs live in vendor_spar.global_config."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_ROOT = PROJECT_ROOT / "logs"


def _load_env_files() -> None:
    """Load KEY=VALUE from .env files (does not override existing env vars)."""
    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "configs" / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_env_files()

PASA_TEST_JSONL = DATA_ROOT / "pasa-dataset" / "AutoScholarQuery" / "test.jsonl"
PASA_REAL_JSONL = DATA_ROOT / "pasa-dataset" / "RealScholarQuery" / "test.jsonl"

# Pipeline defaults (v0 demo)
# semantic_scholar is optional: fast eval uses arxiv + openalex (no rate limits)
DEFAULT_SOURCES = ["arxiv", "openalex"]
DEFAULT_PER_QUERY_TOPK = 10
DEFAULT_JUDGE_THRESHOLD = 0.5
DEFAULT_MAX_RETURN = 20
DEFAULT_MAX_SUBQUERIES = 3

# Semantic Scholar (https://api.semanticscholar.org/)
S2_API_KEY = os.getenv("S2_API_KEY", "").strip().strip('"').strip("'")
# Authenticated tier often 1 req/s — enforce globally in search/s2_client.py
S2_RATE_LIMIT_RPS = float(os.getenv("S2_RATE_LIMIT_RPS", "1"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY", "")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")

# DeepSeek API (Understanding)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "768"))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))
# For JSON extraction, disable thinking by default (faster/cheaper).
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "0").lower() in ("1", "true", "yes")

# LLM backend preference: deepseek | local | openai | auto
SCHOLAR_IR_LLM_BACKEND = os.getenv("SCHOLAR_IR_LLM_BACKEND", "auto").lower()

# Embedding API (OpenAI-compatible /v1/embeddings; e.g. RAGFlow / vLLM / SiliconFlow)
# EMBEDDING_MODEL may be a comma-separated fallback list.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip().strip('"').strip("'")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "").strip().strip('"').strip("'")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3").strip().strip('"').strip("'")
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "30"))
EMBEDDING_MAX_CHARS = int(os.getenv("EMBEDDING_MAX_CHARS", "1800"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Local LLM (Understanding): ScholarIR/LLMS/qwen2.5-gptq
LOCAL_QWEN_GPTQ_PATH = Path(
    os.getenv(
        "SCHOLAR_IR_LOCAL_MODEL_PATH",
        str(PROJECT_ROOT / "LLMS" / "qwen2.5-gptq"),
    )
)
# OpenAI-compatible local server (vLLM / llama.cpp); if set, preferred over in-process HF
LOCAL_LLM_BASE_URL = os.getenv("SCHOLAR_IR_LLM_URL", os.getenv("LOCAL_LLM_BASE_URL", ""))
LOCAL_LLM_MODEL = os.getenv("SCHOLAR_IR_LLM_MODEL", os.getenv("LOCAL_LLM_MODEL", ""))
LOCAL_QWEN_MAX_NEW_TOKENS = int(os.getenv("SCHOLAR_IR_LLM_MAX_TOKENS", "768"))
LOCAL_QWEN_TEMPERATURE = float(os.getenv("SCHOLAR_IR_LLM_TEMPERATURE", "0.2"))
