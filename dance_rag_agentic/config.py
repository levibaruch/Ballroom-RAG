"""Paths, model names, URLs and every tunable number. Other modules import from here.

Models are chosen in settings.toml, overridden by environment variables, overridden by CLI flags
(use_models). Code reads MODELS.answer / MODELS.decision at call time, so a switch applies at once.
"""
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "theory"                     # theory/<style>/<Dance>.pdf (the books, not in git)
BIBR_DIR = PROJECT_ROOT / "extractions" / "bibr"       # one bibr JSON per book
CACHE_DIR = PROJECT_ROOT / "cache" / "agentic"          # everything built from the books
PYMUPDF_CACHE_DIR = CACHE_DIR / "pymupdf"              # one pickle per book
# The index the package reads (built by ingest/build_index.py). "v4" is its name in traces, eval files and
# /v1/health: the fourth index design, the one built from the parser comparison (docs/parser-comparison.md).
INDEX_VERSION = "v4"
INDEX_CACHE_DIR = Path(os.environ.get("DANCE_RAG_INDEX_DIR", CACHE_DIR / "index_v4"))   # env: another index (tests)


def all_book_pdfs():
    """Every technique book in theory/<style>/, sorted by style and dance."""
    return sorted(DATA_DIR.glob("*/*.pdf"))


def select_book_pdfs(dance_names=None):
    """All books, or only those whose file name (= dance) is in dance_names (case-insensitive)."""
    if not dance_names:
        return all_book_pdfs()
    wanted = {name.lower() for name in dance_names}
    selected = [pdf_path for pdf_path in all_book_pdfs() if pdf_path.stem.lower() in wanted]
    missing = wanted - {pdf_path.stem.lower() for pdf_path in selected}
    if missing:
        raise ValueError(f"no PDF for: {sorted(missing)}; available: {[p.stem for p in all_book_pdfs()]}")
    return selected


# ---------------------------------------------------------------- extraction
# bibr lives in its own environment (needs mlx-vlm + libmagic): .venv-bibr in the project root.
# Override with the BIBR_EXECUTABLE environment variable.
PROJECT_BIBR = PROJECT_ROOT / ".venv-bibr" / "bin" / "bibr"
BIBR_EXECUTABLE = os.environ.get("BIBR_EXECUTABLE", str(PROJECT_BIBR) if PROJECT_BIBR.exists() else "bibr")
BIBR_ARGUMENTS = ["--no-llm", "--refs", "off", "--device", "cpu", "--ocr", "paddle-mlx-vlm"]

# ---------------------------------------------------------------- embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # has a Normalize layer: vectors come out length 1, dot = cosine
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge needs this on queries


def _load_api_keys(path):
    """The API keys (API_PROVIDERS, below) from deploy/.env into os.environ; a variable already set wins.
    Only the keys: the other deploy settings there (rate limit, origins, models) are for the VPS, not local runs."""
    if not path.exists():
        return
    wanted = set(API_PROVIDERS.values())
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key.strip() in wanted and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------- other API providers (answer model only)
# A model name with a provider prefix runs on that provider's API instead of Ollama: "cerebras:gpt-oss-120b".
# Only the ANSWER model: the decision model needs Ollama's /v1/systemone (no other provider has it).
API_PROVIDERS = {"cerebras": "CEREBRAS_API_KEY"}      # prefix -> environment variable with the API key
_load_api_keys(PROJECT_ROOT / "deploy" / ".env")     # gitignored; deploy/.env.example shows the lines


def api_provider(model_name):
    """("cerebras", "gpt-oss-120b") for "cerebras:gpt-oss-120b"; (None, name) for an Ollama model."""
    prefix, separator, rest = str(model_name).partition(":")
    return (prefix, rest) if separator and prefix in API_PROVIDERS else (None, model_name)

# ---------------------------------------------------------------- models (Ollama, or an API provider above)
SETTINGS_PATH = Path(__file__).resolve().parent / "settings.toml"
_settings = tomllib.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
OLLAMA_URL = os.environ.get("OLLAMA_URL", _settings.get("ollama", {}).get("url", "http://localhost:11434"))
KEEP_ALIVE = os.environ.get("DANCE_RAG_AGENTIC_KEEP_ALIVE", _settings.get("ollama", {}).get("keep_alive", "30m"))
PRICES = {model: tuple(price) for model, price in _settings.get("prices", {}).items()}   # USD per 1M [in, out]
CLOUD_MODELS = {role: list(_settings.get("cloud", {}).get(role, [])) for role in ("answer", "decision")}


@dataclass
class Models:
    answer: str          # searching agents + answer writer (Ollama 'completion' + 'tools', or 'cerebras:...')
    decision: str        # kind/aspect choices, yes/no checks, claim judge (needs Ollama 'decision')


MODELS = Models(
    answer=os.environ.get("DANCE_RAG_AGENTIC_ANSWER_MODEL",
                          _settings.get("models", {}).get("answer", "gpt-oss:20b")),
    decision=os.environ.get("DANCE_RAG_AGENTIC_DECISION_MODEL",
                            _settings.get("models", {}).get("decision", "autotrust-jev")),
)


# gpt-oss reasoning effort for every answer-model call (agents, writer, repair, polish): low / medium / high.
# Not set before 2026-11-14 = each provider's default (Ollama ~medium, Cerebras somewhat above). "low" = fewer
# reasoning tokens (cheaper, a bit faster); env DANCE_RAG_AGENTIC_REASONING overrides.
REASONING = os.environ.get("DANCE_RAG_AGENTIC_REASONING", _settings.get("models", {}).get("reasoning", "medium"))


class CloudAccessError(RuntimeError):
    """A model server refused the account (401/402/403: not signed in, no credits). Retrying or carrying on
    cannot help, and an answer written without models would be misleading: the run stops with this message."""


def use_models(answer=None, decision=None):
    """Override the models for this process (CLI flags). None keeps the current choice."""
    if answer:
        MODELS.answer = answer
    if decision:
        MODELS.decision = decision

# ---------------------------------------------------------------- routing
# Top kind probability (after rule 1) below this -> UNSURE route. Was "confidence (1 - entropy/ln N) < 0.4": adding
# a 6th kind (GENERAL) lowered every confidence, so 'How do I dance the Fan in Rumba?' (FIGURE 0.62, conf 0.33) went
# UNSURE. On the 100 gold drafts: top p < 0.5 -> 84 ok / 12 UNSURE / 4 wrong; conf < 0.4 -> 72 / 24 / 4 (2026-11-13).
UNSURE_BELOW = 0.5
MIXED_FROM = 0.5             # decision model P(technical + general part) from this -> SplitQuestion
ASPECT_MIN_PROBABILITY = 0.5  # top aspect below this -> aspect "all" (timing 0.36 vs all 0.36 is a coin toss)

# ---------------------------------------------------------------- agents
# Not guessed: start high, then set near the 90th percentile of tool calls in passing eval runs.
AGENT_TOOL_CALLS = int(os.environ.get("DANCE_RAG_AGENT_TOOL_CALLS", 15))   # prose agent and UNSURE agent
FILLGAP_TOOL_CALLS = 3        # one FillGap round (searching evidence for unsupported claims)
CHARACTER_TOOL_CALLS = 6      # CharacterPath agent: code already read the music and principles sections
BACKGROUND_TOOL_CALLS = int(os.environ.get("DANCE_RAG_BACKGROUND_TOOL_CALLS", 6))   # Background agent (FIGURE route): prose for the chart's terms
BACKGROUND_TERMS = 25         # at most this many distinct chart terms are handed to the Background agent
TERM_HEADING_SECTIONS = 4      # TermPath: sections whose heading (2+ words) is named in the question

# ---------------------------------------------------------------- answer + checks
ANSWER_ATTEMPTS = 3           # per answer: first try + at most 2 rewrites after CheckCites
REPAIR_ROUNDS = 2             # JudgeClaims -> (FillGap) -> Answer, at most this often
# SUPPORTED below this = not accepted (-> repair). Was 0.5; 2026-11-13 batch: two wrong chart claims (left/right foot,
# hip design per step) passed at 0.62 and 0.57, while 90% of accepted claims (n=376) scored >= 0.72.
JUDGE_SUPPORTED_FROM = 0.7
JUDGE_CLAIMS_PER_CALL = 8     # claims per decision-model call; one call with 21 claims over two charts crashed it
# Decision calls sent at the same time (JudgeClaims batches + the 'answers the question' check). They are
# independent; one after another they were ~45% of a question's time (2026-11-14, ~1 s each on Ollama Cloud).
# A local decision model gets 1: one Ollama job at a time (two make both slow).
DECISION_PARALLEL_CLOUD = 6
ANSWERS_QUESTION_FROM = 0.5   # decision model P(answer responds to the whole question) below this = repair
DECISION_MAX_STATE_CHARACTERS = 60_000   # Ollama limit for the decision state is 64 KiB

# ---------------------------------------------------------------- trace (--trace)
TRACE_RESULT_LINES = 8        # lines of each agent tool result shown in the trace
TRACE_DIR = CACHE_DIR / "traces"   # every --trace run is also saved here as plain text

# ---------------------------------------------------------------- search output
MAX_HITS = 8                  # sections per search result (small model: keep tool output short)
# The search tool's output format and its single hybrid search were chosen by A/B tests: docs/decisions.md.
SNIPPET_CHARACTERS = 200
READ_UNIT_CHARACTERS = 5000   # read_section shows a whole heading unit up to this size; bigger: the asked part + a list
