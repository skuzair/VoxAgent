import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
OUTPUT_DIR = ROOT_DIR / "output"

DEFAULT_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "mixtral-8x7b-32768")
FALLBACK_LLM_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama3-70b-8192")
LIGHTWEIGHT_LLM_MODEL = os.getenv("GROQ_LIGHTWEIGHT_MODEL", "llama3-8b-8192")

BENCHMARK_LLM_MODELS = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama3-8b-8192",
]


def _dedupe_models(models: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for model in models:
        if not model:
            continue
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def get_llm_model_candidates(preferred_model: Optional[str] = None) -> List[str]:
    return _dedupe_models(
        [
            preferred_model,
            DEFAULT_LLM_MODEL,
            FALLBACK_LLM_MODEL,
            LIGHTWEIGHT_LLM_MODEL,
            "llama-3.3-70b-versatile",
        ]
    )


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return Groq(api_key=api_key)


def read_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    return prompt_path.read_text(encoding="utf-8").strip()


def create_chat_completion(
    messages: Sequence[dict],
    *,
    preferred_model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> Tuple[str, str]:
    client = get_groq_client()
    last_error: Optional[Exception] = None

    for model in get_llm_model_candidates(preferred_model):
        try:
            request = {
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens

            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            return model, content.strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Groq chat completion failed: {last_error}") from last_error
