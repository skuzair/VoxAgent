import json
from typing import Dict, List, Optional, Tuple

from . import DEFAULT_LLM_MODEL, create_chat_completion, read_prompt
from .validation import normalize_step, validate_intent_payload


def _strip_json_fences(raw_response: str) -> str:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _legacy_to_steps(payload: dict, raw_request: str) -> Dict:
    steps = []
    if bool(payload.get("compound")):
        for command in payload.get("commands", []) if isinstance(payload.get("commands"), list) else []:
            intent = str(command.get("intent", "chat")).strip().lower()
            parameters = command.get("parameters", {}) if isinstance(command.get("parameters"), dict) else {}
            normalized_step, _ = normalize_step({"intent": intent, **parameters}, raw_request)
            steps.append(normalized_step)
    else:
        intent = str(payload.get("intent", "chat")).strip().lower()
        parameters = payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {}
        normalized_step, _ = normalize_step({"intent": intent, **parameters}, raw_request)
        steps.append(normalized_step)

    return {
        "type": "compound" if len(steps) > 1 else "single",
        "steps": steps,
    }


def parse_intent_response(raw_response: str, raw_request: str) -> Dict:
    cleaned = _strip_json_fences(raw_response)
    payload = json.loads(cleaned)

    if not isinstance(payload, dict):
        raise ValueError("Intent payload must be a JSON object.")

    if "type" not in payload and ("intent" in payload or "compound" in payload):
        payload = _legacy_to_steps(payload, raw_request)

    return validate_intent_payload(payload, raw_request)


def _format_history(chat_history: List[dict]) -> str:
    if not chat_history:
        return "No prior session history."

    recent_entries = chat_history[-5:]
    formatted = []
    for entry in recent_entries:
        transcript = (entry.get("transcript") or "").strip()
        output = (entry.get("output") or "").strip()
        if not transcript and not output:
            continue
        formatted.append(
            f"User request: {transcript or '[empty]'}\nAssistant result: {output[:400] or '[empty]'}"
        )
    return "\n\n".join(formatted) if formatted else "No prior session history."


def build_intent_messages(transcript: str, chat_history: List[dict]) -> List[dict]:
    prompt = read_prompt("intent_prompt.txt")
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Session context:\n"
                f"{_format_history(chat_history)}\n\n"
                "Transcribed request:\n"
                f"{transcript}"
            ),
        },
    ]


def generate_intent_response(
    transcript: str,
    chat_history: List[dict],
    *,
    preferred_model: Optional[str] = None,
) -> Tuple[str, str]:
    messages = build_intent_messages(transcript, chat_history)
    return create_chat_completion(
        messages,
        preferred_model=preferred_model or DEFAULT_LLM_MODEL,
        temperature=0.0,
        max_tokens=450,
    )


def classify_intent(
    transcript: str,
    chat_history: List[dict],
    *,
    preferred_model: Optional[str] = None,
) -> Dict:
    messages = build_intent_messages(transcript, chat_history)

    try:
        model_used, raw_response = create_chat_completion(
            messages,
            preferred_model=preferred_model or DEFAULT_LLM_MODEL,
            temperature=0.0,
            max_tokens=450,
        )
        payload = parse_intent_response(raw_response, transcript)
        payload["model_used"] = model_used
        payload["raw_response"] = raw_response
        return payload
    except Exception:
        try:
            model_used, raw_response = create_chat_completion(
                messages
                + [
                    {
                        "role": "user",
                        "content": "Return only valid compact JSON using type and steps.",
                    }
                ],
                preferred_model=preferred_model or DEFAULT_LLM_MODEL,
                temperature=0.0,
                max_tokens=450,
            )
            payload = parse_intent_response(raw_response, transcript)
            payload["model_used"] = model_used
            payload["raw_response"] = raw_response
            payload["warnings"] = payload.get("warnings", []) + [
                "The first JSON parse attempt failed, so the classifier retried once."
            ]
            return payload
        except Exception:
            fallback_payload = validate_intent_payload({}, transcript)
            fallback_payload["fallback"] = True
            fallback_payload["model_used"] = preferred_model or DEFAULT_LLM_MODEL
            fallback_payload["raw_response"] = ""
            fallback_payload["warnings"] = fallback_payload.get("warnings", []) + [
                "Intent parsing failed completely, so the request was handled as chat."
            ]
            return fallback_payload
