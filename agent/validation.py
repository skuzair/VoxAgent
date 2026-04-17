from typing import Any, Dict, List, Tuple

SUPPORTED_INTENTS = {"create_file", "write_code", "summarize", "chat"}


def _compact_step(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in step.items()
        if value not in ("", None, [], {}) and value is not False
    }


def _default_chat_step(raw_request: str) -> Dict[str, Any]:
    return {"intent": "chat", "message": raw_request}


def normalize_step(step: Any, raw_request: str) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not isinstance(step, dict):
        warnings.append("Received a malformed step from the model. Fell back to chat.")
        return _default_chat_step(raw_request), warnings

    intent = str(step.get("intent", "chat")).strip().lower()
    if intent not in SUPPORTED_INTENTS:
        warnings.append(f"Unsupported intent '{intent}' was mapped to chat.")
        intent = "chat"

    normalized: Dict[str, Any] = {"intent": intent}

    if intent == "create_file":
        filename = str(step.get("filename", "")).strip()
        if not filename:
            warnings.append("Create file step was missing a filename.")
        else:
            normalized["filename"] = filename

        content = step.get("content", "")
        if content:
            normalized["content"] = str(content)

        if bool(step.get("is_directory", False)):
            normalized["is_directory"] = True

    elif intent == "write_code":
        filename = str(step.get("filename", "")).strip()
        instructions = str(step.get("instructions", step.get("description", ""))).strip()
        language = str(step.get("language", "")).strip() or "python"

        if not filename:
            warnings.append("Write code step was missing a filename.")
        else:
            normalized["filename"] = filename

        if not instructions:
            warnings.append("Write code step was missing instructions.")
            instructions = raw_request

        normalized["language"] = language
        normalized["instructions"] = instructions

    elif intent == "summarize":
        text = str(step.get("text", step.get("text_to_summarize", ""))).strip()
        if not text:
            warnings.append("Summarize step was missing text.")
        else:
            normalized["text"] = text

    else:
        message = str(step.get("message", "")).strip() or raw_request
        if not str(step.get("message", "")).strip():
            warnings.append("Chat step was missing a message and used the transcript instead.")
        normalized["message"] = message

    return _compact_step(normalized), warnings


def validate_intent_payload(payload: Any, raw_request: str) -> Dict[str, Any]:
    warnings: List[str] = []

    if not isinstance(payload, dict):
        warnings.append("Model output was not a JSON object. Fell back to chat.")
        return {
            "type": "single",
            "steps": [_default_chat_step(raw_request)],
            "raw_request": raw_request,
            "fallback": True,
            "warnings": warnings,
        }

    payload_type = str(payload.get("type", "single")).strip().lower()
    if payload_type not in {"single", "compound"}:
        warnings.append(f"Unknown payload type '{payload_type}' was normalized to 'single'.")
        payload_type = "single"

    raw_steps = payload.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        warnings.append("Model output was missing steps. Fell back to chat.")
        steps = [_default_chat_step(raw_request)]
        fallback = True
    else:
        steps = []
        fallback = False
        for raw_step in raw_steps:
            normalized_step, step_warnings = normalize_step(raw_step, raw_request)
            warnings.extend(step_warnings)
            steps.append(normalized_step)

    steps = [step for step in steps if step] or [_default_chat_step(raw_request)]

    executable_steps: List[Dict[str, Any]] = []
    for step in steps:
        intent = step.get("intent")
        if intent == "create_file" and not step.get("filename"):
            warnings.append("A create file step without a filename was skipped.")
            fallback = True
            continue
        if intent == "write_code" and not step.get("filename"):
            warnings.append("A write code step without a filename was skipped.")
            fallback = True
            continue
        if intent == "summarize" and not step.get("text"):
            warnings.append("A summarize step without text was skipped.")
            fallback = True
            continue
        executable_steps.append(step)

    if not executable_steps:
        executable_steps = [_default_chat_step(raw_request)]
        fallback = True
        warnings.append("No executable tool steps remained after validation. Fell back to chat.")

    if len(executable_steps) == 1:
        payload_type = "single"
    else:
        payload_type = "compound"

    return {
        "type": payload_type,
        "steps": executable_steps,
        "raw_request": raw_request,
        "fallback": fallback,
        "warnings": warnings,
    }
