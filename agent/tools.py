from pathlib import Path
from typing import Any, Dict, List

from . import DEFAULT_LLM_MODEL, OUTPUT_DIR, create_chat_completion, read_prompt


class ToolExecutionError(Exception):
    """Raised when a local tool cannot safely complete."""


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _success_result(message: str, output: str = "", action: str = "") -> Dict[str, Any]:
    return {
        "success": True,
        "message": message,
        "output": output,
        "action": action or message,
    }


def _error_result(message: str, action: str = "") -> Dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "output": "",
        "action": action or message,
    }


def _resolve_output_path(filename: str) -> Path:
    clean_name = (filename or "").strip()
    if not clean_name:
        raise ToolExecutionError("A filename is required.")

    candidate = Path(clean_name)
    if candidate.is_absolute():
        raise ToolExecutionError("Absolute paths are not allowed. Save files inside output/ only.")

    resolved = (OUTPUT_DIR / candidate).resolve()
    output_root = OUTPUT_DIR.resolve()

    if resolved != output_root and output_root not in resolved.parents:
        raise ToolExecutionError("Invalid filename. Path traversal outside output/ is blocked.")

    return resolved


def check_output_conflict(filename: str, is_directory: bool = False) -> Dict[str, Any]:
    try:
        target_path = _resolve_output_path(filename)
        exists = target_path.exists()
        return {
            "success": True,
            "exists": exists,
            "path": str(target_path),
            "is_directory": is_directory,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "exists": False,
            "path": "",
            "is_directory": is_directory,
            "message": str(exc),
        }


def create_file(
    filename: str,
    content: str = "",
    is_directory: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    try:
        target_path = _resolve_output_path(filename)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if is_directory or str(filename).endswith(("/", "\\")):
            target_path.mkdir(parents=True, exist_ok=True)
            return _success_result(
                message=f"Created folder at {target_path}",
                output=str(target_path),
                action=f"Created folder '{filename}' in output/.",
            )

        if target_path.exists() and not overwrite:
            return {
                "success": False,
                "message": f"File '{filename}' already exists. Confirm overwrite to replace it.",
                "output": "",
                "action": f"Skipped writing '{filename}' because it already exists.",
                "error_type": "overwrite_required",
                "path": str(target_path),
            }

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content or "", encoding="utf-8")
        return _success_result(
            message=f"Created file at {target_path}",
            output=content or "",
            action=f"Created file '{filename}' in output/.",
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(str(exc), action=f"Failed to create '{filename}'.")


def write_code(filename: str, language: str, description: str, overwrite: bool = False) -> Dict[str, Any]:
    if not filename:
        return _error_result("A filename is required for code generation.")
    if not description:
        return _error_result("A code description is required.")

    prompt = read_prompt("codegen_prompt.txt").format(
        language=language or "python",
        description=description,
    )

    try:
        _, code = create_chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the requested code."},
            ],
            preferred_model=DEFAULT_LLM_MODEL,
            temperature=0.2,
            max_tokens=1800,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Code generation failed: {exc}", action="Code generation failed.")

    code = _strip_code_fences(code)
    file_result = create_file(
        filename=filename,
        content=code,
        is_directory=False,
        overwrite=overwrite,
    )
    if not file_result["success"]:
        return file_result

    file_result["action"] = f"Generated {language or 'code'} and saved it to '{filename}'."
    file_result["message"] = f"Generated code and saved it to {filename}"
    file_result["output"] = code
    return file_result


def summarize_text(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return _error_result("No text was provided to summarize.")

    prompt = read_prompt("summarize_prompt.txt").format(text=text.strip())

    try:
        _, summary = create_chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Summarize the provided text."},
            ],
            preferred_model=DEFAULT_LLM_MODEL,
            temperature=0.2,
            max_tokens=700,
        )
        return _success_result(
            message="Summarized the provided text.",
            output=summary,
            action="Generated a concise summary.",
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Summarization failed: {exc}", action="Summarization failed.")


def general_chat(message: str, history: List[dict]) -> Dict[str, Any]:
    if not message or not message.strip():
        return _error_result("A chat message is required.")

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful local AI agent. Be concise, practical, and grounded in the user's request."
            ),
        }
    ]

    for entry in history[-6:]:
        transcript = (entry.get("transcript") or "").strip()
        output = (entry.get("output") or "").strip()
        if transcript:
            messages.append({"role": "user", "content": transcript})
        if output:
            messages.append({"role": "assistant", "content": output})

    messages.append({"role": "user", "content": message.strip()})

    try:
        _, reply = create_chat_completion(
            messages,
            preferred_model=DEFAULT_LLM_MODEL,
            temperature=0.4,
            max_tokens=900,
        )
        return _success_result(
            message="Generated a chat response.",
            output=reply,
            action="Responded to the user in chat mode.",
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Chat failed: {exc}", action="Chat response failed.")
