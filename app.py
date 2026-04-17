from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from agent import DEFAULT_LLM_MODEL, OUTPUT_DIR
from agent.intent import classify_intent
from agent.memory import (
    append_to_history,
    clear_history,
    get_history,
    get_last_run,
    init_memory,
    set_last_run,
)
from agent.stt import STTError, transcribe_audio
from agent import tools

st.set_page_config(page_title="Voice-Controlled Local AI Agent", page_icon="🎙️", layout="wide")

INTENT_LABELS = {
    "create_file": "Create File",
    "write_code": "Write Code",
    "summarize": "Summarize Text",
    "chat": "General Chat",
}


def get_audio_input(
    source: str, uploaded_file, mic_audio: Optional[dict]
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if source == "Upload":
        if not uploaded_file:
            return None, None, None
        suffix = Path(uploaded_file.name).suffix.lower().replace(".", "") or "wav"
        return uploaded_file.getvalue(), suffix, uploaded_file.name

    if not mic_audio:
        return None, None, None
    return mic_audio.get("bytes"), "webm", "microphone recording"


def get_steps(intent_payload: dict) -> List[dict]:
    steps = intent_payload.get("steps", [])
    return steps if isinstance(steps, list) else []


def derive_display_intent(intent_payload: dict) -> str:
    steps = get_steps(intent_payload)
    if not steps:
        return "Unknown"

    labels = [INTENT_LABELS.get(step.get("intent", ""), step.get("intent", "Unknown")) for step in steps]
    if len(labels) > 1:
        return "Compound: " + " -> ".join(labels)
    return labels[0]


def get_primary_intent(intent_payload: dict) -> str:
    steps = get_steps(intent_payload)
    if not steps:
        return ""
    return steps[0].get("intent", "")


def needs_confirmation(intent_payload: dict) -> bool:
    return any(step.get("intent") in {"create_file", "write_code"} for step in get_steps(intent_payload))


def build_action_summary(intent_payload: dict) -> List[str]:
    summaries: List[str] = []
    for step in get_steps(intent_payload):
        intent = step.get("intent")
        if intent == "create_file":
            noun = "folder" if step.get("is_directory") else "file"
            summaries.append(f"Create {noun}: {step.get('filename', '[missing filename]')}")
        elif intent == "write_code":
            summaries.append(
                f"Generate {step.get('language', 'python')} code in {step.get('filename', '[missing filename]')}"
            )
        elif intent == "summarize":
            summaries.append("Summarize the provided text.")
        else:
            summaries.append("Respond in general chat mode.")
    return summaries


def find_file_conflicts(intent_payload: dict) -> List[dict]:
    conflicts: List[dict] = []
    for step in get_steps(intent_payload):
        if step.get("intent") not in {"create_file", "write_code"}:
            continue
        if step.get("is_directory"):
            continue
        conflict = tools.check_output_conflict(
            filename=step.get("filename", ""),
            is_directory=bool(step.get("is_directory", False)),
        )
        if not conflict.get("success"):
            conflicts.append({"path": "", "message": conflict.get("message", "Invalid file path.")})
        elif conflict.get("exists"):
            conflicts.append(
                {
                    "path": conflict.get("path", ""),
                    "message": f"{step.get('filename', 'A file')} already exists.",
                }
            )
    return conflicts


def execute_single_command(
    step: dict,
    history: List[dict],
    *,
    previous_output: str = "",
    allow_overwrite: bool = False,
) -> Dict:
    intent_name = step.get("intent", "chat")

    if intent_name == "create_file":
        content = step.get("content", "")
        if not content and previous_output:
            content = previous_output
        return tools.create_file(
            filename=step.get("filename", ""),
            content=content,
            is_directory=bool(step.get("is_directory", False)),
            overwrite=allow_overwrite,
        )

    if intent_name == "write_code":
        return tools.write_code(
            filename=step.get("filename", ""),
            language=step.get("language", "") or "python",
            description=step.get("instructions", "") or step.get("message", ""),
            overwrite=allow_overwrite,
        )

    if intent_name == "summarize":
        return tools.summarize_text(text=step.get("text", ""))

    return tools.general_chat(
        message=step.get("message", ""),
        history=history,
    )


def execute_intent_payload(intent_payload: dict, history: List[dict], *, allow_overwrite: bool = False) -> Dict:
    steps = get_steps(intent_payload)
    results = []
    previous_output = ""

    for step in steps:
        result = execute_single_command(
            step,
            history,
            previous_output=previous_output,
            allow_overwrite=allow_overwrite,
        )
        results.append({"intent": step.get("intent", "chat"), "result": result})
        if result.get("success") and result.get("output"):
            previous_output = result["output"]
        if not result.get("success"):
            break

    if not results:
        return {
            "success": False,
            "message": "No executable steps were found.",
            "action": "No action was taken.",
            "output": "",
            "results": [],
        }

    overall_success = all(item["result"]["success"] for item in results)
    if len(results) == 1:
        single_result = results[0]["result"]
        return {
            "success": overall_success,
            "message": single_result["message"],
            "action": single_result["action"],
            "output": single_result.get("output", "") or single_result["message"],
            "results": results,
        }

    action_lines = [item["result"]["action"] for item in results]
    output_blocks = []
    for index, item in enumerate(results, start=1):
        label = INTENT_LABELS.get(item["intent"], item["intent"])
        output = item["result"].get("output", "").strip()
        body = output or item["result"]["message"]
        output_blocks.append(f"Step {index} ({label})\n{body}")

    return {
        "success": overall_success,
        "message": "Completed all requested actions." if overall_success else results[-1]["result"]["message"],
        "action": "\n".join(action_lines),
        "output": "\n\n".join(output_blocks),
        "results": results,
    }


def start_transcription(audio_bytes: bytes, audio_format: str, source_label: str) -> None:
    try:
        with st.spinner("Processing audio with Groq Whisper..."):
            transcript = transcribe_audio(audio_bytes, audio_format)
    except STTError as exc:
        st.session_state["transcript_draft"] = None
        set_last_run(
            {
                "transcript": "",
                "intent_payload": {},
                "action": "Speech-to-text failed.",
                "output": "",
                "status": "error",
                "message": f"Could not understand the audio. {exc}",
                "warnings": [],
            }
        )
        append_to_history(
            {
                "transcript": "",
                "intent": "STT Error",
                "action": "Speech-to-text failed.",
                "output": str(exc),
                "status": "FAILED",
            }
        )
        return

    warnings = []
    if len(transcript.split()) <= 2:
        warnings.append("The transcription is very short. Review it carefully before continuing.")

    st.session_state["transcript_draft"] = {
        "transcript": transcript,
        "source_label": source_label,
    }
    st.session_state["transcript_editor"] = transcript
    st.session_state["pending_action"] = None
    st.session_state["confirm_overwrite"] = False

    set_last_run(
        {
            "transcript": transcript,
            "intent_payload": {},
            "action": f"Transcribed audio from {source_label}.",
            "output": "",
            "status": "warning" if warnings else "info",
            "message": "Review and edit the transcript if needed, then continue to intent classification.",
            "warnings": warnings,
        }
    )


def process_transcript(transcript: str) -> None:
    clean_transcript = (transcript or "").strip()
    if not clean_transcript:
        set_last_run(
            {
                "transcript": "",
                "intent_payload": {},
                "action": "Transcript review blocked execution.",
                "output": "",
                "status": "warning",
                "message": "Please enter a transcript before continuing.",
                "warnings": [],
            }
        )
        return

    with st.spinner(f"Classifying intent with {DEFAULT_LLM_MODEL}..."):
        intent_payload = classify_intent(clean_transcript, get_history(), preferred_model=DEFAULT_LLM_MODEL)

    warnings = list(intent_payload.get("warnings", []))
    conflicts = find_file_conflicts(intent_payload) if needs_confirmation(intent_payload) else []
    if conflicts:
        warnings.append("One or more output files already exist. Overwrite is disabled by default.")

    fallback = bool(intent_payload.get("fallback"))
    message = "Intent classified successfully."
    if fallback:
        message = "The intent output was uncertain, so the request was safely normalized."

    set_last_run(
        {
            "transcript": clean_transcript,
            "intent_payload": intent_payload,
            "action": "Transcript approved and intent classified.",
            "output": "",
            "status": "warning" if warnings or fallback else "info",
            "message": message,
            "warnings": warnings,
        }
    )

    st.session_state["transcript_draft"] = None

    if needs_confirmation(intent_payload):
        st.session_state["pending_action"] = {
            "transcript": clean_transcript,
            "intent_payload": intent_payload,
            "conflicts": conflicts,
        }
        set_last_run(
            {
                "transcript": clean_transcript,
                "intent_payload": intent_payload,
                "action": "Awaiting confirmation before writing inside output/.",
                "output": "",
                "status": "awaiting_confirmation",
                "message": "Review the action summary below and confirm before file execution.",
                "warnings": warnings,
            }
        )
        return

    execution = execute_intent_payload(intent_payload, get_history(), allow_overwrite=False)
    finalize_execution(clean_transcript, intent_payload, execution, warnings=warnings)


def finalize_execution(transcript: str, intent_payload: dict, execution: Dict, *, warnings: Optional[List[str]] = None) -> None:
    warning_list = list(warnings or [])
    status = "success" if execution["success"] else "error"
    message = execution["message"]

    if execution.get("results"):
        last_result = execution["results"][-1]["result"]
        if last_result.get("error_type") == "overwrite_required":
            status = "warning"
            message = "A file already exists. Use overwrite confirmation or edit the transcript to change the filename."

    set_last_run(
        {
            "transcript": transcript,
            "intent_payload": intent_payload,
            "action": execution["action"],
            "output": execution["output"],
            "status": status,
            "message": message if execution["success"] else f"Execution stopped safely. {message}",
            "warnings": warning_list,
        }
    )
    append_to_history(
        {
            "transcript": transcript,
            "intent": derive_display_intent(intent_payload),
            "action": execution["action"],
            "output": execution["output"],
            "status": "SUCCESS" if execution["success"] else "FAILED",
        }
    )


def handle_pending_confirmation(confirm: bool) -> None:
    pending = st.session_state.get("pending_action")
    if not pending:
        return

    transcript = pending["transcript"]
    intent_payload = pending["intent_payload"]
    warnings = list(get_last_run().get("warnings", []))

    if not confirm:
        set_last_run(
            {
                "transcript": transcript,
                "intent_payload": intent_payload,
                "action": "Cancelled by user before file execution.",
                "output": "",
                "status": "warning",
                "message": "The pending action was cancelled.",
                "warnings": warnings,
            }
        )
        append_to_history(
            {
                "transcript": transcript,
                "intent": derive_display_intent(intent_payload),
                "action": "Cancelled by user before file execution.",
                "output": "",
                "status": "CANCELLED",
            }
        )
        st.session_state["pending_action"] = None
        return

    allow_overwrite = bool(st.session_state.get("confirm_overwrite", False))
    with st.spinner("Executing tool actions..."):
        execution = execute_intent_payload(intent_payload, get_history(), allow_overwrite=allow_overwrite)

    finalize_execution(transcript, intent_payload, execution, warnings=warnings)
    st.session_state["pending_action"] = None
    st.session_state["confirm_overwrite"] = False


def send_pending_back_to_edit() -> None:
    pending = st.session_state.get("pending_action")
    if not pending:
        return

    st.session_state["transcript_draft"] = {
        "transcript": pending["transcript"],
        "source_label": "reviewed transcript",
    }
    st.session_state["transcript_editor"] = pending["transcript"]
    st.session_state["pending_action"] = None
    st.session_state["confirm_overwrite"] = False
    set_last_run(
        {
            "transcript": pending["transcript"],
            "intent_payload": {},
            "action": "Returned to transcript review.",
            "output": "",
            "status": "warning",
            "message": "Edit the transcript and continue when ready.",
            "warnings": [],
        }
    )


def render_status(message: str, status: str) -> None:
    if not message:
        return
    if status == "success":
        st.success(message)
    elif status == "error":
        st.error(message)
    elif status in {"warning", "awaiting_confirmation"}:
        st.warning(message)
    else:
        st.info(message)


def render_warning_list(warnings: List[str]) -> None:
    if not warnings:
        return
    for warning in warnings:
        st.caption(f"- {warning}")


def render_transcript_review() -> None:
    draft = st.session_state.get("transcript_draft")
    if not draft:
        return

    with st.container(border=True):
        st.subheader("Transcription")
        st.caption("Review and edit the transcript before intent classification.")
        st.text_area("Editable Transcript", key="transcript_editor", height=180)
        continue_col, discard_col = st.columns(2)
        with continue_col:
            if st.button("Continue", type="primary", use_container_width=True):
                process_transcript(st.session_state.get("transcript_editor", ""))
                st.rerun()
        with discard_col:
            if st.button("Discard", use_container_width=True):
                st.session_state["transcript_draft"] = None
                st.session_state["transcript_editor"] = ""
                set_last_run(
                    {
                        "transcript": "",
                        "intent_payload": {},
                        "action": "Discarded the transcript draft.",
                        "output": "",
                        "status": "warning",
                        "message": "Transcript discarded. Record or upload new audio to try again.",
                        "warnings": [],
                    }
                )
                st.rerun()


def render_confirmation() -> None:
    pending = st.session_state.get("pending_action")
    if not pending:
        return

    with st.container(border=True):
        st.subheader("Action Confirmation")
        st.caption("Human-in-the-loop safeguard before local file changes.")

        for summary in build_action_summary(pending["intent_payload"]):
            st.write(f"- {summary}")

        conflicts = pending.get("conflicts", [])
        if conflicts:
            st.warning("Potential overwrite conflicts detected.")
            for conflict in conflicts:
                st.caption(f"- {conflict['message']} {conflict['path']}".strip())

        st.checkbox(
            "Allow overwrite for existing files in output/",
            key="confirm_overwrite",
            value=False,
        )

        with st.expander("View Structured Intent"):
            st.json(pending["intent_payload"])

        confirm_col, edit_col, cancel_col = st.columns(3)
        with confirm_col:
            if st.button("Confirm Action", type="primary", use_container_width=True):
                handle_pending_confirmation(confirm=True)
                st.rerun()
        with edit_col:
            if st.button("Edit Transcript", use_container_width=True):
                send_pending_back_to_edit()
                st.rerun()
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                handle_pending_confirmation(confirm=False)
                st.rerun()


def render_history() -> None:
    history = list(reversed(get_history()))
    with st.container(border=True):
        st.subheader("History")
        st.caption("Session memory stored in Streamlit state.")
        if not history:
            st.info("No actions yet in this session.")
            return

        for entry in history:
            with st.expander(
                f"{entry['timestamp']} | {entry.get('status', 'UNKNOWN')} | {entry.get('intent', 'Unknown')}",
                expanded=False,
            ):
                st.write(entry.get("action", ""))
                if entry.get("output"):
                    st.text(entry["output"][:1200])


def main() -> None:
    init_memory()

    st.title("Voice-Controlled Local AI Agent")
    st.caption("Minimal demo pipeline: audio input, transcript review, intent validation, safe execution, and history.")

    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.caption(f"LLM default: {DEFAULT_LLM_MODEL} | STT: whisper-large-v3 | Output sandbox: {OUTPUT_DIR}")
    with header_right:
        if st.button("Clear History", use_container_width=True):
            clear_history()
            st.rerun()

    with st.container(border=True):
        st.subheader("Audio Input")
        st.caption("Record from the microphone or upload a supported audio file.")
        source = st.radio("Choose audio source", ("Microphone", "Upload"), horizontal=True)
        input_col, preview_col = st.columns(2)

        with input_col:
            if source == "Microphone":
                mic_audio = mic_recorder(
                    start_prompt="Start recording",
                    stop_prompt="Stop recording",
                    format="webm",
                    key="mic_recorder",
                )
                uploaded_file = None
            else:
                uploaded_file = st.file_uploader(
                    "Upload an audio file",
                    type=["wav", "mp3", "m4a", "ogg", "webm"],
                )
                mic_audio = None

        audio_bytes, audio_format, source_label = get_audio_input(source, uploaded_file, mic_audio)

        with preview_col:
            st.markdown("**Audio Preview**")
            if audio_bytes:
                mime_lookup = {
                    "mp3": "audio/mpeg",
                    "wav": "audio/wav",
                    "m4a": "audio/mp4",
                    "ogg": "audio/ogg",
                    "webm": "audio/webm",
                }
                st.audio(audio_bytes, format=mime_lookup.get(audio_format or "", "audio/wav"))
                st.caption(f"Source: {source_label}")
            else:
                st.info("Record audio or upload a file to begin.")

        if st.button("Run Agent", type="primary", use_container_width=True):
            if not audio_bytes or not audio_format or not source_label:
                set_last_run(
                    {
                        "transcript": "",
                        "intent_payload": {},
                        "action": "No audio source selected.",
                        "output": "",
                        "status": "warning",
                        "message": "Please record audio or upload a supported file before running the agent.",
                        "warnings": [],
                    }
                )
            else:
                st.session_state["pending_action"] = None
                start_transcription(audio_bytes, audio_format, source_label)
            st.rerun()

    render_transcript_review()
    render_confirmation()

    last_run = get_last_run()
    render_status(last_run.get("message", ""), last_run.get("status", "idle"))
    render_warning_list(last_run.get("warnings", []))

    transcript_col, intent_col = st.columns(2)
    with transcript_col:
        with st.container(border=True):
            st.subheader("Transcription")
            current_transcript = (
                st.session_state.get("transcript_editor")
                if st.session_state.get("transcript_draft")
                else last_run.get("transcript")
            )
            if current_transcript:
                st.text_area(
                    "Transcribed Text",
                    value=current_transcript,
                    height=180,
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.info("The transcript will appear here after processing audio.")

    with intent_col:
        with st.container(border=True):
            st.subheader("Intent")
            if last_run.get("intent_payload"):
                st.write(derive_display_intent(last_run["intent_payload"]))
                st.json(last_run["intent_payload"])
            else:
                st.info("Validated intent output will appear here after transcript review.")

    with st.container(border=True):
        st.subheader("Action")
        if last_run.get("action"):
            st.text_area(
                "Action Taken",
                value=last_run["action"],
                height=120,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info("The executed action summary will appear here.")

    with st.container(border=True):
        st.subheader("Output")
        if last_run.get("output"):
            primary_intent = get_primary_intent(last_run.get("intent_payload", {}))
            if primary_intent == "write_code" and len(get_steps(last_run.get("intent_payload", {}))) == 1:
                language = get_steps(last_run.get("intent_payload", {}))[0].get("language", "python")
                st.code(last_run["output"], language=language or "python")
            else:
                st.text_area(
                    "Final Output",
                    value=last_run["output"],
                    height=240,
                    disabled=True,
                    label_visibility="collapsed",
                )
        else:
            st.info("Generated responses and local file results will appear here.")

    render_history()


if __name__ == "__main__":
    main()
