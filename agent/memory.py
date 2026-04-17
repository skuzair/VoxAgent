from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

DEFAULT_LAST_RUN = {
    "transcript": "",
    "intent_payload": {},
    "action": "",
    "output": "",
    "status": "idle",
    "message": "",
}


def init_memory() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "pending_action" not in st.session_state:
        st.session_state["pending_action"] = None
    if "transcript_draft" not in st.session_state:
        st.session_state["transcript_draft"] = None
    if "transcript_editor" not in st.session_state:
        st.session_state["transcript_editor"] = ""
    if "confirm_overwrite" not in st.session_state:
        st.session_state["confirm_overwrite"] = False
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = deepcopy(DEFAULT_LAST_RUN)


def append_to_history(entry: Dict[str, Any]) -> None:
    history_entry = dict(entry)
    history_entry.setdefault("timestamp", datetime.now().strftime("%H:%M:%S"))
    st.session_state["history"].append(history_entry)


def get_history() -> List[Dict[str, Any]]:
    return st.session_state.get("history", [])


def clear_history() -> None:
    st.session_state["history"] = []
    st.session_state["pending_action"] = None
    st.session_state["transcript_draft"] = None
    st.session_state["transcript_editor"] = ""
    st.session_state["confirm_overwrite"] = False
    st.session_state["last_run"] = deepcopy(DEFAULT_LAST_RUN)


def get_last_run() -> Dict[str, Any]:
    return st.session_state.get("last_run", deepcopy(DEFAULT_LAST_RUN))


def set_last_run(data: Dict[str, Any]) -> None:
    current = deepcopy(DEFAULT_LAST_RUN)
    current.update(data)
    st.session_state["last_run"] = current
