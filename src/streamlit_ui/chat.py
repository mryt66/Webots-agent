import base64
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
from llm.client import GeminiClient, create_gemini_client_from_env
from llm.gemini_client import (
    build_generate_content_payload,
    build_request_contents,
)
from llm.prompts import PLANNER_PROMPT
from webots.client import execute_tool_call, fetch_screenshot


APP_BOOT_ID = str(uuid.uuid4())


def _reset_state_on_boot_change() -> None:
    current = st.session_state.get("app_boot_id")
    if str(current or "") == str(APP_BOOT_ID):
        return
    st.session_state.app_boot_id = str(APP_BOOT_ID)
    st.session_state.history = []
    st.session_state.gemini_contents = []


def _display_history() -> List[Dict[str, str]]:
    if "history" not in st.session_state:
        st.session_state.history = []
    history_value = st.session_state.history
    if not isinstance(history_value, list):
        st.session_state.history = []
    return st.session_state.history


def _gemini_contents() -> List[Dict[str, Any]]:
    if "gemini_contents" not in st.session_state:
        st.session_state.gemini_contents = []
    contents_value = st.session_state.gemini_contents
    if not isinstance(contents_value, list):
        st.session_state.gemini_contents = []
    return st.session_state.gemini_contents


def _gemini_client() -> GeminiClient:
    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = create_gemini_client_from_env()
    client_value = st.session_state.gemini_client
    if not isinstance(client_value, GeminiClient):
        st.session_state.gemini_client = create_gemini_client_from_env()
    return st.session_state.gemini_client


def _decode_b64(data_b64: str) -> Optional[bytes]:
    try:
        return base64.b64decode(str(data_b64), validate=True)
    except Exception:
        return None


def render_chat(timeout_s: float = 60.0) -> None:
    _reset_state_on_boot_change()

    col_chat, col_tools = st.columns([3, 1])

    with col_tools:
        tools_placeholder = st.empty()

    def render_tools(events: List[Dict[str, Any]]) -> None:
        lines: List[str] = []
        for event in events:
            tool = str(event.get("tool") or "")
            status = str(event.get("status") or "")
            result = str(event.get("result") or "")
            if not tool:
                continue
            if status == "error" and result:
                lines.append("- " + tool + " (" + status + "): " + result)
            else:
                lines.append("- " + tool + " (" + status + ")")
        tools_placeholder.markdown("\n".join(lines) if lines else "No tools yet.")

    with col_chat:
        if st.button("Clear history", use_container_width=False):
            st.session_state.history = []
            st.session_state.gemini_contents = []
            st.rerun()

        history = _display_history()
        for turn in history:
            role = str(turn.get("role", "user"))
            content = str(turn.get("message", ""))
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)

        query = st.chat_input("Ask something")
        if not query:
            render_tools([])
            return

        gemini_client = _gemini_client()
        if not str(gemini_client.api_key).strip():
            st.warning("Missing GEMINI_API_KEY in environment.")
            render_tools([])
            return

        ok_img, err_img, image_b64, image_mime = fetch_screenshot()
        screenshot_bytes: Optional[bytes] = None
        if not ok_img:
            st.warning(err_img)
            image_b64 = None
            image_mime = None
        else:
            screenshot_bytes = (
                _decode_b64(str(image_b64)) if image_b64 is not None else None
            )

        history.append({"role": "user", "message": str(query)})
        with st.chat_message("user"):
            st.markdown(str(query))
            if screenshot_bytes is not None and image_mime is not None:
                st.image(screenshot_bytes, caption=str(image_mime), width=260)

        contents = _gemini_contents()
        contents.append({"role": "user", "parts": [{"text": str(query)}]})

        live_events: List[Dict[str, Any]] = []
        render_tools(live_events)

        def tool_executor_live(
            name: str, args: Dict[str, Any], timeout_s_value: float
        ) -> Tuple[bool, str]:
            event: Dict[str, Any] = {
                "tool": str(name),
                "status": "running",
                "result": "",
            }
            live_events.append(event)
            render_tools(live_events)
            ok, msg = execute_tool_call(name, args, timeout_s_value)
            event["status"] = "ok" if ok else "error"
            event["result"] = str(msg) if not ok else ""
            render_tools(live_events)
            return ok, str(msg)

        with st.chat_message("assistant"):
            with st.expander("Prompt context (Gemini)", expanded=False):
                include_image = st.checkbox(
                    "Include image (base64) in preview", value=False
                )

                preview_contents = build_request_contents(
                    base_contents=contents,
                    image_b64=image_b64 if include_image else None,
                    image_mime_type=image_mime if include_image else None,
                )
                payload = build_generate_content_payload(
                    contents=preview_contents,
                    system_prompt=PLANNER_PROMPT,
                    allow_tools=False,
                )

                st.subheader("System prompt")
                st.code(str(PLANNER_PROMPT), language="text")

                st.subheader("Messages (contents)")
                options: List[str] = []
                for i, msg in enumerate(preview_contents):
                    role = str(msg.get("role") or "")
                    parts_value = msg.get("parts")
                    parts = parts_value if isinstance(parts_value, list) else []
                    has_inline = any(
                        isinstance(p, dict)
                        and ("inline_data" in p or "inlineData" in p)
                        for p in parts
                    )
                    options.append(
                        str(i) + " | " + role + (" | image" if has_inline else "")
                    )

                selected = st.selectbox("Message preview", options=options)
                try:
                    selected_index = int(str(selected).split("|", 1)[0].strip())
                except Exception:
                    selected_index = 0

                if 0 <= selected_index < len(preview_contents):
                    st.json(preview_contents[selected_index])

                st.subheader("Full payload (generateContent)")
                st.code(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    language="json",
                )

            with st.spinner("Planning..."):
                ok, comment, sequence, updated_contents = (
                    gemini_client.plan_sequence_json(
                        system_prompt=PLANNER_PROMPT,
                        base_contents=contents,
                        image_b64=image_b64,
                        image_mime_type=image_mime,
                        timeout_s=float(timeout_s),
                        max_retries=2,
                    )
                )

                st.session_state.gemini_contents = updated_contents

                if not ok:
                    st.error(str(comment))
                    history.append({"role": "assistant", "message": str(comment)})
                    return

                st.markdown(str(comment))
                history.append({"role": "assistant", "message": str(comment)})

            if not sequence:
                return

            with st.spinner("Executing..."):
                for tool_name in sequence:
                    ok_exec, msg_exec = tool_executor_live(
                        str(tool_name), {}, float(timeout_s)
                    )
                    if not ok_exec:
                        st.error(str(msg_exec))
                        return
