import base64
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import requests
from pydantic import ValidationError
from llm.prompts import tool_declarations
from llm.schema import Plan


GEMINI_DEFAULT_MODEL = "gemini-3-flash-preview"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"


ToolExecutor = Callable[[str, Dict[str, Any], float], Tuple[bool, str]]


def build_request_contents(
    base_contents: List[Dict[str, Any]],
    image_b64: Optional[str],
    image_mime_type: Optional[str],
) -> List[Dict[str, Any]]:
    contents = list(base_contents)
    return _with_inline_image_on_last_user_turn(contents, image_b64, image_mime_type)


def build_generate_content_payload(
    contents: List[Dict[str, Any]],
    system_prompt: str,
    allow_tools: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contents": contents,
        "system_instruction": {"parts": [{"text": str(system_prompt)}]},
    }

    if not allow_tools:
        return payload

    payload["tools"] = [{"function_declarations": tool_declarations()}]
    payload["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}
    return payload


def tool_names() -> List[str]:
    names: List[str] = []
    for decl in tool_declarations():
        if not isinstance(decl, dict):
            continue
        name = decl.get("name")
        if not name:
            continue
        names.append(str(name))
    return names


ALLOWED_TOOL_NAMES: Set[str] = set(tool_names())


def _extract_text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return ""
    texts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text is None:
            continue
        texts.append(str(text))
    return "\n".join([t for t in texts if t.strip()]).strip()


def _extract_function_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return []
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return []

    calls: List[Dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        fn = part.get("functionCall")
        if isinstance(fn, dict):
            calls.append(fn)
            continue
        fn2 = part.get("function_call")
        if isinstance(fn2, dict):
            calls.append(fn2)
            continue
    return calls


def _post_payload(
    url: str, payload: Dict[str, Any], timeout_s: float
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        return False, str(exc), None

    if not response.ok:
        return False, str(response.text), None

    try:
        data = response.json()
    except ValueError:
        return False, str(response.text), None

    if not isinstance(data, dict):
        return False, "invalid_response", None

    return True, "ok", data


def _call_gemini(
    api_key: str,
    model: str,
    contents: List[Dict[str, Any]],
    system_prompt: str,
    timeout_s: float,
    allow_tools: bool,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    url = (
        str(GEMINI_API_BASE).rstrip("/")
        + "/v1beta/models/"
        + str(model)
        + ":generateContent?key="
        + str(api_key)
    )

    payload: Dict[str, Any] = {
        "contents": contents,
        "system_instruction": {"parts": [{"text": str(system_prompt)}]},
    }

    if not allow_tools:
        return _post_payload(url, payload, timeout_s)

    payload_snake = dict(payload)
    payload_snake["tools"] = [{"function_declarations": tool_declarations()}]
    payload_snake["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

    ok, msg, data = _post_payload(url, payload_snake, timeout_s)
    if ok:
        return ok, msg, data

    payload_camel = dict(payload)
    payload_camel["tools"] = [{"functionDeclarations": tool_declarations()}]
    payload_camel["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return _post_payload(url, payload_camel, timeout_s)


def _with_inline_image_on_last_user_turn(
    contents: List[Dict[str, Any]],
    image_b64: Optional[str],
    image_mime_type: Optional[str],
) -> List[Dict[str, Any]]:
    if not image_b64 or not image_mime_type:
        return contents

    try:
        base64.b64decode(str(image_b64), validate=True)
    except Exception:
        return contents

    last_user_index: Optional[int] = None
    for i in range(len(contents) - 1, -1, -1):
        role = contents[i].get("role")
        if role != "user":
            continue
        parts_value = contents[i].get("parts")
        parts = parts_value if isinstance(parts_value, list) else []
        has_text = any(
            isinstance(p, dict) and str(p.get("text") or "").strip() for p in parts
        )
        if has_text:
            last_user_index = i
            break

    if last_user_index is None:
        return contents

    updated = [dict(c) for c in contents]
    last = dict(updated[last_user_index])
    parts_value = last.get("parts")
    parts = parts_value if isinstance(parts_value, list) else []

    new_parts: List[Dict[str, Any]] = [
        {
            "inline_data": {
                "mime_type": str(image_mime_type),
                "data": str(image_b64),
            }
        }
    ]

    for part in parts:
        if isinstance(part, dict) and "text" in part:
            new_parts.append({"text": str(part.get("text") or "")})

    if len(new_parts) == 1:
        return contents

    last["parts"] = new_parts
    updated[last_user_index] = last
    return updated


def _strip_inline_data(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for content in contents:
        if not isinstance(content, dict):
            continue
        role = content.get("role")
        parts_value = content.get("parts")
        parts = parts_value if isinstance(parts_value, list) else []
        new_parts: List[Dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "inline_data" in part or "inlineData" in part:
                continue
            new_parts.append(part)
        cleaned.append({"role": role, "parts": new_parts})
    return cleaned


def run_tool_loop(
    api_key: str,
    model: str,
    system_prompt: str,
    base_contents: List[Dict[str, Any]],
    image_b64: Optional[str],
    image_mime_type: Optional[str],
    tool_executor: ToolExecutor,
    timeout_s: float,
    max_steps: int,
    caller_path: Optional[str],
) -> Tuple[bool, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    contents = list(base_contents)
    tool_events: List[Dict[str, Any]] = []

    allowed_tools: Set[str] = {
        str(d.get("name") or "")
        for d in tool_declarations()
        if isinstance(d, dict) and str(d.get("name") or "").strip()
    }

    def record_tool_event(
        tool: str, args: Dict[str, Any], ok: bool, result: str
    ) -> None:
        tool_events.append(
            {
                "tool": tool,
                "args": dict(args),
                "ok": bool(ok),
                "result": str(result),
            }
        )

    def append_function_response(tool: str, ok: bool, result: str) -> None:
        response_payload: Dict[str, Any] = {
            "ok": bool(ok),
            "result": str(result),
        }
        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": tool,
                            "response": response_payload,
                        }
                    }
                ],
            }
        )

    def execute(tool: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        safe_args = dict(args)
        ok_exec, msg_exec = tool_executor(tool, safe_args, timeout_s)
        record_tool_event(tool, safe_args, ok_exec, msg_exec)
        append_function_response(tool, ok_exec, msg_exec)
        return ok_exec, str(msg_exec)

    caller = str(caller_path or "").strip()
    if caller:
        now = datetime.now(timezone.utc).isoformat()
        contents.append(
            {
                "role": "user",
                "parts": [
                    {"text": "Context: caller_path=" + caller + " utc=" + str(now)}
                ],
            }
        )

    for _step_index in range(max_steps):
        request_contents = _with_inline_image_on_last_user_turn(
            contents, image_b64, image_mime_type
        )

        ok, err, data = _call_gemini(
            api_key,
            model,
            request_contents,
            system_prompt,
            timeout_s,
            allow_tools=True,
        )
        if not ok or data is None:
            return False, str(err), _strip_inline_data(contents), tool_events

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return False, "no_candidates", _strip_inline_data(contents), tool_events

        content = candidates[0].get("content")
        if not isinstance(content, dict):
            return False, "no_content", _strip_inline_data(contents), tool_events

        parts_value = content.get("parts")
        parts = parts_value if isinstance(parts_value, list) else []

        function_calls = _extract_function_calls(data)
        if not function_calls:
            text = _extract_text(data)
            if not text:
                text = "ok"

            contents.append({"role": "model", "parts": [{"text": text}]})
            return True, text, _strip_inline_data(contents), tool_events

        contents.append({"role": "model", "parts": parts})

        for fn in function_calls:
            name = str(fn.get("name") or "")
            if name not in allowed_tools:
                record_tool_event(name, {}, False, "unknown_tool")
                append_function_response(name, False, "unknown_tool")
                return False, "unknown_tool", _strip_inline_data(contents), tool_events

            args_value = fn.get("args")
            if args_value is None:
                args_value = fn.get("arguments")
            args = args_value if isinstance(args_value, dict) else {}

            ok_exec, msg_exec = execute(name, args)
            if not ok_exec:
                return False, msg_exec, _strip_inline_data(contents), tool_events

    return False, "tool_loop_exhausted", _strip_inline_data(contents), tool_events


def _extract_json_object(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start : end + 1]


def _validate_plan(data: Any) -> Tuple[bool, str, str, List[str]]:
    try:
        plan = Plan.model_validate(data)
    except ValidationError as exc:
        return False, str(exc), "", []

    return True, "ok", str(plan.comment), [str(t) for t in list(plan.sequence)]


def plan_sequence_json(
    api_key: str,
    model: str,
    system_prompt: str,
    base_contents: List[Dict[str, Any]],
    image_b64: Optional[str],
    image_mime_type: Optional[str],
    timeout_s: float,
    max_retries: int,
) -> Tuple[bool, str, List[str], List[Dict[str, Any]]]:
    contents = list(base_contents)
    last_error = "invalid_json"

    max_retries_value = int(max_retries)
    if max_retries_value < 0:
        max_retries_value = 0
    if max_retries_value > 2:
        max_retries_value = 2

    for _attempt in range(int(max_retries_value) + 1):
        request_contents = _with_inline_image_on_last_user_turn(
            contents, image_b64, image_mime_type
        )

        ok, err, data = _call_gemini(
            api_key,
            model,
            request_contents,
            system_prompt,
            timeout_s,
            allow_tools=False,
        )
        if not ok or data is None:
            return False, str(err), [], _strip_inline_data(contents)

        text = _extract_text(data)
        if not text:
            last_error = "empty_response"
            contents.append({"role": "user", "parts": [{"text": "Return valid JSON."}]})
            continue

        contents.append({"role": "model", "parts": [{"text": str(text)}]})

        json_text = _extract_json_object(text)
        if json_text is None:
            last_error = "no_json_object"
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Your response was not a JSON object. Return ONLY a JSON object with keys comment and sequence."
                        }
                    ],
                }
            )
            continue

        try:
            parsed = json.loads(str(json_text))
        except Exception:
            last_error = "json_parse_error"
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": 'Your JSON was invalid. Return ONLY valid JSON like {"comment":"...","sequence":["move_forward"]}.'
                        }
                    ],
                }
            )
            continue

        valid, reason, comment, sequence = _validate_plan(parsed)
        if not valid:
            last_error = str(reason)
            allowed = ", ".join(sorted(ALLOWED_TOOL_NAMES))
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Schema validation failed ("
                            + str(reason)
                            + "). sequence must be a list of tool names from: "
                            + allowed
                            + ". Return ONLY the JSON object."
                        }
                    ],
                }
            )
            continue

        return True, comment, sequence, _strip_inline_data(contents)

    return False, str(last_error), [], _strip_inline_data(contents)
