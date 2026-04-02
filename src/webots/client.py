import os
import base64
import requests
from typing import Any, Dict, Optional, Tuple


WEBOTS_API_BASE = "http://127.0.0.1:8000"
WEBOTS_SCREENSHOT_TIMEOUT_S = 10.0

DEFAULT_DISTANCE_M = 0.25
DEFAULT_TORQUE_WHEN_GRIPPING = 30.0


def webots_post(
    path: str, params: Dict[str, Any], timeout_s: float
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    base = str(os.environ.get("WEBOTS_API_BASE") or WEBOTS_API_BASE).rstrip("/")
    url = base + str(path)
    try:
        response = requests.post(url, params=params, timeout=timeout_s)
    except requests.RequestException:
        return False, "webots_api_unreachable", None

    text = str(response.text)
    try:
        data = response.json()
    except ValueError:
        data = None

    if response.ok:
        return True, text if text else "ok", data if isinstance(data, dict) else None
    return False, text, data if isinstance(data, dict) else None


def fetch_screenshot() -> Tuple[bool, str, Optional[str], Optional[str]]:
    url = str(os.environ.get("WEBOTS_API_BASE") or WEBOTS_API_BASE).rstrip("/")
    target = url + "/camera/high-def"

    try:
        response = requests.post(target, timeout=float(WEBOTS_SCREENSHOT_TIMEOUT_S))
    except requests.RequestException:
        return (
            False,
            f"Cannot reach Webots API ({url}). Start the Webots simulation and the controller (port 8000).",
            None,
            None,
        )

    try:
        data = response.json()
    except ValueError:
        return False, str(response.text), None, None

    if isinstance(data, dict) and data.get("ok") is False:
        error = data.get("error")
        error_type = data.get("error_type")
        command = data.get("command")
        message = str(error) if error is not None else "screenshot failed"
        if error_type:
            message = str(error_type) + ": " + message
        if command:
            message = str(command) + " -> " + message
        return False, message, None, None

    if not isinstance(data, dict):
        return False, "invalid response", None, None

    image_b64 = data.get("image_b64")
    mime_type = data.get("mime_type")
    if not image_b64 or not mime_type:
        error = data.get("error")
        return False, str(error) if error else "missing image_b64/mime_type", None, None

    try:
        base64.b64decode(str(image_b64), validate=True)
    except Exception:
        return False, "invalid image_b64", None, None

    return True, "ok", str(image_b64), str(mime_type)


def execute_tool_call(
    name: str, args: Dict[str, Any], timeout_s: float
) -> Tuple[bool, str]:
    _ = args

    if name == "move_forward":
        ok, msg, _data = webots_post(
            "/move/forward", {"distance": float(DEFAULT_DISTANCE_M)}, timeout_s
        )
        return ok, msg

    if name == "move_backward":
        ok, msg, _data = webots_post(
            "/move/backward",
            {"distance": float(DEFAULT_DISTANCE_M)},
            timeout_s,
        )
        return ok, msg

    if name == "rotate_right_90":
        ok, msg, _data = webots_post("/rotate/right-90", {}, timeout_s)
        return ok, msg

    if name == "rotate_left_90":
        ok, msg, _data = webots_post("/rotate/left-90", {}, timeout_s)
        return ok, msg

    if name == "rotate_back":
        ok, msg, _data = webots_post("/rotate/back", {}, timeout_s)
        return ok, msg

    if name == "grab_right":
        ok, msg, _data = webots_post(
            "/gripper/right/grab",
            {"torque_when_gripping": float(DEFAULT_TORQUE_WHEN_GRIPPING)},
            timeout_s,
        )
        return ok, msg

    if name == "release_right":
        ok, msg, _data = webots_post("/gripper/right/release", {}, timeout_s)
        return ok, msg

    return False, "unknown_tool"
