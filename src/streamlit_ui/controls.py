from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


WEBOTS_TIMEOUT_S = 120.0
DEFAULT_DISTANCE = 0.25
DEFAULT_TORQUE_WHEN_GRIPPING = 30.0


def api_post(
    url: str, params: Dict[str, Any], timeout_s: float
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    try:
        response = requests.post(url, params=params, timeout=timeout_s)
        text = str(response.text)
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.ok:
            return True, text, data if isinstance(data, dict) else None
        return False, text, data if isinstance(data, dict) else None
    except requests.RequestException as exc:
        return False, str(exc), None


def fire(url: str, params: Dict[str, Any], timeout_s: float) -> None:
    ok, _text, _data = api_post(url, params, timeout_s)
    if not ok:
        st.error("ERROR")


def render_controls(base: str) -> None:
    st.divider()
    st.header("Controls")

    actions: List[Tuple[str, str]] = [
        ("Move forward", "move_forward"),
        ("Move backward", "move_backward"),
        ("Rotate right 90", "rotate_right_90"),
        ("Rotate left 90", "rotate_left_90"),
        ("Rotate back", "rotate_back"),
        ("Grab right", "grab_right"),
        ("Release right", "release_right"),
    ]
    while len(actions) < 16:
        actions.append(("", ""))

    timeout_value = float(WEBOTS_TIMEOUT_S)

    for row_index in range(4):
        cols = st.columns(4)
        for col_index in range(4):
            action_index = row_index * 4 + col_index
            label, action_id = actions[action_index]
            with cols[col_index]:
                if not label:
                    st.write("")
                    continue

                pressed = st.button(label, key=f"action_{action_index}")
                if not pressed:
                    continue

                if action_id == "move_forward":
                    fire(
                        f"{base}/move/forward",
                        {"distance": float(DEFAULT_DISTANCE)},
                        timeout_value,
                    )
                    continue

                if action_id == "move_backward":
                    fire(
                        f"{base}/move/backward",
                        {"distance": float(DEFAULT_DISTANCE)},
                        timeout_value,
                    )
                    continue

                if action_id == "rotate_right_90":
                    fire(f"{base}/rotate/right-90", {}, timeout_value)
                    continue

                if action_id == "rotate_left_90":
                    fire(f"{base}/rotate/left-90", {}, timeout_value)
                    continue

                if action_id == "rotate_back":
                    fire(f"{base}/rotate/back", {}, timeout_value)
                    continue

                if action_id == "grab_right":
                    fire(
                        f"{base}/gripper/right/grab",
                        {"torque_when_gripping": float(DEFAULT_TORQUE_WHEN_GRIPPING)},
                        timeout_value,
                    )
                    continue

                if action_id == "release_right":
                    fire(f"{base}/gripper/right/release", {}, timeout_value)
                    continue
