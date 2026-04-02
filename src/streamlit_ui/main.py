import os
import sys

from pathlib import Path


_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import streamlit as st

from config import config_path_from_env, load_env_yaml
from llm.client import create_gemini_client_from_env
from llm.gemini_client import GEMINI_DEFAULT_MODEL
from streamlit_ui.chat import render_chat
from streamlit_ui.controls import render_controls


WEBOTS_API_BASE = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(page_title="PR2 API Controller", layout="centered")
    st.title("PR2 API Controller")

    config_path = config_path_from_env()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    _ok, _msg = load_env_yaml(config_path, override=False)

    if "WEBOTS_API_BASE" not in os.environ:
        os.environ["WEBOTS_API_BASE"] = WEBOTS_API_BASE

    if "GEMINI_MODEL" not in os.environ:
        os.environ["GEMINI_MODEL"] = GEMINI_DEFAULT_MODEL

    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = create_gemini_client_from_env()

    base = str(os.environ.get("WEBOTS_API_BASE") or WEBOTS_API_BASE).rstrip("/")
    render_controls(base)
    render_chat()


if __name__ == "__main__":
    main()
