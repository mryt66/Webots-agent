Demo: https://www.youtube.com/watch?v=uTdEIKzpsZk

## Webots Agent (PR2 Sandbox + GenAI Planner)

This project is a small robotics sandbox built on Webots for repeatable experiments with a PR2 robot. The simulation is deterministic (fixed world + fixed top-down observation), but each run introduces variability by randomizing the can position from a fixed set of valid table locations.

The GenAI component is optional: given a user request and the current screenshot, a Gemini-based planner produces a short, strictly-validated tool sequence. The UI can execute the sequence step-by-step through an HTTP bridge.

### Key idea (GenAI)

- **Vision → plan → tools**: the planner sees a top-down screenshot and returns a minimal JSON plan.
- **Closed tool set**: the model can only select from predefined tool names (no free-form actions).
- **Schema validation**: responses are validated before execution; invalid outputs are retried a small fixed number of times.

The tool sequence is constrained to motion primitives (step moves, fixed-angle rotations, gripper actions) and a screenshot primitive. This keeps execution predictable and reproducible.

### Project layout

- Webots controller entrypoint: `controllers/my_controller/my_controller.py`
- HTTP bridge + clients + UI + planner (Python):
	- `src/api/` (FastAPI server glue)
	- `src/webots/` (HTTP client used by the UI)
	- `src/streamlit_ui/` (operator console)
	- `src/llm/` (Gemini planner + JSON validation)
- Config: `conf/config.yaml` (local), `conf/config.example.yaml` (template)

### Configuration

Copy the template and fill your Gemini key:

1. Create `conf/config.yaml` from `conf/config.example.yaml`
2. Set:
	 - `GEMINI_API_KEY`
	 - optional `GEMINI_MODEL`
	 - optional `WEBOTS_API_BASE` (default `http://127.0.0.1:8000`)

You can also point to a different config file via `WEBOTS_AGENT_CONFIG`.

### Install

Dependencies are listed in `pyproject.toml` (no versions). Using uv:

```bash
uv sync
```

### Run

1. Start Webots and open `worlds/pr2.wbt` (controller: `my_controller`).
2. Run the operator UI:

```bash
streamlit run src/streamlit_ui/main.py
```

The UI will call the Webots HTTP endpoints (movement, gripper, screenshot) and can optionally ask Gemini to produce an action plan from the screenshot.

### Tools and endpoints

The **planner** is restricted to a closed set of tool names:

- `move_forward`, `move_backward`
- `rotate_right_90`, `rotate_left_90`, `rotate_back`
- `grab_right`, `release_right`

Screenshots are handled as a deterministic **perception endpoint** (`/camera/high-def`) and used as model input and operator feedback.

