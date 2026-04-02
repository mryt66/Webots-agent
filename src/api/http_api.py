import queue
import threading

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ApiCommand:
    name: str
    params: Dict[str, Any]
    done: threading.Event
    response: Dict[str, Any]


class CommandDispatcher:
    def __init__(self) -> None:
        self._queue: queue.Queue[ApiCommand] = queue.Queue()

    def submit_and_wait(
        self, name: str, params: Dict[str, Any], timeout_s: float
    ) -> Dict[str, Any]:
        command = ApiCommand(
            name=name, params=params, done=threading.Event(), response={}
        )
        self._queue.put(command)
        finished = command.done.wait(timeout_s)
        if not finished:
            return {"ok": False, "error": "timeout"}
        return command.response

    def get_nowait(self) -> Optional[ApiCommand]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def start_api_server_in_thread(
    dispatcher: CommandDispatcher, host: str, port: int
) -> threading.Thread:
    try:
        from fastapi import FastAPI
        from fastapi import HTTPException
        import uvicorn
    except ModuleNotFoundError as exc:
        name = getattr(exc, "name", None)
        missing = str(name) if name else str(exc)
        raise RuntimeError(
            "Missing dependency: "
            + missing
            + ". Install it in the Python environment used by the Webots controller."
        ) from exc

    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    def dispatch(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response = dispatcher.submit_and_wait(name=name, params=params, timeout_s=120.0)
        if not bool(response.get("ok", False)):
            raise HTTPException(status_code=500, detail=response)
        return response

    @app.post("/move/forward")
    def move_forward(distance: float = 0.25) -> Dict[str, Any]:
        return dispatch("move_forward", {"distance": float(distance)})

    @app.post("/move/backward")
    def move_backward(distance: float = 0.25) -> Dict[str, Any]:
        return dispatch("move_backward", {"distance": float(distance)})

    @app.post("/rotate/right-90")
    def rotate_right_90() -> Dict[str, Any]:
        return dispatch("rotate_right_90", {})

    @app.post("/rotate/left-90")
    def rotate_left_90() -> Dict[str, Any]:
        return dispatch("rotate_left_90", {})

    @app.post("/rotate/back")
    def rotate_back() -> Dict[str, Any]:
        return dispatch("rotate_back", {})

    @app.post("/gripper/right/grab")
    def grab_right(torque_when_gripping: float = 30.0) -> Dict[str, Any]:
        return dispatch(
            "grab_right", {"torque_when_gripping": float(torque_when_gripping)}
        )

    @app.post("/gripper/right/release")
    def release_right() -> Dict[str, Any]:
        return dispatch("release_right", {})

    @app.post("/base/go-home")
    def go_home() -> Dict[str, Any]:
        return dispatch("go_home", {})

    @app.post("/camera/high-def")
    def camera_high_def() -> Dict[str, Any]:
        return dispatcher.submit_and_wait(
            name="capture_high_def",
            params={},
            timeout_s=120.0,
        )

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def run_server() -> None:
        try:
            server.run()
        except BaseException as exc:
            _ = exc

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread
