import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from llm.gemini_client import (
    GEMINI_DEFAULT_MODEL,
    ToolExecutor,
    plan_sequence_json,
    run_tool_loop,
)


@dataclass(frozen=True)
class GeminiClient:
    api_key: str
    model: str

    def run_tool_loop(
        self,
        system_prompt: str,
        base_contents: List[Dict[str, Any]],
        image_b64: Optional[str],
        image_mime_type: Optional[str],
        tool_executor: ToolExecutor,
        timeout_s: float,
        max_steps: int,
        caller_path: Optional[str],
    ) -> Tuple[bool, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        return run_tool_loop(
            api_key=self.api_key,
            model=self.model,
            system_prompt=system_prompt,
            base_contents=base_contents,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            tool_executor=tool_executor,
            timeout_s=timeout_s,
            max_steps=max_steps,
            caller_path=caller_path,
        )

    def plan_sequence_json(
        self,
        system_prompt: str,
        base_contents: List[Dict[str, Any]],
        image_b64: Optional[str],
        image_mime_type: Optional[str],
        timeout_s: float,
        max_retries: int = 2,
    ) -> Tuple[bool, str, List[str], List[Dict[str, Any]]]:
        return plan_sequence_json(
            api_key=self.api_key,
            model=self.model,
            system_prompt=system_prompt,
            base_contents=base_contents,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            timeout_s=timeout_s,
            max_retries=int(max_retries),
        )


def create_gemini_client_from_env() -> GeminiClient:
    api_key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
    model = str(os.environ.get("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL).strip()
    if not model:
        model = GEMINI_DEFAULT_MODEL
    return GeminiClient(api_key=api_key, model=model)
