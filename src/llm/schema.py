from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field


ToolName = Literal[
    "move_forward",
    "move_backward",
    "rotate_right_90",
    "rotate_left_90",
    "rotate_back",
    "grab_right",
    "release_right",
]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment: str = Field(default="")
    sequence: List[ToolName] = Field(default_factory=list)
