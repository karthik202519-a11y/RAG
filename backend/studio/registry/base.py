"""Method registry: machine-readable description of every selectable method.

The frontend builds its selectors and parameter forms from this metadata, so a
control exists only if a method is registered. `status` is the honesty field:

  implemented            runs for real in this backend
  requires_credentials   implemented, needs a configured provider to run
  unavailable            described for learning, NOT executable; `unavailable_reason` says why
"""
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["implemented", "requires_credentials", "unavailable"]


class ParamSpec(BaseModel):
    name: str
    label: str
    type: Literal["int", "float", "bool", "select", "text"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    unit: str | None = None
    help: str = ""
    # Show this parameter only when another parameter has one of these values.
    visible_when: dict[str, list[Any]] | None = None


class MethodSpec(BaseModel):
    id: str
    category: str            # e.g. "chunking", "parser"
    stage: str               # pipeline stage
    name: str
    summary: str             # beginner-friendly
    how_it_works: str        # technical
    advantages: list[str]
    disadvantages: list[str]
    suitable_documents: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    params: list[ParamSpec] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)       # models / external dependencies
    performance_notes: str = ""
    cost_notes: str = ""
    needs_cloud_credentials: bool = False
    status: Status = "implemented"
    unavailable_reason: str | None = None
    references: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    compare_with: list[str] = Field(default_factory=list)   # ids of alternatives
    recommended_experiment: str = ""


class Registry:
    def __init__(self) -> None:
        self._methods: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec, replace: bool = False) -> MethodSpec:
        if spec.id in self._methods and not replace:
            raise ValueError(f"duplicate method id {spec.id}")
        if spec.status == "unavailable" and not spec.unavailable_reason:
            raise ValueError(f"{spec.id}: unavailable methods must say why")
        self._methods[spec.id] = spec
        return spec

    def get(self, method_id: str) -> MethodSpec | None:
        return self._methods.get(method_id)

    def list(self, category: str | None = None) -> list[MethodSpec]:
        return [m for m in self._methods.values() if category is None or m.category == category]

    def categories(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self._methods.values():
            out[m.category] = out.get(m.category, 0) + 1
        return out


registry = Registry()
