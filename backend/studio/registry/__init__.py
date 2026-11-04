"""Method registry. Importing this package registers every method spec."""
from . import chunking_specs, parser_specs, retrieval_specs  # noqa: F401  (import for side effects)
from .base import MethodSpec, ParamSpec, Registry, registry

__all__ = ["MethodSpec", "ParamSpec", "Registry", "registry"]
