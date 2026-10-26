"""Pluggable backend modules for motion-coder (via run_module)."""

from .registry import expand_semantic_modules, load_registry, resolve_module

__all__ = ["load_registry", "resolve_module", "expand_semantic_modules"]
