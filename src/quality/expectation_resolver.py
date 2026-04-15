from __future__ import annotations

from src.quality.base import ExpectationResolver


class GeExpectationResolver(ExpectationResolver):
    def __init__(self, gxe) -> None:
        self._gxe = gxe

    def resolve(self, type_name: str) -> type:
        pascal = "".join(w.capitalize() for w in type_name.split("_"))
        if hasattr(self._gxe, pascal):
            return getattr(self._gxe, pascal)
        if hasattr(self._gxe, type_name):
            return getattr(self._gxe, type_name)
        raise AttributeError(f"Expectation '{type_name}' não encontrada no GE.")
