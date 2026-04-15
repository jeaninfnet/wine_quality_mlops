from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ExpectationResolver(ABC):
    @abstractmethod
    def resolve(self, type_name: str) -> type:
        ...


class QualityValidator(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @abstractmethod
    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        ...


class QualityReportWriterBase(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @abstractmethod
    def write(self, summary: dict[str, Any], output_dir: Path) -> Path:
        ...
