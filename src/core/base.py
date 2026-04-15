from abc import ABC, abstractmethod
from pathlib import Path
import logging


class PipelineStep(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        ...


class DataLoader(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def load(self, destination_dir: Path) -> list[Path]:
        ...
