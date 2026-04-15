"""
Conversão CSV → Parquet com detecção de separador e normalização de nomes de colunas
(espacos → snake_case), compatível com winequalityN.csv (Kaggle).
"""
from __future__ import annotations

import time
from pathlib import Path
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.base import PipelineStep


def _normalizar_nomes(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        x = c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        out.append(x)
    return out


class WineCsvToParquetIngester(PipelineStep):
    def __init__(
        self,
        raw_dir: Path,
        output_path: Path,
        compression: str = "snappy",
        validate_schema: bool = True,
        required_columns: list[str] | None = None,
        skip_if_exists: bool = True,
        force: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger or logging.getLogger(__name__))
        self.raw_dir = Path(raw_dir)
        self.output_path = Path(output_path)
        self.compression = compression
        self.validate_schema = validate_schema
        self.required_columns = required_columns or []
        self.skip_if_exists = skip_if_exists
        self.force = force

    def run(self) -> Path:
        if (
            not self.force
            and self.skip_if_exists
            and self.output_path.exists()
            and self.output_path.stat().st_size > 0
        ):
            self.logger.info("Parquet já existe; ingestão ignorada: %s", self.output_path)
            return self.output_path

        csvs = sorted(self.raw_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"Nenhum CSV em {self.raw_dir}")

        path_csv = csvs[0]
        self.logger.info("Lendo '%s'...", path_csv.name)
        t0 = time.monotonic()

        df = None
        for sep in (";", ",", "\t"):
            try:
                trial = pd.read_csv(path_csv, sep=sep, engine="python")
                if trial.shape[1] >= 10:
                    df = trial
                    self.logger.info("Separador detectado: %r", sep)
                    break
            except Exception:
                continue
        if df is None:
            df = pd.read_csv(path_csv)

        df.columns = _normalizar_nomes(list(df.columns))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(self.output_path), compression=self.compression)

        if self.validate_schema and self.required_columns:
            missing = [c for c in self.required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Colunas obrigatórias ausentes: {missing}. Encontradas: {list(df.columns)}")

        self.logger.info(
            "Parquet gravado em %.1fs — %d linhas, %d colunas → %s",
            time.monotonic() - t0,
            len(df),
            df.shape[1],
            self.output_path,
        )
        return self.output_path
