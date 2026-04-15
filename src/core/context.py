from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import logging


class PipelineContext:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.config_dir = self.root_dir / "config"
        self.secrets_path = self.root_dir / "secrets.env"

        self._garantir_sys_path()

        from src.utils.config_loader import load_yaml
        from src.utils.logger import get_logger

        self.data_cfg: dict[str, Any] = load_yaml(self.config_dir / "data.yaml")
        self.pipeline_cfg: dict[str, Any] = load_yaml(self.config_dir / "pipeline.yaml")
        self.quality_cfg: dict[str, Any] = load_yaml(self.config_dir / "quality.yaml")

        self.logger: logging.Logger = get_logger(
            name="pipeline",
            logging_config=self.pipeline_cfg["logging"],
        )

        caminhos = self.pipeline_cfg["paths"]
        self.raw_dir = self.root_dir / caminhos["raw_data_dir"]
        self.processed_dir = self.root_dir / caminhos["processed_data_dir"]
        self.output_path = self.processed_dir / caminhos["output_filename"]
        self.features_dir = self.root_dir / caminhos["features_data_dir"]
        self.features_path = self.features_dir / caminhos["features_filename"]

    @classmethod
    def from_notebook(cls, notebook_path: str | Path) -> "PipelineContext":
        root = Path(notebook_path).resolve().parent.parent
        return cls(root)

    def _garantir_sys_path(self) -> None:
        for p in (str(self.root_dir), str(self.config_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)

    @property
    def kaggle_dataset(self) -> str:
        return self.data_cfg["kaggle"]["dataset"]

    @property
    def kaggle_file_pattern(self) -> str:
        return self.data_cfg["kaggle"].get("file_pattern", "*.csv")

    @property
    def kaggle_expected_files(self) -> list[str]:
        return self.data_cfg["kaggle"].get("expected_files") or []

    @property
    def ingest_compression(self) -> str:
        return self.data_cfg["ingest"].get("compression", "snappy")

    @property
    def ingest_validate_schema(self) -> bool:
        return self.data_cfg["ingest"].get("validate_schema", True)

    @property
    def required_columns(self) -> list[str]:
        return self.data_cfg["schema"].get("required_columns", [])

    @property
    def skip_download(self) -> bool:
        return self.pipeline_cfg["execution"].get("skip_download_if_exists", True)

    @property
    def force_download(self) -> bool:
        return self.pipeline_cfg["execution"].get("force_redownload", False)

    @property
    def skip_ingest(self) -> bool:
        return self.pipeline_cfg["execution"].get("skip_ingest_if_exists", True)

    @property
    def force_ingest(self) -> bool:
        return self.pipeline_cfg["execution"].get("force_ingest", False)

    def run_step(self, etapa: str) -> None:
        if etapa == "ingestion":
            self._executar_ingestao()
        elif etapa == "quality":
            self._executar_qualidade()
        elif etapa == "preprocessing":
            self._executar_preprocessamento()
        else:
            raise ValueError(
                f"Etapa desconhecida: '{etapa}'. Use: ingestion, quality, preprocessing"
            )

    def _executar_ingestao(self) -> None:
        from src.ingestion.downloader import KaggleDownloader
        from src.ingestion.wine_csv_ingester import WineCsvToParquetIngester

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        downloader = KaggleDownloader(
            secrets_path=self.secrets_path,
            dataset=self.kaggle_dataset,
            file_pattern=self.kaggle_file_pattern,
            expected_files=self.kaggle_expected_files,
            skip_if_exists=self.skip_download,
            force=self.force_download,
            logger=self.logger,
        )
        downloader.load(destination_dir=self.raw_dir)

        ingester = WineCsvToParquetIngester(
            raw_dir=self.raw_dir,
            output_path=self.output_path,
            compression=self.ingest_compression,
            validate_schema=self.ingest_validate_schema,
            required_columns=self.required_columns,
            skip_if_exists=self.skip_ingest,
            force=self.force_ingest,
            logger=self.logger,
        )
        ingester.run()

    def _executar_qualidade(self) -> None:
        import pandas as pd

        import great_expectations as gx
        import great_expectations.expectations as gxe

        from src.quality.expectation_resolver import GeExpectationResolver
        from src.quality.ge_validator import GreatExpectationsValidator
        from src.quality.report_writer import QualityReportWriter

        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Parquet não encontrado: {self.output_path}. Rode run_step('ingestion') antes."
            )

        df = pd.read_parquet(self.output_path)
        self.logger.info("Qualidade: %d linhas × %d colunas", *df.shape)

        config_unificado = {**self.pipeline_cfg, **self.quality_cfg}
        resolver = GeExpectationResolver(gxe)
        validator = GreatExpectationsValidator(resolver, self.logger, gx)
        summary = validator.validate(df, config_unificado)

        quality_cfg = self.quality_cfg.get("quality", {})
        output_dir = self.root_dir / quality_cfg.get("output_dir", "outputs/quality")
        writer = QualityReportWriter(self.logger)
        report_path = writer.write(summary, output_dir)

        self.logger.info("Relatório de qualidade: %s", report_path)

    def _executar_preprocessamento(self) -> None:
        from src.wine_pipeline.build_features import build_white_wine_opinion_dataset

        if not self.output_path.exists():
            raise FileNotFoundError(f"Parquet bruto ausente: {self.output_path}")
        build_white_wine_opinion_dataset(
            raw_parquet_path=self.output_path,
            features_parquet_path=self.features_path,
            logger=self.logger,
        )
