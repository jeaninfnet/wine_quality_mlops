from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.quality.base import QualityValidator


class GreatExpectationsValidator(QualityValidator):
    def __init__(self, resolver, logger: logging.Logger, gx) -> None:
        super().__init__(logger)
        self._resolver = resolver
        self._gx = gx

    def validate(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        quality_cfg = config.get("quality", {})
        suite_name = quality_cfg.get("suite_name", "default_suite")
        fail_on_error = quality_cfg.get("fail_pipeline_on_error", True)
        table_expectations = config.get("table_expectations", [])
        column_expectations = config.get("column_expectations", {})

        total_definido = len(table_expectations) + sum(
            len(v) for v in column_expectations.values()
        )
        self._logger.info("Checks de qualidade definidos: %d", total_definido)

        context, batch_def, suite = self._construir_contexto_efemero(df, suite_name)
        self._popular_suite(suite, table_expectations, column_expectations)

        validation_def = context.validation_definitions.add(
            self._gx.ValidationDefinition(
                name=f"{suite_name}_validation",
                data=batch_def,
                suite=suite,
            )
        )
        results = validation_def.run(batch_parameters={"dataframe": df})
        total = len(results.results)
        passed = sum(1 for r in results.results if r.success)
        failed = total - passed

        if fail_on_error and not results.success:
            raise RuntimeError(f"Qualidade reprovada: {failed}/{total} falhas.")

        return {
            "success": results.success,
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results,
        }

    def _construir_contexto_efemero(self, df: pd.DataFrame, suite_name: str):
        context = self._gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas("pipeline_source")
        asset = data_source.add_dataframe_asset(name="input_data")
        batch_def = asset.add_batch_definition_whole_dataframe("full_batch")
        suite = context.suites.add(self._gx.ExpectationSuite(name=suite_name))
        return context, batch_def, suite

    def _popular_suite(self, suite, table_expectations, column_expectations) -> None:
        for exp in table_expectations or []:
            cls = self._resolver.resolve(exp["type"])
            kwargs = exp.get("kwargs", {})
            suite.add_expectation(cls(**kwargs))
        for coluna, exps in (column_expectations or {}).items():
            for exp in exps:
                cls = self._resolver.resolve(exp["type"])
                kwargs = exp.get("kwargs", {})
                suite.add_expectation(cls(column=coluna, **kwargs))
