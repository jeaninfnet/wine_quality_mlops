from __future__ import annotations

import fnmatch
import os
import zipfile
from pathlib import Path
import logging

from dotenv import load_dotenv

from src.core.base import DataLoader


def _formatar_tamanho(tamanho_bytes: int) -> str:
    for unidade in ("B", "KB", "MB", "GB"):
        if tamanho_bytes < 1024:
            return f"{tamanho_bytes:.1f} {unidade}"
        tamanho_bytes //= 1024
    return f"{tamanho_bytes:.1f} TB"


def _extrair_zip(zip_path: Path, destination_dir: Path, logger: logging.Logger) -> None:
    logger.info("  [UNZIP] Extraindo '%s'...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_dir)
    zip_path.unlink()
    logger.info("  [UNZIP] Concluído.")


class KaggleDownloader(DataLoader):
    def __init__(
        self,
        secrets_path: Path,
        dataset: str,
        file_pattern: str = "*.csv",
        expected_files: list[str] | None = None,
        skip_if_exists: bool = True,
        force: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger or logging.getLogger(__name__))
        self.secrets_path = Path(secrets_path)
        self.dataset = dataset
        self.file_pattern = file_pattern
        self.expected_files: list[str] = expected_files or []
        self.skip_if_exists = skip_if_exists
        self.force = force
        self._validar_credenciais()

    def load(self, destination_dir: Path) -> list[Path]:
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        arquivos = self._resolver_arquivos_esperados()
        self.logger.info("Dataset  : %s", self.dataset)
        self.logger.info("Arquivos : %s", arquivos)

        if (
            not self.force
            and self.skip_if_exists
            and self._todos_presentes(destination_dir, arquivos)
        ):
            self.logger.info("Arquivos já presentes; download ignorado.")
            return [destination_dir / f for f in arquivos]

        api = self._autenticar()
        baixados: list[Path] = []
        for nome_arquivo in arquivos:
            dest_path = destination_dir / nome_arquivo
            if (
                not self.force
                and self.skip_if_exists
                and dest_path.exists()
                and dest_path.stat().st_size > 0
            ):
                baixados.append(dest_path)
                continue
            self.logger.info("  [DOWN] Baixando '%s'...", nome_arquivo)
            api.dataset_download_file(
                dataset=self.dataset,
                file_name=nome_arquivo,
                path=str(destination_dir),
                force=True,
                quiet=False,
            )
            zip_path = Path(str(dest_path) + ".zip")
            if zip_path.exists() and not dest_path.exists():
                _extrair_zip(zip_path, destination_dir, self.logger)
            if dest_path.exists():
                baixados.append(dest_path)
            else:
                raise RuntimeError(f"Arquivo esperado não encontrado: {dest_path}")
        return baixados

    def _validar_credenciais(self) -> None:
        load_dotenv(dotenv_path=str(self.secrets_path))
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            raise EnvironmentError(
                f"Defina KAGGLE_USERNAME e KAGGLE_KEY em {self.secrets_path}"
            )
        self.logger.info("Credenciais do Kaggle OK.")

    def _autenticar(self):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api

    def _resolver_arquivos_esperados(self) -> list[str]:
        if self.expected_files:
            return self.expected_files
        return self._listar_arquivos_remotos()

    def _listar_arquivos_remotos(self) -> list[str]:
        api = self._autenticar()
        arquivos = api.dataset_list_files(self.dataset).files
        return sorted(f.name for f in arquivos if fnmatch.fnmatch(f.name, self.file_pattern))

    @staticmethod
    def _todos_presentes(diretorio: Path, nomes: list[str]) -> bool:
        return all(
            (diretorio / f).exists() and (diretorio / f).stat().st_size > 0 for f in nomes
        )
