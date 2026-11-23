"""BEIR dataset loader for NFCorpus and compatible collections."""

from __future__ import annotations

import gzip
import hashlib
import logging
import shutil
import tempfile
import zipfile
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


logger = logging.getLogger(__name__)
_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
# A alternativa usa duas revisões fixas do BEIR no Hugging Face.
# Uma fornece o corpus e as consultas, e a outra fornece os julgamentos de relevância.
_NFCORPUS_HF_URL = (
    "https://huggingface.co/datasets/BeIR/nfcorpus/resolve/"
    "e28763e68d85db0fa71652ba3c4afabf8c5b3bb7/"
)
_NFCORPUS_QRELS_HF_URL = (
    "https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/"
    "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65/"
)
# Estes SHA-256 identificam os arquivos compactados das revisões acima.
_NFCORPUS_TEXT_HASHES = {
    "corpus.jsonl.gz": "ae4d6ab2ab88d880e7e180776c956a09c19c43a703457742cd850d2422035802",
    "queries.jsonl.gz": "baba3504a2736b0e32265962f04acc4fecfaf9649e0ea08e49ff441246f75364",
}
_QRELS_FILES = tuple(Path("qrels") / f"{split}.tsv" for split in ("train", "dev", "test"))
_DATASET_FILES = (Path("corpus.jsonl"), Path("queries.jsonl"), *_QRELS_FILES)
_DOWNLOAD_TIMEOUT_SECONDS = 15


class _DatasetSourceError(Exception):
    """A source could not provide a usable dataset."""


def _missing_files(directory: Path) -> list[Path]:
    # Um arquivo vazio pode indicar um download interrompido.
    missing = []
    for name in _DATASET_FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    return missing


def _download_file(url: str, destination: Path) -> None:
    # O timeout limita a espera por operações de rede. A cópia lê o conteúdo em blocos.
    try:
        with urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
    except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
        raise _DatasetSourceError(f"{url}: {exc}") from exc
    # Quando o servidor informa o tamanho, compare os bytes recebidos para detectar truncamento.
    if content_length is not None and content_length.isdecimal():
        if destination.stat().st_size != int(content_length):
            raise _DatasetSourceError(f"{url}: incomplete download")


def _extract_dataset(archive: Path, dataset_name: str, destination: Path) -> None:
    """Extract the five expected files without trusting paths from the archive."""
    try:
        with zipfile.ZipFile(archive) as package:
            # O ZIP do BEIR contém uma pasta com o nome da coleção dentro do arquivo.
            for name in _DATASET_FILES:
                member = f"{dataset_name}/{name.as_posix()}"
                with package.open(member) as source, (destination / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise _DatasetSourceError(f"{archive.name}: {exc}") from exc


def _download_nfcorpus_hf(destination: Path) -> None:
    # Verifique os dois GZIPs antes de descompactá-los para o formato JSONL do carregador.
    for filename, expected_hash in _NFCORPUS_TEXT_HASHES.items():
        compressed = destination / filename
        _download_file(_NFCORPUS_HF_URL + filename, compressed)
        with compressed.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected_hash:
                raise _DatasetSourceError(f"Checksum mismatch for {filename}")
        with gzip.open(compressed, "rb") as source:
            with (destination / filename.removesuffix(".gz")).open("wb") as output:
                shutil.copyfileobj(source, output)

    # Os TSVs já vêm no formato esperado. O cabeçalho confirma a ordem das três colunas.
    for name in _QRELS_FILES:
        path = destination / name
        _download_file(_NFCORPUS_QRELS_HF_URL + name.name, path)
        with path.open("rb") as qrels:
            if qrels.readline().strip() != b"query-id\tcorpus-id\tscore":
                raise _DatasetSourceError(f"Invalid qrels header in {name}")


class DataLoader:
    """Downloads (when absent) and loads BEIR train/dev/test splits."""

    def __init__(
        self,
        dataset_name: str = "nfcorpus",
        data_dir: str | Path = "datasets",
    ) -> None:
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)

    def _resolve_data_path(self) -> Path:
        local_dir = self.data_dir / self.dataset_name
        # Com os cinco arquivos disponíveis, a leitura pode começar offline.
        if not _missing_files(local_dir):
            return local_dir

        self.data_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"{self.dataset_name}.zip"
        local_archive = self.data_dir / archive_name
        beir_url = _BEIR_URL + archive_name

        # Prepare a fonte em uma pasta temporária. Ela é removida ao sair do bloco,
        # inclusive quando algum download ou extração falha.
        with tempfile.TemporaryDirectory(prefix=".beir-download-", dir=self.data_dir) as temp:
            staging = Path(temp)
            (staging / "qrels").mkdir()
            self._prepare_dataset(staging, local_archive, beir_url)

            # Só mova os arquivos para a pasta da coleção após preparar o conjunto completo.
            # Complete os ausentes ou vazios e preserve os arquivos locais já preenchidos.
            for name in _missing_files(local_dir):
                destination = local_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / name).replace(destination)
        return local_dir

    def _prepare_dataset(self, staging: Path, local_archive: Path, beir_url: str) -> None:
        # Priorize o ZIP local para economizar uma nova transferência.
        if local_archive.is_file():
            try:
                _extract_dataset(local_archive, self.dataset_name, staging)
                if not _missing_files(staging):
                    return
                logger.warning("Cached BEIR archive is incomplete: %s", local_archive)
            except _DatasetSourceError as exc:
                logger.warning("Could not read cached BEIR archive: %s", exc)

        # O ZIP original do BEIR reúne os arquivos dos três splits em uma transferência.
        try:
            archive = staging / local_archive.name
            _download_file(beir_url, archive)
            _extract_dataset(archive, self.dataset_name, staging)
            if _missing_files(staging):
                raise _DatasetSourceError(f"Dataset files are missing or empty in {beir_url}")
            return
        except _DatasetSourceError as primary_error:
            # O par de repositórios do Hugging Face está configurado apenas para o NFCorpus.
            if self.dataset_name != "nfcorpus":
                raise
            logger.warning("BEIR archive unavailable (%s); trying BEIR on Hugging Face.", primary_error)
            try:
                _download_nfcorpus_hf(staging)
                if _missing_files(staging):
                    raise _DatasetSourceError("Hugging Face dataset files are missing or empty")
            except _DatasetSourceError as fallback_error:
                # A mensagem final mostra o que falhou em cada fonte e onde colocar os arquivos locais.
                raise _DatasetSourceError(
                    f"BEIR archive: {primary_error}. BEIR on Hugging Face: {fallback_error}. "
                    f"Retry with network access or place the five BEIR files in {self.data_dir / self.dataset_name}."
                ) from fallback_error

    def load_data(self) -> dict[str, tuple[dict[str, Any], dict[str, str], dict[str, dict[str, int]]]]:
        """Load train, dev, and test splits from BEIR."""
        try:
            from beir.datasets.data_loader import GenericDataLoader

            data_path = self._resolve_data_path()

            # Cada split devolve a coleção, suas consultas e os respectivos julgamentos.
            corpus_train, queries_train, qrels_train = GenericDataLoader(
                str(data_path)
            ).load(split="train")
            corpus_dev, queries_dev, qrels_dev = GenericDataLoader(
                str(data_path)
            ).load(split="dev")
            corpus_test, queries_test, qrels_test = GenericDataLoader(
                str(data_path)
            ).load(split="test")
            return {
                "train": (corpus_train, queries_train, qrels_train),
                "dev": (corpus_dev, queries_dev, qrels_dev),
                "test": (corpus_test, queries_test, qrels_test),
            }
        except Exception as e:
            raise ValueError(
                f"Error occurred while loading dataset {self.dataset_name}: {e}"
            ) from e
