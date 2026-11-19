"""BEIR dataset loader for NFCorpus and compatible collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
        if (
            (local_dir / "corpus.jsonl").is_file()
            and (local_dir / "queries.jsonl").is_file()
            and (local_dir / "qrels").is_dir()
        ):
            return local_dir

        from beir import util

        self.data_dir.mkdir(parents=True, exist_ok=True)
        url = (
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
            f"{self.dataset_name}.zip"
        )
        return Path(util.download_and_unzip(url, str(self.data_dir)))

    def load_data(self) -> dict[str, tuple[dict[str, Any], dict[str, str], dict[str, dict[str, int]]]]:
        """Load train, dev, and test splits from BEIR."""
        try:
            from beir.datasets.data_loader import GenericDataLoader

            data_path = self._resolve_data_path()

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
