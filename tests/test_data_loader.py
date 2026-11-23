import gzip
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from src.utils.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        # Cada teste recebe uma pasta temporária e uma coleção mínima com um documento,
        # uma consulta e um julgamento em cada split.
        self.data_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.dataset_dir = self.data_dir / "nfcorpus"
        self.files = {
            "corpus.jsonl": b'{"_id": "D1", "title": "Title", "text": "A document"}\n',
            "queries.jsonl": b'{"_id": "Q1", "text": "A query?"}\n',
        }
        for split in ("train", "dev", "test"):
            self.files[f"qrels/{split}.tsv"] = b"query-id\tcorpus-id\tscore\nQ1\tD1\t2\n"

    def write_local_file(self, name: str, content: bytes) -> None:
        path = self.dataset_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def archive_bytes(self) -> bytes:
        # Reproduza a estrutura do ZIP do BEIR em memória para dispensar um download real.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in self.files.items():
                archive.writestr(f"nfcorpus/{name}", data)
        return buffer.getvalue()

    def assert_dataset_loads(self) -> None:
        splits = DataLoader(data_dir=self.data_dir).load_data()
        self.assertEqual(set(splits), {"train", "dev", "test"})
        for corpus, queries, qrels in splits.values():
            self.assertEqual(corpus["D1"]["text"], "A document")
            self.assertEqual(queries, {"Q1": "A query?"})
            self.assertEqual(qrels, {"Q1": {"D1": 2}})

    @staticmethod
    def response(data: bytes, *, content_length: int | None = None) -> io.BytesIO:
        # Imite uma resposta HTTP e permita anunciar um tamanho maior para simular truncamento.
        response = io.BytesIO(data)
        response.headers = {"Content-Length": str(len(data) if content_length is None else content_length)}
        return response

    def test_complete_dataset_and_cached_zip_load_without_network(self) -> None:
        archive = self.data_dir / "nfcorpus.zip"
        archive.write_bytes(self.archive_bytes())
        with patch("src.utils.data_loader.urlopen") as request:
            # A primeira carga usa o ZIP. A segunda usa apenas os arquivos já extraídos.
            self.assert_dataset_loads()
            archive.unlink()
            self.assert_dataset_loads()
        request.assert_not_called()

    def test_primary_download_fills_missing_qrels_and_preserves_local_files(self) -> None:
        # Use um texto diferente na cópia local para detectar uma substituição indevida.
        local_queries = b'{"_id": "Q1", "text": "Locally supplied query?"}\n'
        for name, content in self.files.items():
            if name != "qrels/dev.tsv":
                self.write_local_file(name, content)
        self.write_local_file("queries.jsonl", local_queries)

        with patch("src.utils.data_loader.urlopen", return_value=self.response(self.archive_bytes())) as request:
            splits = DataLoader(data_dir=self.data_dir).load_data()

        self.assertEqual(splits["dev"][1], {"Q1": "Locally supplied query?"})
        self.assertEqual(splits["dev"][2], {"Q1": {"D1": 2}})
        self.assertEqual(request.call_count, 1)
        self.assertGreater(request.call_args.kwargs["timeout"], 0)

    def test_connection_timeout_falls_back_to_beir_hugging_face(self) -> None:
        # Simule o formato da alternativa com dois GZIPs e três TSVs.
        remote_files = {
            name + ".gz": gzip.compress(data)
            for name, data in self.files.items()
            if name.endswith(".jsonl")
        }
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in remote_files.items()}
        remote_files.update({Path(name).name: data for name, data in self.files.items() if name.endswith(".tsv")})

        def fetch(url: str, *, timeout: int) -> io.BytesIO:
            self.assertGreater(timeout, 0)
            if url.endswith("nfcorpus.zip"):
                raise URLError(TimeoutError("connection timed out"))
            self.assertTrue(url.startswith("https://huggingface.co/datasets/BeIR/"))
            return self.response(remote_files[url.rsplit("/", 1)[-1]])

        # Substitua a rede e os hashes de referência pelos da coleção de teste.
        # O carregador ainda verifica os hashes, extrai os arquivos e lê os três splits.
        with (
            patch("src.utils.data_loader._NFCORPUS_TEXT_HASHES", hashes),
            patch("src.utils.data_loader.urlopen", side_effect=fetch) as request,
            self.assertLogs("src.utils.data_loader", level="WARNING"),
        ):
            self.assert_dataset_loads()
            self.assert_dataset_loads()
        # As seis chamadas são a tentativa ao ZIP e os cinco downloads da primeira carga.
        self.assertEqual(request.call_count, 6)

    def test_two_failed_sources_leave_local_directory_intact(self) -> None:
        # O ZIP chega incompleto e a fonte alternativa está inacessível.
        self.write_local_file("corpus.jsonl", self.files["corpus.jsonl"])
        partial_zip = self.response(b"partial", content_length=200)
        with (
            patch("src.utils.data_loader.urlopen", side_effect=[partial_zip, URLError("offline")]),
            self.assertLogs("src.utils.data_loader", level="WARNING"),
            self.assertRaisesRegex(ValueError, "incomplete download.*Hugging Face: .*offline"),
        ):
            DataLoader(data_dir=self.data_dir).load_data()

        # A pasta temporária foi removida e só resta o arquivo local que já existia.
        self.assertEqual(list(self.data_dir.iterdir()), [self.dataset_dir])
        self.assertEqual(list(self.dataset_dir.iterdir()), [self.dataset_dir / "corpus.jsonl"])
