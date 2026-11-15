class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_data(self):
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader

            data_path = util.download_and_unzip(
                    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(self.dataset_name),
                    "datasets",
                )
            
            corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
            corpus2, queries2, qrels2 = GenericDataLoader(data_path).load(split="val")
            corpus3, queries3, qrels3 = GenericDataLoader(data_path).load(split="test")
            return {
                "train": (corpus, queries, qrels),
                "val": (corpus2, queries2, qrels2),
                "test": (corpus3, queries3, qrels3),
            }
        except Exception as e:
            raise ValueError(f"Error occurred while loading dataset {self.dataset_name}: {e}")