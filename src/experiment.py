from src.pre_processing import PreProcessor
from src.utils import DataLoader
from src.retrieval import Retriever

class Experiment:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config['dataset_name'])
        self.pre_processor = PreProcessor(config)
        self.retriever = Retriever(config['retrieval_model'])

    def setup(self):
        # Load and preprocess data
        data = self.data_loader.load_data()
        preprocessed_data = self.pre_processor.preprocess(data)
        return preprocessed_data

    def run(self, data):
        preprocessed_data = self.pre_processor.preprocess(data)
        results = self.retriever.retrieve(preprocessed_data)
        return results

if __name__ == "__main__":
    config = {
        'dataset_name': 'nfcorpus',
        # Add other configuration parameters as needed
    }
    experiment = Experiment(config)
    data = experiment.setup()
    results = experiment.run(data)
    print(results)