class Experiment:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config['data_path'])
        self.pre_processor = PreProcessor(config)
        self.retriever = Retriever(config['retrieval_model'])

    def run(self, data):
        preprocessed_data = self.pre_processor.preprocess(data)
        results = self.retriever.retrieve(preprocessed_data)
        return results