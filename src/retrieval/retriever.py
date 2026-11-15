class Retriever:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Initialize the retrieval model here (e.g., load a pre-trained model)

    def retrieve(self, query: str, top_k: int = 5):
        return