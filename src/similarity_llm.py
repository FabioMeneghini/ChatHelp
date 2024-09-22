from txtai import Embeddings

class SimilarityLLM:
    def __init__(self, model_path):
        try:
            self.embeddings = Embeddings(path=model_path, content=True)
            self.model_path = model_path
        except Exception as e:
            raise e
    
    def get_path(self):
        return self.model_path

    def get_vector(self, text):
        return self.embeddings.transform(text).tolist()