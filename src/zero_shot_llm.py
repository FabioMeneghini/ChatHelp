from txtai.pipeline import Labels

class ZeroShotLLM:
    def __init__(self, model_name, *tags):
        try:
            self.labels = Labels(model_name)
            self.tags = list(tags)
        except Exception as e:
            raise e

    def classify(self, data):
        label_index = self.labels(data, self.tags)[0][0]
        return self.tags[label_index]
