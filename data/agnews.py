from datasets import load_dataset
from data.glue import GLUEDataset

class AGNewsDataset(GLUEDataset):
    def __init__(self, split="train", max_length=128, model_name="roberta-base"):
        super().__init__("ag_news", split, max_length, model_name)

    def _get_label_map(self):
        return {0: "World", 1: "Sports", 2: "Business", 3: "Science"}

    def _get_text(self, item):
        return item['text']

    def _get_label(self, item):
        return item['label']
    
    @staticmethod
    def get_metric():
        return "accuracy"

