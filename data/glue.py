from datasets import load_dataset
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset

class GLUEDataset(Dataset):
    def __init__(self, task_name, split="train", max_length=128, model_name="roberta-base"):
        self.task_name = task_name
        if task_name == "yelp":
            self.dataset = load_dataset("yelp_polarity")["test" if split == "validation" else split]
        elif task_name == "ag_news":
            self.dataset = load_dataset("ag_news")["test" if split == "validation" else split]
        else:
            self.dataset = load_dataset("glue", task_name)[split]
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label_map = self._get_label_map()

    def _get_label_map(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer.encode_plus(
            self._get_text(item),
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self._get_label(item), dtype=torch.long),
            'index': idx # added for unlearning
        }

    def _get_text(self, item):
        raise NotImplementedError("Subclasses must implement this method")

    def _get_label(self, item):
        raise NotImplementedError("Subclasses must implement this method")

    def get_labels(self):
        return list(self.label_map.keys())

    @staticmethod
    def get_metric():
        raise NotImplementedError("Subclasses must implement this method")


class SST2Dataset(GLUEDataset):
    def __init__(self, split="train", max_length=128, model_name="roberta-base"):
        super().__init__("sst2", split, max_length, model_name)

    def _get_label_map(self):
        return {0: "negative", 1: "positive"}

    def _get_text(self, item):
        return item['sentence']

    def _get_label(self, item):
        return item['label']

    @staticmethod
    def get_metric():
        return "accuracy"


class CoLADataset(GLUEDataset):
    def __init__(self, split="train", max_length=128):
        super().__init__("cola", split, max_length)

    def _get_label_map(self):
        return {0: "unacceptable", 1: "acceptable"}

    def _get_text(self, item):
        return item['sentence']

    def _get_label(self, item):
        return item['label']

    @staticmethod
    def get_metric():
        return "matthews_correlation"


class MNLIDataset(GLUEDataset):
    def __init__(self, split="train", max_length=128):
        if split == "validation":
            split = "validation_matched"  # Use matched validation set by default
        super().__init__("mnli", split, max_length)

    def _get_label_map(self):
        return {0: "entailment", 1: "neutral", 2: "contradiction"}

    def _get_text(self, item):
        return item['premise'] + " [SEP] " + item['hypothesis']

    def _get_label(self, item):
        return item['label']

    @staticmethod
    def get_metric():
        return "accuracy"


class RTEDataset(GLUEDataset):
    def __init__(self, split="train", max_length=128):
        super().__init__("rte", split, max_length)

    def _get_label_map(self):
        return {0: "entailment", 1: "not_entailment"}

    def _get_text(self, item):
        return item['sentence1'] + " [SEP] " + item['sentence2']

    def _get_label(self, item):
        return item['label']

    @staticmethod
    def get_metric():
        return "accuracy"


class MRPCDataset(GLUEDataset):
    def __init__(self, split="train", max_length=128):
        super().__init__("mrpc", split, max_length)

    def _get_label_map(self):
        return {0: "not_equivalent", 1: "equivalent"}

    def _get_text(self, item):
        return item['sentence1'] + " [SEP] " + item['sentence2']

    def _get_label(self, item):
        return item['label']

    @staticmethod
    def get_metric():
        return "f1"