from datasets import load_dataset
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset

# class YelpDataset(Dataset):
#     def __init__(self, split="train", max_length=128):
#         # Load the dataset
#         if split == "validation":
#             split = "test"  # Yelp uses "test" instead of "validation"
#         self.dataset = load_dataset("yelp_polarity")[split]
        
#         # Initialize the RoBERTa tokenizer
#         self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
#         # Set maximum sequence length
#         self.max_length = max_length

#         # Add label map
#         self.label_map = {0: "negative", 1: "positive"}

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         text = item['text']
#         label = item['label']

#         # Tokenize and encode the text
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'label': torch.tensor(label, dtype=torch.long)
#         }

#     def get_labels(self):
#         return list(self.label_map.keys())  # Return [0, 1]

# Example usage:
# train_dataset = YelpDataset(split="train")
# val_dataset = YelpDataset(split="validation")  # This will use the "test" split


from datasets import load_dataset
from data.glue import GLUEDataset

class YelpDataset(GLUEDataset):
    def __init__(self, split="train", max_length=128, model_name="roberta-base"):
        super().__init__("yelp", split, max_length, model_name)
        # self.dataset = load_dataset("yelp_polarity")[split]

    def _get_label_map(self):
        return {0: "negative", 1: "positive"}

    def _get_text(self, item):
        return item['text']

    def _get_label(self, item):
        return item['label']
    
    @staticmethod
    def get_metric():
        return "accuracy"