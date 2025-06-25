from datasets import load_dataset
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset

class SST2Dataset(Dataset):
    def __init__(self, split="train", max_length=128):
        # Load the dataset
        self.dataset = load_dataset("glue", "sst2")[split]
        
        # Initialize the RoBERTa tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Set maximum sequence length
        self.max_length = max_length


        # Add label map
        self.label_map = {0: "negative", 1: "positive"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['sentence']
        label = item['label']

        # Tokenize and encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def get_labels(self):
        return list(self.label_map.keys())  # Return [0, 1]

# Example usage:
# train_dataset = SST2Dataset(split="train")
# val_dataset = SST2Dataset(split="validation")
