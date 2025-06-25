from logging.handlers import RotatingFileHandler
import os
import argparse
import logging
from datetime import datetime
import sys
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


from model.feddpg import AdaptivePrompt
from data.glue import SST2Dataset, CoLADataset, MNLIDataset, RTEDataset, MRPCDataset
from data.yelp_polarity import YelpDataset
from data.agnews import AGNewsDataset

def get_dataset_class(dataset_name):
    dataset_map = {
        "sst2": SST2Dataset,
        "cola": CoLADataset,
        "mnli": MNLIDataset,
        "rte": RTEDataset,
        "mrpc": MRPCDataset,
        "yelp": YelpDataset,
        "agnews": AGNewsDataset
    }
    return dataset_map.get(dataset_name)

def calculate_performance(metric, true_labels, predictions):
    if metric == "accuracy":
        return accuracy_score(true_labels, predictions)
    elif metric == "matthews_correlation":
        return matthews_corrcoef(true_labels, predictions)
    elif metric == "f1":
        return f1_score(true_labels, predictions)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def generate_filename(args, is_checkpoint=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "eval"
    
    if is_checkpoint:
        return f"checkpoints/adaptive_prompt_fl_{args.train_dataset}_{mode}_e{args.eval_dataset}_p{args.prompt_length}_{timestamp}.pt"
    else:
        return f"logs/adaptive_prompt_fl_{args.train_dataset}_{mode}_e{args.eval_dataset}_p{args.prompt_length}_{timestamp}.log"

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def evaluate(args):
    device = torch.device(args.device)
    
    # Load the model
    model = AdaptivePrompt(model_name='roberta-base', num_labels=2, prompt_length=args.prompt_length)
    
    # Load only the prompt generator state dict
    prompt_generator_state_dict = torch.load(args.checkpoint_path)
    model.prompt_generator.load_state_dict(prompt_generator_state_dict)
    
    model.to(device)
    model.eval()

    # Prepare the dataset
    DatasetClass = get_dataset_class(args.eval_dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.eval_dataset}")
    
    eval_dataset = DatasetClass(split="validation", max_length=128)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    metric = DatasetClass.get_metric()
    logger.info(f"Using metric: {metric}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    score = calculate_performance(metric, all_labels, all_preds)
    logger.info(f"Evaluation {metric} on {args.eval_dataset}: {score:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Prompt model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the prompt generator checkpoint")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Dataset to evaluate on")
    parser.add_argument("--train_dataset", type=str, required=True, help="Dataset used for training")
    parser.add_argument("--prompt_length", type=int, default=10, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation")

    args = parser.parse_args()
    
    args.log_file = generate_filename(args)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    global logger
    logger = setup_logging(args.log_file)
    
    logger.info(f"Evaluation configuration:")
    logger.info(f"Checkpoint path: {args.checkpoint_path}")
    logger.info(f"Evaluation dataset: {args.eval_dataset}")
    logger.info(f"Training dataset: {args.train_dataset}")
    logger.info(f"Prompt length: {args.prompt_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")

    evaluate(args)

if __name__ == "__main__":
    main()
