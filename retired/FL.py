import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset.sst2 import SST2Dataset
from model.roberta import RoBERTaClassifier, save_prompts
import logging
from logging.handlers import RotatingFileHandler
import sys
import argparse
from tqdm import tqdm
import random

# Set up logging
def setup_logging(log_file='terminal_output.txt'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Replace the existing logging setup with this function call
logger = setup_logging()

def client_train(model, train_dataloader, args, device):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.local_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    model.train()
    for epoch in range(args.local_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_loss += loss.item()

    return [p.data.clone() for p in model.prompt_embeddings]

def aggregate_prompts(prompt_list):
    aggregated_prompts = []
    for layer_prompts in zip(*prompt_list):
        aggregated_prompts.append(torch.mean(torch.stack(layer_prompts), dim=0))
    return aggregated_prompts

def federated_train(args):
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load the SST-2 dataset
    logger.info("Loading SST-2 dataset...")
    full_dataset = SST2Dataset(split="train", max_length=128)
    val_dataset = SST2Dataset(split="validation", max_length=128)

    # Create global model
    logger.info(f"Initializing global RoBERTa model with {args.prompt_length} prompt tokens...")
    global_model = RoBERTaClassifier(model_name='roberta-base', num_labels=2, prompt_length=args.prompt_length)
    global_model.to(device)

    # Prepare validation dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Federated learning loop
    for round in range(args.num_rounds):
        logger.info(f"Starting federated round {round + 1}/{args.num_rounds}")

        # Simulate client selection
        client_datasets = []
        for _ in range(args.num_clients):
            client_size = len(full_dataset) // args.num_clients
            client_indices = random.sample(range(len(full_dataset)), client_size)
            client_datasets.append(Subset(full_dataset, client_indices))

        # Client training
        client_prompts = []
        for client_id, client_dataset in enumerate(client_datasets):
            logger.info(f"Training client {client_id + 1}/{args.num_clients}")
            client_model = RoBERTaClassifier(model_name='roberta-base', num_labels=2, prompt_length=args.prompt_length)
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)

            client_dataloader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)
            client_prompts.append(client_train(client_model, client_dataloader, args, device))

        # Server aggregation
        logger.info("Aggregating client prompts")
        aggregated_prompts = aggregate_prompts(client_prompts)
        for i, prompt in enumerate(aggregated_prompts):
            global_model.prompt_embeddings[i].data = prompt.to(device)

        # Validation
        global_model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with autocast(device_type='cuda'):
                    outputs = global_model(input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.shape[0]

        accuracy = correct_predictions.float() / total_predictions
        logger.info(f"Round {round + 1}/{args.num_rounds} - Validation accuracy: {accuracy:.4f}")

    # Save the trained prompts
    logger.info("Saving trained prompts...")
    save_prompts(global_model, args.output_file)
    logger.info(f"Prompts saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Prompt Tuning for RoBERTa on SST-2 dataset")
    parser.add_argument("--prompt_length", type=int, default=10, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of federated rounds")
    parser.add_argument("--num_clients", type=int, default=100, help="Number of clients per round")
    parser.add_argument("--local_epochs", type=int, default=50, help="Number of local training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training (default: GPU 0 if available, else CPU)")
    parser.add_argument("--output_file", type=str, default="checkpoints/deep-prompt-fl-sst2.pt", help="Output file for saved prompts")
    parser.add_argument("--log_file", type=str, default="terminal_output.txt", help="File to save terminal output")

    args = parser.parse_args()
    
    logger.info(f"Federated training configuration:")
    logger.info(f"Prompt length: {args.prompt_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of rounds: {args.num_rounds}")
    logger.info(f"Number of clients: {args.num_clients}")
    logger.info(f"Local epochs: {args.local_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output file: {args.output_file}")

    federated_train(args)
