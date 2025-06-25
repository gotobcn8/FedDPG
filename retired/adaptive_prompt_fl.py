from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup
# from dataset.sst2 import SST2Dataset
from model.feddpg import AdaptivePrompt
import logging
from logging.handlers import RotatingFileHandler
import sys
import argparse
from tqdm import tqdm
import random
import os
from datetime import datetime
from data.glue import SST2Dataset, CoLADataset, MNLIDataset, RTEDataset, MRPCDataset
from data.yelp_polarity import YelpDataset
from data.agnews import AGNewsDataset
import numpy as np

def get_dataset_class(dataset_name):
    dataset_map = {
        "sst2": SST2Dataset,
        # "cola": CoLADataset,
        # "mnli": MNLIDataset,
        # "rte": RTEDataset,
        # "mrpc": MRPCDataset,
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
        num_labels = len(set(true_labels))
        if num_labels > 2:
            return f1_score(true_labels, predictions, average='macro')
        else:
            return f1_score(true_labels, predictions)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def generate_filename(args, is_checkpoint=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dev" if args.dev_mode else "full"
    
    if is_checkpoint:
        return f"checkpoints/adaptive_prompt_fl_{args.dataset}_{mode}_c{args.num_clients}_f{args.client_fraction}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}_{timestamp}.pt"
    else:
        return f"logs/adaptive_prompt_fl_{args.dataset}_{mode}_c{args.num_clients}_f{args.client_fraction}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}_{timestamp}.log"

# Set up logging
def setup_logging(log_file):
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

def client_train(model, train_dataloader, args, device):
    optimizer = AdamW(list(model.prompt_generator.parameters()) + list(model.classifier.parameters()), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.local_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    model.train()
    for epoch in range(args.local_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.local_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.shape[0]

            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions.float() / total_predictions
        logger.info(f"Epoch {epoch+1}/{args.local_epochs} - TrainingLoss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

    return {
        'prompt_generator': model.prompt_generator.state_dict(),
        'classifier': model.classifier.state_dict()
    }

def aggregate_model_states(model_states):
    if not model_states:    
        raise ValueError("No model states provided for aggregation")

    aggregated_state = {
        'prompt_generator': {},
        'classifier': {}
    }

    for component in ['prompt_generator', 'classifier']:
        reference_state = model_states[0][component]
        for key in reference_state.keys():
            stacked_params = torch.stack([state[component][key].to('cpu') for state in model_states])
            aggregated_state[component][key] = torch.mean(stacked_params, dim=0)

    return aggregated_state

def create_non_iid_splits(data, num_clients, alpha, num_classes):
    client_distributions = np.random.dirichlet(alpha * np.ones(num_classes), size=num_clients)
    
    class_indices = [np.where(np.array(data.targets) == i)[0] for i in range(num_classes)]
    
    client_data = [[] for _ in range(num_clients)]
    
    for c, class_idx in enumerate(class_indices):
        for i, idx in enumerate(np.random.permutation(class_idx)):
            client = np.random.choice(num_clients, p=client_distributions[:, c])
            client_data[client].append(idx)
    
    return client_data

def assign_samples_to_devices(client_datasets, alpha):
    total_samples = sum(len(dataset) for dataset in client_datasets)
    device_proportions = np.random.dirichlet(alpha * np.ones(len(client_datasets)))
    
    device_sample_counts = np.round(device_proportions * total_samples).astype(int)
    
    # Adjust to ensure the total matches
    device_sample_counts[-1] = total_samples - device_sample_counts[:-1].sum()
    
    return [dataset[:count] for dataset, count in zip(client_datasets, device_sample_counts)]

def federated_train(args):
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load the dataset
    logger.info(f"Loading {args.dataset} dataset...")
    DatasetClass = get_dataset_class(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    full_dataset = DatasetClass(split="train", max_length=128, model_name=args.model_name)
    val_dataset = DatasetClass(split="validation", max_length=128, model_name=args.model_name)

    metric = DatasetClass.get_metric()
    logger.info(f"Using metric: {metric}")

    if args.dev_mode:
        logger.info("Running in dev mode with minimal data")
        full_dataset = Subset(full_dataset, range(min(100, len(full_dataset))))
        val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))
        args.num_clients = min(args.num_clients, 2)
        args.num_rounds = 1
        args.local_epochs = 1

    tokenizer = full_dataset.tokenizer if hasattr(full_dataset, 'tokenizer') else DatasetClass(split="train", max_length=128, model_name=args.model_name).tokenizer
    label_map = full_dataset.label_map if hasattr(full_dataset, 'label_map') else DatasetClass(split="train", max_length=128, model_name=args.model_name).label_map
    num_labels = len(label_map)

    logger.info(f"Number of labels: {num_labels}")

    if args.use_non_iid:
        logger.info(f"Creating non-IID splits for {args.num_clients} clients...")
        
        num_classes = len(full_dataset.label_map)
        
        client_indices = create_non_iid_splits(full_dataset, args.num_clients, args.alpha_split, num_classes)
        
        client_datasets = {i: Subset(full_dataset, indices) for i, indices in enumerate(client_indices)}
        
        # Assign samples to devices using the assign_samples_to_devices function
        client_datasets = assign_samples_to_devices([client_datasets[i] for i in range(args.num_clients)], args.alpha_device)
        client_datasets = {i: dataset for i, dataset in enumerate(client_datasets)}
        
        logger.info(f"Non-IID splits created and assigned to devices successfully.")
    else:
        logger.info(f"Using IID data splitting for {args.num_clients} clients...")
        
        samples_per_client = len(full_dataset) // args.num_clients
        all_indices = list(range(len(full_dataset)))
        random.shuffle(all_indices)

        client_datasets = {}
        for client_id in range(args.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = all_indices[start_idx:end_idx]
            client_datasets[client_id] = Subset(full_dataset, client_indices)

    logger.info(f"Initializing global AdaptivePrompt model with {args.prompt_length} prompt tokens and {num_labels} labels...")
    global_model = AdaptivePrompt(model_name=args.model_name, num_labels=num_labels, prompt_length=args.prompt_length)
    
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    
    global_model.to(device)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    num_selected_clients = max(1, int(args.num_clients * args.client_fraction))

    for round in range(args.num_rounds):
        logger.info("#" * 80)
        logger.info(f"#{'':^78}#")
        logger.info(f"#{'Starting Federated Round':^78}#")
        logger.info(f"#{f'Round {round + 1}/{args.num_rounds}':^78}#")
        logger.info(f"#{'':^78}#")
        logger.info("#" * 80)

        selected_clients = random.sample(range(args.num_clients), num_selected_clients)
        logger.info(f"Selected clients: {[c + 1 for c in selected_clients]} ({num_selected_clients} clients)")

        client_model_states = []
        for idx, client_id in enumerate(selected_clients):
            logger.info("-" * 80)
            logger.info(f"Client {idx + 1} / {num_selected_clients}")
            logger.info(f"Client ID: {client_id + 1}")
            logger.info(f"Data samples: {len(client_datasets[client_id])}/{len(full_dataset)}")

            sample_indices = random.sample(range(len(client_datasets[client_id])), min(2, len(client_datasets[client_id])))
            for i, sample_idx in enumerate(sample_indices):
                sample = client_datasets[client_id][sample_idx]
                input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                label = label_map[sample['label'].item()]
                logger.info(f"Sample {i + 1}:")
                logger.info(f"  Input: {input_text}")
                logger.info(f"  Label: {label}")
            logger.info("-" * 80)
            
            client_dataset = client_datasets[client_id]
            client_dataloader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)
            
            client_model = AdaptivePrompt(model_name=args.model_name, num_labels=num_labels, prompt_length=args.prompt_length)
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)
            
            client_model_state = client_train(client_model, client_dataloader, args, device)
            client_model_states.append(client_model_state)

        logger.info(f"Aggregating model states from {num_selected_clients} clients")
        aggregated_model_state = aggregate_model_states(client_model_states)
        global_model.prompt_generator.load_state_dict(aggregated_model_state['prompt_generator'])
        global_model.classifier.load_state_dict(aggregated_model_state['classifier'])

        global_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
                    outputs = global_model(input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        score = calculate_performance(metric, all_labels, all_preds)

        logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric}: {score:.4f}")

    # Save the trained MLP state
    logger.info("Saving trained MLP state...")
    torch.save(global_model.prompt_generator.state_dict(), args.output_file)
    logger.info(f"MLP state saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Federated Adaptive Prompt Tuning for RoBERTa")
    parser.add_argument("--dataset", type=str, choices=["sst2", "cola", "mnli", "rte", "mrpc", "yelp", "agnews"], required=True, help="Dataset to use for training")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the pre-trained model")
    parser.add_argument("--prompt_length", type=int, default=10, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--num_clients", type=int, default=100, help="Total number of clients")
    parser.add_argument("--client_fraction", type=float, default=0.2, help="Fraction of clients to select each round")
    parser.add_argument("--local_epochs", type=int, default=100, help="Number of local training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--dev-mode", action="store_true", help="Run in development mode with minimal data")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision for training")
    parser.add_argument("--use-non-iid", action="store_true", help="Use non-IID data splitting")
    parser.add_argument("--alpha-split", type=float, default=0.5, help="Alpha parameter for Dirichlet distribution in non-IID data splitting")
    parser.add_argument("--alpha_device", type=float, default=5.0, help="Alpha parameter for Dirichlet distribution in device sample assignment")


    args = parser.parse_args()
    
    # Generate file names
    args.log_file = generate_filename(args)
    args.output_file = generate_filename(args, is_checkpoint=True)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_file)

    # Get dataset class and number of labels
    DatasetClass = get_dataset_class(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    label_map = DatasetClass(split="train", max_length=128, model_name=args.model_name).label_map
    num_labels = len(label_map)
    assert num_labels > 1, "num_labels must be set to a value greater than 1."
    
    logger.info(f"Federated training configuration:")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Dev mode: {'Enabled' if args.dev_mode else 'Disabled'}")
    logger.info(f"Prompt length: {args.prompt_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of rounds: {args.num_rounds}")
    logger.info(f"Total number of clients: {args.num_clients}")
    logger.info(f"Client fraction: {args.client_fraction}")
    logger.info(f"Local epochs: {args.local_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    logger.info(f"Device: {args.device} (GPU: {gpu_name})")
    logger.info(f"Use AMP: {'Yes' if args.use_amp else 'No'}")
    logger.info(f"Log file: {args.log_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Using non-IID splitting: {'Yes' if args.use_non_iid else 'No'}")
    if args.use_non_iid:
        logger.info(f"Alpha for non-IID splitting: {args.alpha_split}")
        logger.info(f"Alpha for device sample assignment: {args.alpha_device}")
        

    federated_train(args)

if __name__ == "__main__":
    main()