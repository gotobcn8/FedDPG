import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import random
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from data.glue import SST2Dataset, CoLADataset, MNLIDataset, RTEDataset, MRPCDataset
from data.yelp_polarity import YelpDataset
from data.agnews import AGNewsDataset
from model.feddpg import AdaptivePrompt

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
        return f"checkpoints/adaptive_prompt_fl_unlearning_{args.dataset}_{mode}_c{args.num_clients}_f{args.client_fraction}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}_{timestamp}.pt"
    else:
        return f"logs/adaptive_prompt_fl_unlearning_{args.dataset}_{mode}_c{args.num_clients}_f{args.client_fraction}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}_{timestamp}.log"

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

def client_train(model, train_dataloader, args, device, relabeled_indices=None, relabeled_labels=None):
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
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if relabeled_indices is not None and relabeled_labels is not None:
                # Map global indices to batch indices
                batch_indices = [i for i, idx in enumerate(batch['index']) if idx in relabeled_indices]
                if batch_indices:
                    relabeled_mask = torch.zeros_like(labels, dtype=torch.bool)
                    relabeled_mask[batch_indices] = True
                    labels[relabeled_mask] = torch.tensor([relabeled_labels[relabeled_indices.index(idx)] for idx in batch['index'] if idx in relabeled_indices]).to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
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
    # Collect all labels from the dataset
    labels = [data._get_label(item) for item in data]

    # Create class indices based on the collected labels
    class_indices = [np.where(np.array(labels) == i)[0] for i in range(num_classes)]
    
    client_distributions = np.random.dirichlet(alpha * np.ones(num_classes), size=num_clients)
    client_data = [[] for _ in range(num_clients)]
    
    for c, class_idx in enumerate(class_indices):
        # Normalize the probabilities to ensure they sum to 1
        probabilities = client_distributions[:, c]
        probabilities /= probabilities.sum()  # Normalize

        if not np.isclose(probabilities.sum(), 1.0):
            print(f"Warning: Probabilities for class {c} do not sum to 1 after normalization: {probabilities.sum()}")

        for i, idx in enumerate(np.random.permutation(class_idx)):
            client = np.random.choice(num_clients, p=probabilities)
            client_data[client].append(idx)
    
    return client_data

def assign_samples_to_devices(client_datasets, alpha):
    total_samples = sum(len(dataset) for dataset in client_datasets)
    device_proportions = np.random.dirichlet(alpha * np.ones(len(client_datasets)))
    
    device_sample_counts = np.round(device_proportions * total_samples).astype(int)
    
    # Adjust to ensure the total matches
    device_sample_counts[-1] = total_samples - device_sample_counts[:-1].sum()
    
    return [dataset[:count] for dataset, count in zip(client_datasets, device_sample_counts)]

def federated_train(args, global_model, client_datasets, device):
    logger.info("Starting Federated Training...")
    val_dataloader = DataLoader(args.val_dataset, batch_size=args.batch_size)
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
            logger.info(f"Data samples: {len(client_datasets[client_id])}/{len(args.full_dataset)}")

            sample_indices = random.sample(range(len(client_datasets[client_id])), min(2, len(client_datasets[client_id])))
            for i, sample_idx in enumerate(sample_indices):
                sample = client_datasets[client_id][sample_idx]
                input_text = args.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
                label = args.label_map[sample['label'].item()]
                logger.info(f"Sample {i + 1}:")
                logger.info(f"  Input: {input_text}")
                logger.info(f"  Label: {label}")
            logger.info("-" * 80)
            
            client_dataset = client_datasets[client_id]
            client_dataloader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)
            
            client_model = AdaptivePrompt(model_name=args.model_name, num_labels=args.num_labels, prompt_length=args.prompt_length)
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

                with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
                    outputs = global_model(input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        score = calculate_performance(args.metric, all_labels, all_preds)

        logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {args.metric}: {score:.4f}")

    logger.info("Federated Training Completed.")
    return global_model

def perform_unlearning(args, global_model, client_datasets, device):
    logger.info("#" * 80)
    logger.info(f"#{'':^78}#")
    logger.info(f"#{'Starting Unlearning Process':^78}#")
    logger.info(f"#{'':^78}#")
    logger.info("#" * 80)

    client_id = args.unlearning_client_id
    portion_unlearn = args.portion_unlearn

    if client_id < 0 or client_id >= args.num_clients:
        raise ValueError(f"Invalid client_id {client_id}. Must be between 0 and {args.num_clients -1}.")

    client_dataset = client_datasets[client_id]
    num_unlearn = int(len(client_dataset) * portion_unlearn)

    if num_unlearn <= 0:
        raise ValueError(f"Portion to unlearn is too small, resulting in zero data points to unlearn.")

    logger.info(f"Selected Client for Unlearning: Client {client_id + 1}")
    logger.info(f"Number of data points to unlearn: {num_unlearn}")

    if len(client_dataset) < num_unlearn:
        raise ValueError(f"Client {client_id + 1} does not have enough data points to unlearn.")

    # Select data points to unlearn
    unlearn_indices = random.sample(range(len(client_dataset)), num_unlearn)
    logger.info(f"Data point indices to unlearn: {unlearn_indices}")

    # Relabel the selected data points
    relabeled_labels = []
    for idx in unlearn_indices:
        original_label = client_dataset[idx]['label'].item()
        available_labels = list(range(args.num_labels))
        available_labels.remove(original_label)
        new_label = random.choice(available_labels)
        relabeled_labels.append(new_label)
        logger.info(f"Relabeling data point {idx}: {original_label} -> {new_label}")

    # Prepare DataLoader with relabeled data
    client_dataloader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)

    # Create a copy of the global model for the client
    client_model = AdaptivePrompt(model_name=args.model_name, num_labels=args.num_labels, prompt_length=args.prompt_length)
    client_model.load_state_dict(global_model.state_dict())
    client_model.to(device)

    # Perform local training with relabeled data
    logger.info("Training client model with relabeled data for unlearning...")
    client_model_state = client_train(
        model=client_model,
        train_dataloader=client_dataloader,
        args=args,
        device=device,
        relabeled_indices=unlearn_indices,
        relabeled_labels=torch.tensor(relabeled_labels).to(device)
    )

    # Replace the global model with the unlearned client model
    logger.info("Replacing global model with the unlearned client model...")
    global_model.prompt_generator.load_state_dict(client_model_state['prompt_generator'])
    global_model.classifier.load_state_dict(client_model_state['classifier'])

    logger.info("Unlearning Process Completed.")
    return global_model, client_id, unlearn_indices

def evaluate_selected_clients(model, selected_clients, client_datasets, args, device):
    logger.info(f"Evaluating performance of {len(selected_clients)} selected clients...")
    client_performance = {}

    for client_id in selected_clients:
        client_dataset = client_datasets[client_id]
        dataloader = DataLoader(client_dataset, batch_size=args.batch_size)
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
                    outputs = model(input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        score = calculate_performance(args.metric, all_labels, all_preds)
        client_performance[client_id + 1] = score  # 1-based indexing for client IDs
        logger.info(f"Client {client_id + 1} {args.metric}: {score:.4f}")

    return client_performance

def main():
    parser = argparse.ArgumentParser(description="Federated Adaptive Prompt Tuning with Unlearning Simulation")
    parser.add_argument("--dataset", type=str, choices=["sst2", "cola", "mnli", "rte", "mrpc", "yelp", "agnews"], required=True, help="Dataset to use for training")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the pre-trained model")
    parser.add_argument("--prompt_length", type=int, default=10, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--num_clients", type=int, default=100, help="Total number of clients")
    parser.add_argument("--client_fraction", type=float, default=0.2, help="Fraction of clients to select each round")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--dev-mode", action="store_true", help="Run in development mode with minimal data")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision for training")
    parser.add_argument("--use-non-iid", action="store_true", help="Use non-IID data splitting")
    parser.add_argument("--alpha-split", type=float, default=0.5, help="Alpha parameter for Dirichlet distribution in non-IID data splitting")
    parser.add_argument("--alpha_device", type=float, default=5.0, help="Alpha parameter for Dirichlet distribution in device sample assignment")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint to load")
    parser.add_argument("--unlearning_client_id", type=int, required=True, help="Client ID (0-based index) to perform unlearning")
    parser.add_argument("--portion_unlearn", type=float, required=True, help="Portion of data points to unlearn")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "f1", "matthews_correlation"], help="Metric to evaluate performance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_eval_clients", type=int, default=5, help="Number of clients to evaluate for performance before and after unlearning")

    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Generate file names
    args.log_file = generate_filename(args)
    args.output_file = generate_filename(args, is_checkpoint=True)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_file)

    # Log the seed
    logger.info(f"Random Seed: {args.seed}")

    # Get dataset class and number of labels
    DatasetClass = get_dataset_class(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    full_dataset = DatasetClass(split="train", max_length=128, model_name=args.model_name)
    val_dataset = DatasetClass(split="validation", max_length=128, model_name=args.model_name)

    args.full_dataset = full_dataset
    args.val_dataset = val_dataset

    args.metric = args.metric

    label_map = full_dataset.label_map if hasattr(full_dataset, 'label_map') else DatasetClass(split="train", max_length=128, model_name=args.model_name).label_map
    args.label_map = label_map
    args.num_labels = len(label_map)
    assert args.num_labels > 1, "num_labels must be set to a value greater than 1."
    
    args.tokenizer = full_dataset.tokenizer if hasattr(full_dataset, 'tokenizer') else DatasetClass(split="train", max_length=128, model_name=args.model_name).tokenizer

    logger.info(f"Federated training configuration:")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Number of labels: {args.num_labels}")
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
    logger.info(f"Number of evaluation clients: {args.num_eval_clients}")

    # Create non-IID splits if required
    if args.use_non_iid:
        logger.info(f"Creating non-IID splits for {args.num_clients} clients...")
        
        num_classes = args.num_labels
        
        client_indices = create_non_iid_splits(full_dataset, args.num_clients, args.alpha_split, num_classes)
        
        client_datasets = {i: Subset(full_dataset, indices) for i, indices in enumerate(client_indices)}
        
        # Assign samples to devices using the assign_samples_to_devices function
        client_datasets_list = assign_samples_to_devices([client_datasets[i] for i in range(args.num_clients)], args.alpha_device)
        client_datasets = {i: client_datasets_list[i] for i in range(args.num_clients)}
        
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

    # Initialize global AdaptivePrompt model
    logger.info(f"Initializing global AdaptivePrompt model with {args.prompt_length} prompt tokens and {args.num_labels} labels...")
    global_model = AdaptivePrompt(model_name=args.model_name, num_labels=args.num_labels, prompt_length=args.prompt_length)
    
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        
        # Load the state dicts into the respective components
        global_model.prompt_generator.load_state_dict(checkpoint)
        # global_model.classifier.load_state_dict(checkpoint['classifier'])
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.info("No checkpoint provided. Initializing model with random weights.")

    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    
    global_model.to(args.device)

    # Evaluate global model before federated training
    logger.info("Evaluating global model before federated training...")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    global_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)

            with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
                outputs = global_model(input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    pre_train_score = calculate_performance(args.metric, all_labels, all_preds)
    logger.info(f"Pre-Training Global Model {args.metric}: {pre_train_score:.4f}")

    # Perform federated training
    global_model = federated_train(args, global_model, client_datasets, args.device)

    # Evaluate global model after federated training but before unlearning
    logger.info("Evaluating global model after federated training but before unlearning...")
    global_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)

            with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
                outputs = global_model(input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    post_train_score = calculate_performance(args.metric, all_labels, all_preds)
    logger.info(f"Post-Training Global Model {args.metric}: {post_train_score:.4f}")

    # Save the checkpoint after training
    logger.info("Saving global model checkpoint after federated training...")
    torch.save({
        'prompt_generator': global_model.prompt_generator.state_dict(),
        'classifier': global_model.classifier.state_dict()
    }, args.output_file)
    logger.info(f"Global model checkpoint saved to {args.output_file}")

    # Select evaluation clients
    eval_clients = random.sample(range(args.num_clients), args.num_eval_clients)
    logger.info(f"Selected Evaluation Clients: {[c + 1 for c in eval_clients]}")

    # Evaluate selected clients before unlearning
    logger.info("Evaluating selected clients before unlearning...")
    pre_unlearn_client_performance = evaluate_selected_clients(global_model, eval_clients, client_datasets, args, args.device)

    # Perform unlearning
    global_model, unlearned_client_id, unlearned_indices = perform_unlearning(args, global_model, client_datasets, args.device)

    # Evaluate global model after unlearning
    logger.info("Evaluating global model after unlearning...")
    global_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)

            with autocast(device_type='cuda' if 'cuda' in args.device else 'cpu'):
                outputs = global_model(input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    post_unlearn_score = calculate_performance(args.metric, all_labels, all_preds)
    logger.info(f"Post-Unlearning Global Model {args.metric}: {post_unlearn_score:.4f}")

    # Save the checkpoint after unlearning
    logger.info("Saving global model checkpoint after unlearning...")
    torch.save({
        'prompt_generator': global_model.prompt_generator.state_dict(),
        'classifier': global_model.classifier.state_dict()
    }, args.output_file)
    logger.info(f"Global model checkpoint saved to {args.output_file}")

    # Evaluate selected clients after unlearning
    logger.info("Evaluating selected clients after unlearning...")
    post_unlearn_client_performance = evaluate_selected_clients(global_model, eval_clients, client_datasets, args, args.device)

    # Log the comparison
    logger.info("#" * 80)
    logger.info(f"{'Performance Comparison':^80}")
    logger.info("#" * 80)
    logger.info(f"Pre-Training {args.metric}: {pre_train_score:.4f}")
    logger.info(f"Post-Training {args.metric}: {post_train_score:.4f}")
    logger.info(f"Post-Unlearning {args.metric}: {post_unlearn_score:.4f}")
    logger.info(f"Unlearning performed by Client {unlearned_client_id + 1} on data points {unlearned_indices}")
    logger.info("#" * 80)
    
    # Log client performance comparison
    logger.info("#" * 80)
    logger.info(f"{'Client Performance Before and After Unlearning':^80}")
    logger.info("#" * 80)
    for client_id in eval_clients:
        pre_perf = pre_unlearn_client_performance.get(client_id + 1, None)
        post_perf = post_unlearn_client_performance.get(client_id + 1, None)
        if pre_perf is not None and post_perf is not None:
            delta = post_perf - pre_perf
    logger.info(f"Client {client_id + 1}: Before Unlearning {args.metric}: {pre_perf:.4f}, After Unlearning {args.metric}: {post_perf:.4f}")
    logger.info("#" * 80)

if __name__ == "__main__":
    main()