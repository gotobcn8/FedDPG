import os
import random
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from model.feddpg import AdaptivePrompt
from data.glue import SST2Dataset, CoLADataset, MNLIDataset, RTEDataset, MRPCDataset
from data.yelp_polarity import YelpDataset
from data.agnews import AGNewsDataset

import numpy as np

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

def client_train(model, train_dataloader, args, device):
    optimizer = AdamW(list(model.module.prompt_generator.parameters()) + list(model.module.classifier.parameters()), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.local_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    model.train()
    for epoch in range(args.local_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.local_epochs}", disable=not args.local_rank == 0)
        
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
        if args.local_rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.local_epochs} - TrainingLoss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

    return {
        'prompt_generator': model.module.prompt_generator.state_dict(),
        'classifier': model.module.classifier.state_dict()
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
    # Generate class distributions for each client
    client_distributions = np.random.dirichlet(alpha * np.ones(num_classes), size=num_clients)
    
    # Split data by class
    class_indices = [np.where(np.array(data.targets) == i)[0] for i in range(num_classes)]
    
    client_data = [[] for _ in range(num_clients)]
    
    # Distribute data to clients
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

def federated_train(args, local_rank):
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)

    if local_rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        if dist.is_initialized():
            logger.info(f"Number of GPUs used in this run: {dist.get_world_size()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"This process is running on GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    DatasetClass = get_dataset_class(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    full_dataset = DatasetClass(split="train", max_length=128, model_name=args.model_name)
    
    if args.dataset == "mnli":
        val_dataset_matched = DatasetClass(split="validation_matched", max_length=128, model_name=args.model_name)
        val_dataset_mismatched = DatasetClass(split="validation_mismatched", max_length=128, model_name=args.model_name)
    else:
        val_dataset = DatasetClass(split="validation", max_length=128, model_name=args.model_name)

    metric = DatasetClass.get_metric()
    if local_rank == 0:
        logger.info(f"Using metric: {metric}")

    if args.dev_mode:
        if local_rank == 0:
            logger.info("Running in dev mode with minimal data")
        full_dataset = Subset(full_dataset, range(min(100, len(full_dataset))))
        if args.dataset == "mnli":
            val_dataset_matched = Subset(val_dataset_matched, range(min(10, len(val_dataset_matched))))
            val_dataset_mismatched = Subset(val_dataset_mismatched, range(min(10, len(val_dataset_mismatched))))
        else:
            val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))
        args.num_clients = min(args.num_clients, 2)
        args.num_rounds = 1
        args.local_epochs = 1

    tokenizer = full_dataset.tokenizer if hasattr(full_dataset, 'tokenizer') else DatasetClass(split="train", max_length=128, model_name=args.model_name).tokenizer
    label_map = full_dataset.label_map if hasattr(full_dataset, 'label_map') else DatasetClass(split="train", max_length=128, model_name=args.model_name).label_map
    num_labels = len(label_map)  # Dynamically determine number of labels

    if local_rank == 0:
        logger.info(f"Number of labels: {num_labels}")

    if args.use_non_iid:
        if local_rank == 0:
            logger.info(f"Creating non-IID splits for {args.num_clients} clients...")
        
        # Get the number of classes for the dataset
        num_classes = len(full_dataset.label_map)
        
        # Create non-IID splits
        client_indices = create_non_iid_splits(full_dataset, args.num_clients, args.alpha_split, num_classes)
        
        # Create client datasets
        client_datasets = {i: Subset(full_dataset, indices) for i, indices in enumerate(client_indices)}
        
        # Assign samples to devices
        client_datasets = assign_samples_to_devices([client_datasets[i] for i in range(args.num_clients)], args.alpha_device)
        client_datasets = {i: dataset for i, dataset in enumerate(client_datasets)}

        if local_rank == 0:
            logger.info(f"Non-IID splits created successfully.")
    else:
        if local_rank == 0:
            logger.info(f"Using IID data splitting for {args.num_clients} clients...")
        
        # Original IID splitting logic
        samples_per_client = len(full_dataset) // args.num_clients
        all_indices = list(range(len(full_dataset)))
        random.shuffle(all_indices)

        client_datasets = {}
        for client_id in range(args.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = all_indices[start_idx:end_idx]
            client_datasets[client_id] = Subset(full_dataset, client_indices)

    if local_rank == 0:
        logger.info(f"Initializing global AdaptivePrompt model with {args.prompt_length} prompt tokens and {num_labels} labels...")
    global_model = AdaptivePrompt(model_name=args.model_name, num_labels=num_labels, prompt_length=args.prompt_length)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    
    global_model.to(device)
    if args.local_rank != -1:
        global_model = DDP(global_model, device_ids=[args.local_rank])

    if args.dataset == "mnli":
        val_sampler_matched = DistributedSampler(val_dataset_matched) if args.local_rank != -1 else None
        val_sampler_mismatched = DistributedSampler(val_dataset_mismatched) if args.local_rank != -1 else None
        val_dataloader_matched = DataLoader(val_dataset_matched, batch_size=args.batch_size, sampler=val_sampler_matched)
        val_dataloader_mismatched = DataLoader(val_dataset_mismatched, batch_size=args.batch_size, sampler=val_sampler_mismatched)
    else:
        val_sampler = DistributedSampler(val_dataset) if args.local_rank != -1 else None
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    num_selected_clients = max(1, int(args.num_clients * args.client_fraction))

    for round in range(args.num_rounds):
        if local_rank == 0:
            logger.info("#" * 80)
            logger.info(f"#{'':^78}#")
            logger.info(f"#{'Starting Federated Round':^78}#")
            logger.info(f"#{f'Round {round + 1}/{args.num_rounds}':^78}#")
            logger.info(f"#{'':^78}#")
            logger.info("#" * 80)

        selected_clients = random.sample(range(args.num_clients), num_selected_clients)
        if local_rank == 0:
            logger.info(f"Selected clients: {[c + 1 for c in selected_clients]} ({num_selected_clients} clients)")

        client_model_states = []
        for idx, client_id in enumerate(selected_clients):
            if local_rank == 0:
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
            client_sampler = DistributedSampler(client_dataset) if args.local_rank != -1 else None
            client_dataloader = DataLoader(client_dataset, batch_size=args.batch_size, sampler=client_sampler, shuffle=(client_sampler is None))
            
            # Clone the global model state for each client
            client_model = AdaptivePrompt(model_name=args.model_name, num_labels=num_labels, prompt_length=args.prompt_length)
            
            # Handle DDP prefix
            state_dict = global_model.state_dict()
            if args.local_rank != -1:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            client_model.load_state_dict(state_dict)
            client_model.to(device)
            if args.local_rank != -1:
                client_model = DDP(client_model, device_ids=[args.local_rank])
            
            client_model_state = client_train(client_model, client_dataloader, args, device)
            client_model_states.append(client_model_state)

        if local_rank == 0:
            logger.info(f"Aggregating model states from {num_selected_clients} clients")
            aggregated_model_state = aggregate_model_states(client_model_states)
            global_model.module.prompt_generator.load_state_dict(aggregated_model_state['prompt_generator'])
            global_model.module.classifier.load_state_dict(aggregated_model_state['classifier'])

        dist.barrier()

        global_model.eval()
        if args.dataset == "mnli":
            all_preds_matched = []
            all_labels_matched = []
            all_preds_mismatched = []
            all_labels_mismatched = []

            with torch.no_grad():
                for batch in val_dataloader_matched:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
                        outputs = global_model(input_ids, attention_mask=attention_mask)

                    _, preds = torch.max(outputs, dim=1)
                    all_preds_matched.extend(preds.cpu().numpy())
                    all_labels_matched.extend(labels.cpu().numpy())

                for batch in val_dataloader_mismatched:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
                        outputs = global_model(input_ids, attention_mask=attention_mask)

                    _, preds = torch.max(outputs, dim=1)
                    all_preds_mismatched.extend(preds.cpu().numpy())
                    all_labels_mismatched.extend(labels.cpu().numpy())

            if args.local_rank != -1:
                all_preds_matched = torch.tensor(all_preds_matched).to(device)
                all_labels_matched = torch.tensor(all_labels_matched).to(device)
                all_preds_mismatched = torch.tensor(all_preds_mismatched).to(device)
                all_labels_mismatched = torch.tensor(all_labels_mismatched).to(device)
                gathered_preds_matched = [torch.zeros_like(all_preds_matched) for _ in range(dist.get_world_size())]
                gathered_labels_matched = [torch.zeros_like(all_labels_matched) for _ in range(dist.get_world_size())]
                gathered_preds_mismatched = [torch.zeros_like(all_preds_mismatched) for _ in range(dist.get_world_size())]
                gathered_labels_mismatched = [torch.zeros_like(all_labels_mismatched) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_preds_matched, all_preds_matched)
                dist.all_gather(gathered_labels_matched, all_labels_matched)
                dist.all_gather(gathered_preds_mismatched, all_preds_mismatched)
                dist.all_gather(gathered_labels_mismatched, all_labels_mismatched)
                if local_rank == 0:
                    all_preds_matched = torch.cat(gathered_preds_matched).cpu().numpy()
                    all_labels_matched = torch.cat(gathered_labels_matched).cpu().numpy()
                    all_preds_mismatched = torch.cat(gathered_preds_mismatched).cpu().numpy()
                    all_labels_mismatched = torch.cat(gathered_labels_mismatched).cpu().numpy()
                    score_matched = calculate_performance(metric, all_labels_matched, all_preds_matched)
                    score_mismatched = calculate_performance(metric, all_labels_mismatched, all_preds_mismatched)
                    logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric} (matched): {score_matched:.4f}")
                    logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric} (mismatched): {score_mismatched:.4f}")
            else:
                score_matched = calculate_performance(metric, all_labels_matched, all_preds_matched)
                score_mismatched = calculate_performance(metric, all_labels_mismatched, all_preds_mismatched)
                logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric} (matched): {score_matched:.4f}")
                logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric} (mismatched): {score_mismatched:.4f}")
        else:
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

            if args.local_rank != -1:
                all_preds = torch.tensor(all_preds).to(device)
                all_labels = torch.tensor(all_labels).to(device)
                gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
                gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_preds, all_preds)
                dist.all_gather(gathered_labels, all_labels)
                if local_rank == 0:
                    all_preds = torch.cat(gathered_preds).cpu().numpy()
                    all_labels = torch.cat(gathered_labels).cpu().numpy()
                    score = calculate_performance(metric, all_labels, all_preds)
                    logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric}: {score:.4f}")
            else:
                score = calculate_performance(metric, all_labels, all_preds)
                logger.info(f"Round {round + 1}/{args.num_rounds} - Validation {metric}: {score:.4f}")

    if local_rank == 0:
        logger.info("Saving trained model state...")
        torch.save(global_model.module.prompt_generator.state_dict(), args.output_file)
        logger.info(f"Model state saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Federated Adaptive Prompt Tuning for RoBERTa")
    parser.add_argument("--dataset", type=str, choices=["sst2", "cola", "mnli", "rte", "mrpc", "yelp", "agnews"], required=True, help="Dataset to use for training")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model name to use for training")
    parser.add_argument("--prompt_length", type=int, default=10, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--num_clients", type=int, default=100, help="Total number of clients")
    parser.add_argument("--client_fraction", type=float, default=0.2, help="Fraction of clients to select each round")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--dev-mode", action="store_true", help="Run in development mode with minimal data")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--use_non_iid", action="store_true", help="Use non-IID data splitting")
    parser.add_argument("--alpha_split", type=float, default=1.0, help="Alpha parameter for Dirichlet distribution in non-IID splitting")
    parser.add_argument("--alpha_device", type=float, default=5.0, help="Alpha parameter for Dirichlet distribution in device sample assignment")

    args = parser.parse_args()
    
    args.log_file = generate_filename(args)
    args.output_file = generate_filename(args, is_checkpoint=True)
    
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    global logger
    logger = setup_logging(args.log_file)
    
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device(args.device)

    DatasetClass = get_dataset_class(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    label_map = DatasetClass(split="train", max_length=128, model_name=args.model_name).label_map
    num_labels = len(label_map)
    assert num_labels > 1, "num_labels must be set to a value greater than 1."

    if args.local_rank in [-1, 0]:
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
        logger.info(f"Local rank: {args.local_rank}")
        logger.info(f"Log file: {args.log_file}")
        logger.info(f"Output file: {args.output_file}")
        logger.info(f"Using non-IID splitting: {'Yes' if args.use_non_iid else 'No'}")
        if args.use_non_iid:
            logger.info(f"Alpha for non-IID splitting: {args.alpha_split}")
            logger.info(f"Alpha for device sample assignment: {args.alpha_device}")

    federated_train(args, args.local_rank)

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()