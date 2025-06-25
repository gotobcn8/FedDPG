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
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from model.feddpg import AdaptivePrompt
from dataset.glue import MRPCDataset, SST2Dataset, CoLADataset, RTEDataset

def get_dataset_class(dataset_name):
    dataset_map = {
        "sst2": SST2Dataset,
        "cola": CoLADataset,
        "rte": RTEDataset,
        "mrpc": MRPCDataset
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
    mode = "dev" if args.dev_mode else "full"
    
    if is_checkpoint:
        return f"checkpoints/adaptive_prompt_fl_multi_dataset_{mode}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}_{timestamp}.pt"
    else:
        return f"logs/adaptive_prompt_fl_multi_dataset_{mode}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}_{timestamp}.log"

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

def evaluate_model(model, eval_dataloader, metric, device, local_rank):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if local_rank != -1:
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
            return score
    else:
        score = calculate_performance(metric, all_labels, all_preds)
        return score

def show_prediction_examples(model, dataloader, device, num_examples=5):
    model.eval()
    all_examples = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            for i in range(input_ids.size(0)):
                all_examples.append({
                    'input': input_ids[i],
                    'true_label': labels[i].item(),
                    'predicted_label': preds[i].item()
                })
            
            if len(all_examples) >= num_examples:
                break
    
    return random.sample(all_examples, num_examples)

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
        logger.info(f"Number of GPUs used in this run: {dist.get_world_size()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"This process is running on GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    # Define the 4 GLUE datasets for each client
    client_datasets = {
        0: ("sst2", SST2Dataset(split="train", max_length=128)),
        1: ("cola", CoLADataset(split="train", max_length=128)),
        2: ("mrpc", MRPCDataset(split="train", max_length=128)),
        3: ("rte", RTEDataset(split="train", max_length=128))
    }

    # Prepare validation datasets and dataloaders
    val_datasets = {
        "sst2": SST2Dataset(split="validation", max_length=128),
        "cola": CoLADataset(split="validation", max_length=128),
        "mrpc": MRPCDataset(split="validation", max_length=128),  # Changed to "validation"
        "rte": RTEDataset(split="validation", max_length=128)
    }

    val_dataloaders = {}
    for dataset_name, dataset in val_datasets.items():
        val_sampler = DistributedSampler(dataset) if args.local_rank != -1 else None
        val_dataloaders[dataset_name] = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

    args.num_clients = 4  # Set the number of clients to 4
    num_selected_clients = 4  # Always select all clients

    if args.dev_mode:
        if local_rank == 0:
            logger.info("Running in dev mode with minimal data")
        for client_id in client_datasets:
            client_datasets[client_id] = (client_datasets[client_id][0], 
                                          client_datasets[client_id][1][:100])  # Use only 100 samples
        args.num_rounds = 1
        args.local_epochs = 1

    eval_dataset_class = get_dataset_class(args.eval_dataset)
    eval_dataset = eval_dataset_class(split="validation", max_length=128)
    eval_metric = eval_dataset_class.get_metric()

    if local_rank == 0:
        logger.info(f"Initializing global AdaptivePrompt model with {args.prompt_length} prompt tokens...")
    global_model = AdaptivePrompt(model_name='roberta-base', num_labels=2, prompt_length=args.prompt_length)
    
    if args.mode == 'unlearning' and args.checkpoint_file:
        if local_rank == 0:
            logger.info(f"Loading checkpoint from {args.checkpoint_file}")
        checkpoint = torch.load(args.checkpoint_file, map_location=device)
        global_model.load_state_dict(checkpoint)
    elif args.mode == 'unlearning' and not args.checkpoint_file:
        raise ValueError("Checkpoint file must be specified for unlearning mode")
    
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    
    global_model.to(device)
    if args.local_rank != -1:
        global_model = DDP(global_model, device_ids=[args.local_rank])

    eval_sampler = DistributedSampler(eval_dataset) if args.local_rank != -1 else None
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, sampler=eval_sampler)

    for round in range(args.num_rounds):
        if local_rank == 0:
            logger.info("#" * 80)
            logger.info(f"#{'':^78}#")
            logger.info(f"#{'Starting Federated Round':^78}#")
            logger.info(f"#{f'Round {round + 1}/{args.num_rounds}':^78}#")
            logger.info(f"#{'':^78}#")
            logger.info("#" * 80)

        if args.mode in ['learning', 'both']:
            # Learning phase
            if local_rank == 0:
                logger.info("Starting learning phase")
            client_model_states = []
            for client_id in range(args.num_clients):
                dataset_name, client_dataset = client_datasets[client_id]
                if local_rank == 0:
                    logger.info(f"Training on client {client_id + 1} with dataset: {dataset_name}")
                    logger.info(f"Data samples: {len(client_dataset)}")

                client_sampler = DistributedSampler(client_dataset) if args.local_rank != -1 else None
                client_dataloader = DataLoader(client_dataset, batch_size=args.batch_size, sampler=client_sampler, shuffle=(client_sampler is None))
                
                client_model_state = client_train(global_model, client_dataloader, args, device)
                client_model_states.append(client_model_state)

            if local_rank == 0:
                logger.info(f"Aggregating model states from {args.num_clients} clients")
                aggregated_model_state = aggregate_model_states(client_model_states)
                global_model.module.prompt_generator.load_state_dict(aggregated_model_state['prompt_generator'])
                global_model.module.classifier.load_state_dict(aggregated_model_state['classifier'])

        if args.mode in ['unlearning', 'both']:
            # Unlearning phase
            if local_rank == 0:
                logger.info("Starting unlearning phase")
            
            unlearn_client_id = args.unlearn_client
            dataset_name, unlearn_dataset = client_datasets[unlearn_client_id]
            
            if local_rank == 0:
                logger.info(f"Unlearning client {unlearn_client_id + 1} with dataset: {dataset_name}")
                logger.info(f"Data samples to unlearn: {len(unlearn_dataset)}")

            unlearn_sampler = DistributedSampler(unlearn_dataset) if args.local_rank != -1 else None
            unlearn_dataloader = DataLoader(unlearn_dataset, batch_size=args.batch_size, sampler=unlearn_sampler, shuffle=(unlearn_sampler is None))
            
            # Show examples and metrics before unlearning
            if local_rank == 0:
                logger.info("Examples before unlearning:")
                examples_before = show_prediction_examples(global_model, unlearn_dataloader, device)
                for i, example in enumerate(examples_before):
                    logger.info(f"Example {i+1}:")
                    logger.info(f"  Input: {example['input']}")
                    logger.info(f"  True label: {example['true_label']}")
                    logger.info(f"  Predicted label: {example['predicted_label']}")
                
                metric = val_datasets[dataset_name].get_metric()
                score_before = evaluate_model(global_model, unlearn_dataloader, metric, device, local_rank)
                logger.info(f"Metric before unlearning: {metric} = {score_before:.4f}")
            
            unlearned_model_state = client_unlearn(global_model, unlearn_dataloader, args, device)
            
            if local_rank == 0:
                logger.info("Updating global model with unlearned state")
                global_model.module.prompt_generator.load_state_dict(unlearned_model_state['prompt_generator'])
                global_model.module.classifier.load_state_dict(unlearned_model_state['classifier'])

                # Show examples and metrics after unlearning
                logger.info("Examples after unlearning:")
                examples_after = show_prediction_examples(global_model, unlearn_dataloader, device)
                for i, example in enumerate(examples_after):
                    logger.info(f"Example {i+1}:")
                    logger.info(f"  Input: {example['input']}")
                    logger.info(f"  True label: {example['true_label']}")
                    logger.info(f"  Predicted label: {example['predicted_label']}")
                
                score_after = evaluate_model(global_model, unlearn_dataloader, metric, device, local_rank)
                logger.info(f"Metric after unlearning: {metric} = {score_after:.4f}")

        dist.barrier()

        # Evaluation on all datasets
        if local_rank == 0:
            logger.info("Evaluating global model on all datasets:")
        
        scores = {}
        for dataset_name, dataloader in val_dataloaders.items():
            metric = val_datasets[dataset_name].get_metric()
            score = evaluate_model(global_model, dataloader, metric, device, local_rank)
            if local_rank == 0:
                scores[dataset_name] = score
                logger.info(f"  {dataset_name.upper()} - {metric}: {score:.4f}")
        
        if local_rank == 0:
            avg_score = sum(scores.values()) / len(scores)
            logger.info(f"Average score across all datasets: {avg_score:.4f}")

    if local_rank == 0:
        logger.info("Saving trained model state...")
        torch.save(global_model.module.prompt_generator.state_dict(), args.output_file)
        logger.info(f"Model state saved to {args.output_file}")
        
        
def client_unlearn(model, train_dataloader, args, device):
    optimizer = AdamW(list(model.module.prompt_generator.parameters()) + list(model.module.classifier.parameters()), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.local_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    model.train()
    for epoch in range(args.local_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Unlearning Epoch {epoch+1}/{args.local_epochs}", disable=not args.local_rank == 0)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Flip the labels
            labels = 1 - labels

            optimizer.zero_grad()

            with autocast(device_type='cuda' if 'cuda' in device.type else 'cpu'):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(train_dataloader)
        if args.local_rank == 0:
            logger.info(f"Unlearning Epoch {epoch+1}/{args.local_epochs} - Loss: {epoch_loss:.4f}")

    return {
        'prompt_generator': model.module.prompt_generator.state_dict(),
        'classifier': model.module.classifier.state_dict()
    }

def main():
    parser = argparse.ArgumentParser(description="Federated Adaptive Prompt Tuning for RoBERTa with 4 GLUE datasets")
    parser.add_argument("--prompt_length", type=int, default=10, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--dev-mode", action="store_true", help="Run in development mode with minimal data")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--eval_dataset", type=str, choices=["sst2", "cola", "mrpc", "rte"], default="sst2", help="Dataset to use for evaluation")
    parser.add_argument("--mode", type=str, choices=['learning', 'unlearning', 'both'], default='learning', 
                        help="Choose the mode of operation: learning, unlearning, or both")
    parser.add_argument("--unlearn_client", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Specify which client's data to unlearn (0-3)")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                        help="Path to the checkpoint file to load for unlearning")

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

    if args.local_rank in [-1, 0]:
        logger.info(f"Federated training configuration:")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Unlearn client: {args.unlearn_client}")
        logger.info(f"Checkpoint file: {args.checkpoint_file}")
        logger.info(f"Dev mode: {'Enabled' if args.dev_mode else 'Disabled'}")
        logger.info(f"Prompt length: {args.prompt_length}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Number of rounds: {args.num_rounds}")
        logger.info(f"Local epochs: {args.local_epochs}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Local rank: {args.local_rank}")
        
    logger.info("Calling federated_train function")
    federated_train(args, args.local_rank)
    logger.info("federated_train function completed")

    if args.local_rank != -1:
        dist.destroy_process_group()
    logger.info("Script execution completed")

if __name__ == "__main__":
    main()