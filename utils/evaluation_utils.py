import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

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
    
    
def evaluate_global_model(args, logger, model, val_data, r=None): # r = -1 for before unlearning process
    model.eval()
    all_preds = []
    all_labels = []
    
    dataloader = DataLoader(val_data, batch_size=args.batch_size)
    
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)

            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    score = calculate_performance(args.metric, all_labels, all_preds)
    if r == -1:
        logger.info(f"Global Model Validation {args.metric} Before Unlearning: {score:.4f}")
    elif r is not None:
        logger.info(f"Round {r + 1}/{args.num_rounds} - Validation {args.metric}: {score:.4f}")
    else:
        logger.info(f"Global Model Validation {args.metric} After Unlearning: {score:.4f}")
    
    
def evaluate_client_model(args, logger, model, client_datasets, indices, before_unlearning=False):
    model.eval()
    client_scores = {}

    for client_id in indices:
        all_preds = []
        all_labels = []
        dataloader = DataLoader(client_datasets[client_id], batch_size=args.batch_size)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['label'].to(args.device)

                with autocast(device_type='cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        score = calculate_performance(args.metric, all_labels, all_preds)
        client_scores[client_id] = score

    for client_id, score in client_scores.items():
        if before_unlearning:
            logger.info(f"Performance Impact - Before Unlearning - Client {client_id + 1} Validation {args.metric}: {score:.4f}")
        else:
            logger.info(f"Performance Impact - After Unlearning - Client {client_id + 1} Validation {args.metric}: {score:.4f}")