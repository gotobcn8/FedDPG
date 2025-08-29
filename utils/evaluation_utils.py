import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from collections.abc import Iterable
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np
import pickle
import json
class Indicator:
    def __init__(self,keys = None):
        if keys is not None:
            if isinstance(keys,str):
                # Invalid Input
                keys = [keys]
            elif not isinstance(keys,Iterable):
                raise Exception
            self.keys = set(keys)
        else:
            self.keys = set()
        self.results = {
            key:[] for key in self.keys
        }
        
    def insert(self,insert_key,value):
        if insert_key not in self.keys:
            self.keys.add(insert_key)
            self.results[insert_key] = []
            
        self.results[insert_key].append(value)
    
    def mean(self,mean_key):
        if mean_key not in self.keys:
            raise Exception(f'{mean_key} doesn\'t exist in this list')
        
        return np.mean(self.results[mean_key])
    
    def max(self,max_key):
        if max_key not in self.keys:
            raise Exception(f'{max_key} doesn\'t exist in this list')
        
        return np.max(self.results[max_key])
    
    def min(self,min_key):
        if min_key not in self.keys:
            raise Exception(f'{min_key} doesn\'t exist in this list')
        
        return np.max(self.results[min_key])

    def save(self,file_name = 'results.pkl',formation = 'pickle'):
        if formation == 'pickle':
            with open(file_name,'wb') as f:
                pickle.dump(self.results,f)
        elif formation == 'json':
            with open(file_name,'w') as f:
                json.dump(self.results,f)
        
    
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
    
    
def evaluate_global_model(
    args, 
    logger, 
    model, 
    val_data, 
    r=None, 
    indicator:Indicator = None,
    before_unlearn = False,
    index = '',
): # r = -1 for before unlearning process
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
    indicator.insert(f'global_{before_unlearn}_{index}',score)
    # if r == -1:
    #     logger.info(f"Global Model Validation {args.metric} Before Unlearning: {score:.4f}")
    # elif r is not None:
    #     logger.info(f"Round {r + 1}/{args.num_rounds} - Validation {args.metric}: {score:.4f}")
    # else:
    #     logger.info(f"Global Model Validation {args.metric} After Unlearning: {score:.4f}")
    
    
def evaluate_client_model(
    args, 
    logger, 
    model, 
    client_datasets, 
    indices, 
    before_unlearning = False,
    indicator:Indicator = None,
):
    model.eval()
    client_scores = {}
    if isinstance(indices,int):
        indices = [indices]
        
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
        indicator.insert(f'{client_id}_beforeunlearn_{before_unlearning}',score)
        client_scores[client_id] = score

    for client_id, score in client_scores.items():
        if before_unlearning:
            logger.info(f"Performance Impact - Before Unlearning - Client {client_id + 1} Validation {args.metric}: {score:.4f}")
        else:
            logger.info(f"Performance Impact - After Unlearning - Client {client_id + 1} Validation {args.metric}: {score:.4f}")