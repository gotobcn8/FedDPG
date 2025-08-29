import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm  # Import tqdm for progress bar
from torch.optim import AdamW

def client_learn(logger,model, dataset, args):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.local_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    model.train()
    for epoch in range(args.local_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.local_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
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

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions.float() / total_predictions
        logger.info(f"Epoch {epoch+1}/{args.local_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
        

    # Return the state dictionaries of the prompt generator and classifier
    return {
        'prompt_generator': model.prompt_generator.state_dict(),
        'classifier': model.classifier.state_dict()
    }

def client_unlearn(logger, model, dataset, args, device, relabeled_indices, relabeled_labels):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.local_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    model.train()
    for epoch in range(args.local_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.local_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Relabel the data points that need to be unlearned
            for idx, relabeled_idx in enumerate(relabeled_indices):
                if relabeled_idx in batch['index']:
                    labels[batch['index'] == relabeled_idx] = relabeled_labels[idx]

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
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

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions.float() / total_predictions
        logger.info(f"Epoch {epoch+1}/{args.local_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
        

    # Return the state dictionaries of the prompt generator and classifier
    return {
        'prompt_generator': model.prompt_generator.state_dict(),
        'classifier': model.classifier.state_dict()
    }
