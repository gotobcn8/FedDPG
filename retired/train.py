import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from model.roberta import RoBERTaClassifier, save_prompts
from dataset.yelp_polarity import YelpPolarityDataset
from dataset.sst2 import SST2Dataset
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(args):
    # Set the device to GPU 0
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # # Load the Yelp Polarity dataset using the custom YelpPolarityDataset class
    # logger.info("Loading Yelp Polarity dataset...")
    # train_dataset = YelpPolarityDataset(split="train", max_length=512)
    # val_dataset = YelpPolarityDataset(split="test", max_length=512)  # Yelp Polarity uses 'test' as validation
    
    # Load the SST-2 dataset using the custom SST2Dataset class
    logger.info("Loading SST-2 dataset...")
    train_dataset = SST2Dataset(split="train", max_length=128)
    val_dataset = SST2Dataset(split="validation", max_length=128)

    # Initialize tokenizer and model
    logger.info(f"Initializing RoBERTa model with {args.prompt_length} prompt tokens...")
    model = RoBERTaClassifier(model_name='roberta-base', num_labels=2, prompt_length=args.prompt_length)
    model.to(device)

    # Calculate and log the number of trainable parameters for prompts
    num_trainable_params = sum(p.numel() for p in model.prompt_embeddings.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters (prompts): {num_trainable_params}")

    # Prepare dataloaders
    logger.info("Preparing dataloaders...")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Initialize the GradScaler for AMP
    scaler = GradScaler()

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # Runs the forward pass with autocasting
            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

            # Scales loss and calls backward()
            scaler.scale(loss).backward()

            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)
            scheduler.step()

            # Updates the scale for next iteration
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                with autocast(device_type='cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.shape[0]

        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = correct_predictions.float() / total_predictions
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        logger.info(f"Validation accuracy: {accuracy:.4f}")

    # Save the trained prompts
    logger.info("Saving trained prompts...")
    save_prompts(model, args.output_file)
    logger.info(f"Prompts saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa prompts on Yelp Polarity dataset")
    parser.add_argument("--prompt_length", type=int, default=5, help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--output_file", type=str, default="checkpoints/sst2_prompts.pt", help="Output file for saved prompts")

    args = parser.parse_args()
    
    logger.info(f"Training configuration:")
    logger.info(f"Prompt length: {args.prompt_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output file: {args.output_file}")

    train(args)
