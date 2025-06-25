# utils/logging_utils.py
import torch
import logging
from logging.handlers import RotatingFileHandler
import sys

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


def log_config(args, logger):
        # Log the seed
    logger.info(f"Random Seed: {args.seed}")
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
    logger.info(f"Mode: {args.mode}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    logger.info(f"Device: {args.device} (GPU: {gpu_name})")
    logger.info(f"Log file: {args.log_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Using non-IID splitting: {'Yes' if args.use_non_iid else 'No'}")
    if args.use_non_iid:
        logger.info(f"Alpha for non-IID splitting: {args.alpha_split}")
        logger.info(f"Alpha for device sample assignment: {args.alpha_device}")