import argparse
import os
import random
import numpy as np
import torch
from utils.logging_utils import setup_logging
from config import DEFAULT_CONFIG
from utils.file_name_utils import generate_filename
from federated.server import server

def main():
    parser = argparse.ArgumentParser(description="Federated Adaptive Prompt Tuning")
    parser.add_argument("--dataset", type=str, choices=["sst2", "cola", "mnli", "rte", "mrpc", "yelp", "agnews"], default='sst2', help="Dataset to use for training")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"], help="Name of the pre-trained model")
    parser.add_argument("--prompt_length", type=int, default=DEFAULT_CONFIG["prompt_length"], help="Length of the prompt")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_CONFIG["num_rounds"], help="Number of federated rounds")
    parser.add_argument("--num_clients", type=int, default=DEFAULT_CONFIG["num_clients"], help="Total number of clients")
    parser.add_argument("--client_fraction", type=float, default=DEFAULT_CONFIG["client_fraction"], help="Fraction of clients to select each round")
    parser.add_argument("--local_epochs", type=int, default=DEFAULT_CONFIG["local_epochs"], help="Number of local training epochs")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"], help="Device to use for training")
    parser.add_argument("--dev_mode", action="store_true", help="Run in development mode with minimal data")
    parser.add_argument("--use_non_iid", action="store_true", help="Use non-IID data splitting")
    parser.add_argument("--alpha_split", type=float, default=DEFAULT_CONFIG["alpha_split"], help="Alpha parameter for Dirichlet distribution in non-IID data splitting")
    parser.add_argument("--alpha_device", type=float, default=DEFAULT_CONFIG["alpha_device"], help="Alpha parameter for Dirichlet distribution in device sample assignment")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CONFIG["checkpoint_path"], help="Path to the checkpoint to load")
    parser.add_argument("--unlearning_client_id", type=int, default=DEFAULT_CONFIG["unlearning_client_id"], help="Client ID (0-based index) to perform unlearning")
    parser.add_argument("--portion_unlearn", type=float, default=DEFAULT_CONFIG["portion_unlearn"], help="Portion of data points to unlearn")
    parser.add_argument("--metric", type=str, default=DEFAULT_CONFIG["metric"], choices=["accuracy", "f1", "matthews_correlation"], help="Metric to evaluate performance")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed for reproducibility")
    parser.add_argument("--num_eval_clients", type=int, default=DEFAULT_CONFIG["num_eval_clients"], help="Number of clients to evaluate for performance before and after unlearning")
    parser.add_argument("--mode", type=str, choices=['learning', 'unlearning', 'both'], default=DEFAULT_CONFIG["mode"], help="Choose the mode of operation: learning, unlearning, or both")
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"], help="alpha")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Generate file names
    args.log_file = generate_filename(args, "log")
    args.output_file = generate_filename(args, "checkpoint")
    args.unlearned_output_file = generate_filename(args, "unlearned_checkpoint")

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Set the model global initialization
    args.is_model_init = False

    # Run the server with the provided arguments and logger
    server(args, logger)

if __name__ == "__main__":
    main()
