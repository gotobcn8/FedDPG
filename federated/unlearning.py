import random
import torch
from torch.utils.data import DataLoader
from federated.client import client_unlearn
from utils.data_utils import prepare_data
from utils.model_utils import initialize_model

def fu_round(args, logger, global_model, client_datasets, unlearn_indices):




    client_dataset = client_datasets[args.unlearning_client_id]

    # Relabel the selected data points
    relabeled_labels = []
    for idx in unlearn_indices:
        original_label = client_dataset[idx]['label'].item()
        available_labels = list(range(args.num_labels))
        available_labels.remove(original_label)
        new_label = random.choice(available_labels)
        relabeled_labels.append(new_label)
        logger.info(f"Relabeling data point {idx}: {original_label} -> {new_label}")

    # Create a copy of the global model for the client
    client_model = initialize_model(args)
    client_model.load_state(global_model.state_dict())
    client_model.to(args.device)

    # Perform local training with relabeled data
    logger.info("Training client model with relabeled data for unlearning...")
    client_model_state = client_unlearn(
        logger=logger,
        model=client_model,
        dataset=client_dataset,
        args=args,
        device=args.device,
        relabeled_indices=unlearn_indices,
        relabeled_labels=torch.tensor(relabeled_labels).to(args.device)
    )

    # Replace the global model with the unlearned client model
    logger.info("Unlearning Process Completed.")

    return client_model_state
