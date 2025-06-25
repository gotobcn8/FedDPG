# utils/data_utils.py

from data.glue import SST2Dataset, CoLADataset, MNLIDataset, RTEDataset, MRPCDataset
from data.yelp_polarity import YelpDataset
from data.agnews import AGNewsDataset

from torch.utils.data import Subset
import random
from utils.non_iid_utils import create_non_iid_splits, assign_samples_to_devices

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

def prepare_data(args, logger):
    DatasetClass = get_dataset_class(args.dataset)
    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load datasets
    full_dataset = DatasetClass(split="train", max_length=128, model_name=args.model_name)
    val_dataset = DatasetClass(split="validation", max_length=128, model_name=args.model_name)

    # Handle development mode
    if args.dev_mode:
        logger.info("Running in dev mode with minimal data")
        full_dataset = Subset(full_dataset, range(min(100, len(full_dataset))))
        val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))
        args.num_clients = min(args.num_clients, 2)
        args.num_rounds = 1
        args.local_epochs = 1

    # Determine label map and tokenizer
    label_map = full_dataset.label_map if hasattr(full_dataset, 'label_map') else DatasetClass(split="train", max_length=128, model_name=args.model_name).label_map
    tokenizer = full_dataset.tokenizer if hasattr(full_dataset, 'tokenizer') else DatasetClass(split="train", max_length=128, model_name=args.model_name).tokenizer

    # Handle non-IID data splitting
    if args.use_non_iid:
        logger.info(f"Creating non-IID splits for {args.num_clients} clients...")
        num_classes = len(label_map)
        client_indices = create_non_iid_splits(full_dataset, args.num_clients, args.alpha_split, num_classes)
        client_datasets = {i: Subset(full_dataset, indices) for i, indices in enumerate(client_indices)}
        client_datasets = assign_samples_to_devices([client_datasets[i] for i in range(args.num_clients)], args.alpha_device)
        client_datasets = {i: dataset for i, dataset in enumerate(client_datasets)}
        logger.info("Non-IID splits created and assigned to clients successfully.")
    else:
        logger.info(f"Using IID data splitting for {args.num_clients} clients...")
        samples_per_client = len(full_dataset) // args.num_clients
        all_indices = list(range(len(full_dataset)))
        random.shuffle(all_indices)
        client_datasets = {i: Subset(full_dataset, all_indices[i * samples_per_client:(i + 1) * samples_per_client]) for i in range(args.num_clients)}
        
    # Update args with label map and tokenizer
    args.label_map = label_map
    args.num_labels = len(label_map)
    args.tokenizer = tokenizer

    return full_dataset, val_dataset, client_datasets