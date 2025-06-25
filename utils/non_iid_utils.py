import numpy as np

def create_non_iid_splits(data, num_clients, alpha, num_classes):
    client_distributions = np.random.dirichlet(alpha * np.ones(num_classes), size=num_clients)
    
    class_indices = [np.where(np.array(data.targets) == i)[0] for i in range(num_classes)]
    
    client_data = [[] for _ in range(num_clients)]
    
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