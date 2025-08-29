import random
import torch
from torch.utils.data import DataLoader
from federated.client import client_unlearn
from utils.data_utils import prepare_data
from utils.model_utils import initialize_model
import copy

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

def fu_round0810(args, logger, global_model, client_datasets, unlearn_indices):
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

def compute_unlearn_term(self, round_attack, attackers_round, round):
    ## Init unlearn term
    unlearning_term = global_model * 0.0
    alpha = - self.alpha
    # compute beta constraint in lipschitz inequality
    list_beta = []
    for idx in range(len(self.beta)): # idx: round_id
        beta = self.beta[idx]
        if idx in round_attack:
            for cid in attackers_round[round_attack.index(idx)]:
                beta -= 1.0 * self.client_vols[cid]/self.data_vol
                
        beta = beta * alpha + 1
        list_beta.append(beta)
            
        # compute unlearning-term
    for idx in range(len(round_attack)):
        round_id = round_attack[idx]
        # compute u-term at round round_id (attack round)
        unlearning_term = unlearning_term * list_beta[round_id]
        for c_id in attackers_round[idx]:
            unlearning_term += 1.0 * self.client_vols[c_id]/self.data_vol * self.grads_all_round[round_id][str(c_id)].to(self.model.get_device())
            self.grads_all_round[round_id][str(c_id)].cpu()
                
        if idx == len(round_attack) - 1: continue
        for r_id in range(round_id + 1, round_attack[idx + 1]):
            unlearning_term = unlearning_term * list_beta[r_id]
    unlearning_term = unlearning_term * self.theta
    return unlearning_term

def dict_multiply(d, added):
    '''
    Compute dict * base type only
    '''
    for name in d:
        d[name] *= added

def dict_minus(s, d):
    for name in s:
        s[name] -= d[name]

@torch.no_grad
def fastFedULIID(args, logger, global_model, target_client, round, grads_all_round, client_weights):
    # N = len()
    N = args.num_clients
    delta_t = copy.deepcopy(global_model)
    delta_t = delta_t.state_dict()
    for name in delta_t:
        delta_t[name] = 0.0
        
    print(type(global_model))
    global_model = global_model.state_dict()
    for t in range(1, round + 1):
        # delta_t = delta_t * (1 + args.alpha)
        dict_multiply(delta_t, 1. + args.alpha)
        # delta_t *= (1 + args.alpha)
        # for param in delta_t.parameters():
        #     param.mul_(1 + args.alpha)
            
        CN = len(grads_all_round[t-1])
        
        other_clients = [
            cid for cid in grads_all_round[t-1].keys() if int(cid) != target_client
        ]
        sum_other = copy.deepcopy(global_model)
        # sum_other = sum_other.state_dict()
        for i,cid in enumerate(other_clients):
            for param_name, param in grads_all_round[t-1][cid].items():
                if i == 0:
                    sum_other[param_name] = 0 
                sum_other[param_name] += param.to(global_model[param_name].device)
                # grads_all_round[param_name].cpu()
            # sum_other += grads_all_round[t-1][cid].to(global_model.device)
        
        for name in sum_other:
            delta_t[name] += (1 / (CN * (CN - 1))) * sum_other[name]
        # delta_t += (1 / (CN * (CN - 1))) * sum_other
        
        if str(target_client) in grads_all_round[t-1]:
            for name, param in grads_all_round[t-1][str(target_client)].items():
                delta_t[name] -= (1 / CN) * param.to(global_model[name].device)

        # for cid in grads_all_round[t-1].keys():
        #     grads_all_round[t-1][cid].cpu()
    dict_minus(global_model,delta_t)
    return global_model

@torch.no_grad
def fastFedUL_NonIID(args, logger, global_model, target_client, round, grads_all_round, client_datasets):
    # N = len()
    delta_t = copy.deepcopy(global_model)
    delta_t = delta_t.state_dict()
    for name in delta_t:
        delta_t[name] = 0.0
        
    print(type(global_model))
    global_model = global_model.state_dict()
    for t in range(1, round + 1):
        # delta_t = delta_t * (1 + args.alpha)
        dict_multiply(delta_t, 1. + args.alpha)
        # delta_t *= (1 + args.alpha)
        # for param in delta_t.parameters():
        #     param.mul_(1 + args.alpha)

        
        other_clients = [
            cid for cid in grads_all_round[t-1].keys() if int(cid) != target_client
        ]
        other_total_weight = 0
        for cid in other_clients:
            other_total_weight += client_datasets[cid]
        
        total_weight = 0
        for cid in grads_all_round[t-1].keys():
            total_weight += client_datasets[cid]
            
        sum_other = copy.deepcopy(global_model)
        # sum_other = sum_other.state_dict()
        for i,cid in enumerate(other_clients):
            for param_name, param in grads_all_round[t-1][cid].items():
                if i == 0:
                    sum_other[param_name] = 0 
                sum_other[param_name] +=  param.to(global_model[param_name].device) * (client_datasets[cid] / other_total_weight - client_datasets[cid] / total_weight)
                # grads_all_round[param_name].cpu()
            # sum_other += grads_all_round[t-1][cid].to(global_model.device)
        
        for name in sum_other:
            delta_t[name] += sum_other[name]
        # delta_t += (1 / (CN * (CN - 1))) * sum_other
        
        if str(target_client) in grads_all_round[t-1]:
            for name, param in grads_all_round[t-1][str(target_client)].items():
                delta_t[name] -= (client_datasets[target_client] / total_weight) * param.to(global_model[name].device)

        # for cid in grads_all_round[t-1].keys():
        #     grads_all_round[t-1][cid].cpu()
    dict_minus(global_model,delta_t)
    return global_model

# def fastFedUL(args, logger, global_model, unlearn_indices, client_weights, round, grads_all_round):
#     idx = unlearn_indices[0]
#     # global_model
#     alpha = -args.alpha
#     list_beta = []
#     betas = []
#     for idx in range(len(beta)):
#         beta = betas[idx]
#         beta = beta * alpha + 1
#         list_beta.append(beta)
    
#     for rid in range(round):
#         unlearning_term = unlearning_term * list_beta[rid]

#         for c in range(unlearn_indices):
#             unlearn_term += (
#                 client_weights[c] * grads_all_round[rid][str(c)].to(args.device)
#             )
#             grads_all_round[rid][str(c)].cpu()
    
#         # if rid < round - 1:
#         #     next_round_id = 
        
#     unlearning_term = unlearning_term * args.theta
#     return unlearn_term