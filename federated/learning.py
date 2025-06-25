import random
from federated.client import client_learn
from utils.aggregation_utils import aggregate_model_states
from utils.model_utils import initialize_model

def fl_round(args, logger, global_model, r, client_datasets, selected_clients):


    logger.info("#" * 80)
    logger.info(f"#{'':^78}#")
    logger.info(f"#{'Starting Federated Round':^78}#")
    logger.info(f"#{f'Round {r + 1}/{args.num_rounds}':^78}#")
    logger.info(f"#{'':^78}#")
    logger.info("#" * 80)

    client_model_states = []
    for idx, client_id in enumerate(selected_clients):
        logger.info("-" * 80)
        logger.info(f"Client {idx + 1} / {len(selected_clients)}")
        logger.info(f"Client ID: {client_id + 1}")
        logger.info("-" * 80)

        client_dataset = client_datasets[client_id]

        client_model = initialize_model(args)
        client_model.load_state(global_model.state_dict())
        client_model.to(args.device)

        client_model_state = client_learn(logger, client_model, client_dataset, args)
        client_model_states.append(client_model_state)

    logger.info(f"Aggregating model states from {args.num_clients} clients")
    aggregated_model_state = aggregate_model_states(client_model_states)
    
    return aggregated_model_state
