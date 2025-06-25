import random
from utils.data_utils import prepare_data
from utils.evaluation_utils import evaluate_client_model, evaluate_global_model
from utils.logging_utils import log_config
from utils.model_utils import initialize_model
from federated.learning import fl_round
from federated.unlearning import fu_round

def server(args, logger):    
    # Load the dataset
    _, val_dataset, client_datasets = prepare_data(args, logger)
    
    # Log the config
    log_config(args, logger)
    
    # Initialize global model
    logger.info(f"Initializing global FedDPG model with {args.prompt_length} prompt vectors and {args.num_labels} labels...")
    if args.checkpoint_path and not args.is_model_init:
        logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    global_model = initialize_model(args)
    global_model.to(args.device)
    
    logger.info(f"Total trainable parameters: {global_model.total_trainable_parameters()}")

    # for round in range(args.num_rounds):
    # Choose operation mode
    if args.mode not in ['learning', 'unlearning', 'both']:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    if args.mode in ['learning', 'both']:
        logger.info("Starting federated learning rounds...")
        # Select clients for this round
        selected_clients = random.sample(range(args.num_clients), int(args.num_clients * args.client_fraction))
        logger.info(" ")
        logger.info("+" * 80)
        logger.info("+" * 80)
        logger.info(f"Selected clients: {[c + 1 for c in selected_clients]} ({len(selected_clients)} clients)")

        for learning_round in range(args.num_rounds):
            
            # Perform federated learning round
            global_model_state = fl_round(args, logger, global_model, learning_round, client_datasets, selected_clients)
            global_model.load_state(global_model_state)
            
            # Evaluate the model
            evaluate_global_model(args, logger, global_model, val_dataset, learning_round)
            
            # Save the global model state
            global_model.save_state(args.output_file)
            logger.info(f"Model state saved to {args.output_file}")
        
    if args.mode in ['unlearning', 'both']:
        logger.info("#" * 80)
        logger.info(f"#{'':^78}#")
        logger.info(f"#{'Starting Federated Unlearning Round':^78}#")
        logger.info(f"#{'':^78}#")
        logger.info("#" * 80)
        
        # Get the data indicies for performing the unlearning for the requested client
        client_dataset = client_datasets[args.unlearning_client_id]
        num_unlearn = int(len(client_dataset) * args.portion_unlearn)

        logger.info(f"Selected Client for Unlearning: Client {args.unlearning_client_id + 1}")
        logger.info(f"Number of data points to unlearn: {num_unlearn}")

        # Select data points to unlearn
        unlearn_indices = random.sample(range(len(client_dataset)), num_unlearn)
        logger.info(f"Data point indices to unlearn: {unlearn_indices}")
        
        # Get random clients to evaluate the model
        random_evaluation_clients = random.sample(range(args.num_clients), args.num_eval_clients)
        
        # Evaluate the model before unlearning for clients' local data
        evaluate_client_model(args, logger, global_model, client_datasets, random_evaluation_clients, before_unlearning=True)
        
        # Evaluate the global model before unlearning
        evaluate_global_model(args, logger, global_model, val_dataset, r=-1)
        
        # Perform federated unlearning round
        global_model_state = fu_round(args, logger, global_model, client_datasets, unlearn_indices)
        global_model.load_state(global_model_state)
        
        
        # Evaluate the model after unlearning
        evaluate_client_model(args, logger, global_model, client_datasets, random_evaluation_clients, before_unlearning=False)
        
        # Evaluate the global model
        evaluate_global_model(args, logger, global_model, val_dataset)
        
        # Save the global model state
        global_model.save_state(args.unlearned_output_file)
    

    logger.info("Done!")
