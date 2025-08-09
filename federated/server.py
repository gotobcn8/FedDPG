import random
from utils.data_utils import prepare_data
from utils.evaluation_utils import evaluate_client_model, evaluate_global_model, Indicator
from utils.logging_utils import log_config
from utils.model_utils import initialize_model
from federated.learning import fl_round
from federated.unlearning import fu_round, fastFedULIID, fastFedUL_NonIID
import os
import copy
import torch

def __init_client_model_store_path(args):
    args.store_dir = 'feddpg_' + str(args.num_clients) + '_' + str(args.dataset)
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

def server(args, logger):    
    # Load the dataset
    _, val_dataset, client_datasets= prepare_data(args, logger)
    
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
    indicator = Indicator()
    __init_client_model_store_path(args)
    # To store the grads for all rounds
    grads_all_round = []
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
            global_model_state, client_model_states = fl_round(args, logger, global_model, learning_round, client_datasets, selected_clients)
            print(global_model_state.keys())
            save_client_grads(grads_all_round, client_model_states, global_model)
            global_model.load_state(global_model_state)
            # Evaluate the model
            evaluate_global_model(args, logger, global_model, val_dataset, learning_round, indicator,before_unlearn=True)
            
            # Save the global model state
            global_model.save_state(args.output_file)
            logger.info(f"Model state saved to {args.output_file}")
    
    if args.mode in ['unlearning', 'both']:
        logger.info("#" * 80)
        logger.info(f"#{'':^78}#")
        logger.info(f"#{'Starting Federated Unlearning Round':^78}#")
        logger.info(f"#{'':^78}#")
        logger.info("#" * 80)

        client_dataset = client_datasets[args.unlearning_client_id]
        
        num_unlearn = int(len(client_dataset) * args.portion_unlearn)

        logger.info(f"Selected Client for Unlearning: Client {args.unlearning_client_id + 1}")
        logger.info(f"Number of data points to unlearn: {num_unlearn}")

            # Select data points to unlearn
        unlearn_indices = random.sample(range(len(client_dataset)), num_unlearn)
        logger.info(f"Data point indices to unlearn: {unlearn_indices}")
        
        # Get random clients to evaluate the model
        random_evaluation_clients = random.sample(range(args.num_clients), args.num_clients)
        
        # Evaluate the model before unlearning for clients' local data
        evaluate_client_model(
            args, 
            logger, 
            global_model, 
            client_datasets, 
            random_evaluation_clients, 
            before_unlearning = True,
            indicator = indicator,
        )
        
        # Evaluate the global model before unlearning
        evaluate_global_model(args, logger, global_model, val_dataset, r=-1, indicator = indicator, before_unlearn = True)
        
        original_prompt_generator = copy.deepcopy(global_model.prompt_generator)
        for ucid in random_evaluation_clients:
            # for client in range(args.num_clients):
            unlearnApi(
                logger,args,client_datasets,val_dataset,global_model,grads_all_round,args.num_rounds,ucid,indicator,
            )
            global_model.prompt_generator = original_prompt_generator
            # torch.load_state(original_prompt_generator)

    indicator.save(
        file_name=f'{args.dataset}_{args.model_name}_{args.num_rounds}_{args.num_clients}.json',
        formation='json',
    )
    
    logger.info("Done!")


def print_grads_round(grads_all_round):
    for round in grads_all_round:
        print(round)

def unlearnApi(
    logger, 
    args, 
    client_datasets, 
    val_dataset,
    global_model,
    grads_all_round, 
    cur_round,
    target_client,
    indicator,
):
    # if args.use_non_iid:
    #     global_model_state = fastFedUL_NonIID(
    #         args,
    #         logger,
    #         global_model.prompt_generator,
    #         target_client=target_client,
    #         round = cur_round,
    #         grads_all_round = grads_all_round,
    #         client_weights = None,
    #     )
    # # Perform federated unlearning round
    # else:
    global_model_state = fastFedULIID(
        args,
        logger,
        global_model.prompt_generator,
        target_client=target_client,
        round = cur_round,
        grads_all_round = grads_all_round,
        client_weights = None,
    )
    # global_model.prompt_generator.load_state(global_model_state)
    # global_model_state = fu_round(args, logger, global_model, client_datasets, unlearn_indices)
    global_model.load_state(global_model_state)
    
    # Evaluate the model after unlearning
    evaluate_client_model(
        args,
        logger, 
        global_model, 
        client_datasets, 
        target_client, 
        before_unlearning = False,
        indicator = indicator,
    )
    
    # Evaluate the global model
    evaluate_global_model(args, logger, global_model, val_dataset, indicator = indicator,before_unlearn = False,index = target_client)

def save_client_grads(grads_all_round, client_model_states, global_model):
    '''
    grads_all_round: [
        round: 
            client_changed :{}
    ],
    
    client_model_states : {
        cid: model_state,
    }
    
    global_model : torch model
    '''
    client_changed = {}
    global_state_dict = global_model.prompt_generator.state_dict()
    i = 0
    for cid, model_state in client_model_states.items():
        if i == 0:
            print(model_state.keys())
            i = 1
        tmp_difference = {}
        for key,param in model_state.items():
            tmp_difference[key] = param - global_state_dict[key]
            
        client_changed[cid] = tmp_difference
    
    # print(client_model_states[0].keys())
    
    grads_all_round.append(client_changed)