import torch

def aggregate_model_states(model_states):
    if not model_states:    
        raise ValueError("No model states provided for aggregation")

    aggregated_state = {
        'prompt_generator': {},
        'classifier': {}
    }

    for component in ['prompt_generator', 'classifier']:
        reference_state = model_states[0][component]
        for key in reference_state.keys():
            stacked_params = torch.stack([state[component][key].to('cpu') for state in model_states])
            aggregated_state[component][key] = torch.mean(stacked_params, dim=0)

    return aggregated_state