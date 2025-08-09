# utils/model_utils.py

import torch
from model.feddpg import FedDPG
import pdb
def initialize_model(args):
    model = FedDPG(model_name=args.model_name, num_labels=args.num_labels, prompt_length=args.prompt_length)
    if args.checkpoint_path and not args.is_model_init:
        model.load_state(path=args.checkpoint_path)
        args.is_model_init = True
    model.to(args.device)
    # print(model)
    # pdb.set_trace()
    return model

# def load_checkpoint(model, checkpoint_path, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.prompt_generator.load_state_dict(checkpoint['prompt_generator'])
#     model.classifier.load_state_dict(checkpoint['classifier'])
#     return model
