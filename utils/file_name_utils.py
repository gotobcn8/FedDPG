import os
from datetime import datetime
from config import DEFAULT_CONFIG
  
# Function to generate filenames for logs and checkpoints
def generate_filename(args, file_type):
    # Ensure directories exist
    os.makedirs(DEFAULT_CONFIG["log_dir"], exist_ok=True)
    os.makedirs(DEFAULT_CONFIG["checkpoint_dir"], exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dev" if args.dev_mode else "full"
    filename = f"{timestamp}_{args.dataset}_{mode}_c{args.num_clients}_f{args.client_fraction}_r{args.num_rounds}_e{args.local_epochs}_p{args.prompt_length}"
    
    match file_type:
        case "log":
            return os.path.join(DEFAULT_CONFIG["log_dir"], f"fed_dgp_{filename}.log")
        case "checkpoint":
            return os.path.join(DEFAULT_CONFIG["checkpoint_dir"], f"fed_dgp_{filename}.pt")
        case "unlearned_checkpoint":
            return os.path.join(DEFAULT_CONFIG["checkpoint_dir"], f"fed_dgp_{filename}_unlearned.pt")
        case _:
            raise ValueError(f"Invalid file type: {file_type}")
    