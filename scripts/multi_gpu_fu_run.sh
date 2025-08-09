#!/bin/bash

# Array of all available datasets
# "mnli" is too large for GPU, so we exclude it
# "yelp" "agnews" can be added
all_datasets=("sst2" "cola" "rte" "mrpc")

# Default values
dev_mode=""
nproc_per_node=1
prompt_length=5
batch_size=32
num_rounds=10
num_clients=4
client_fraction=1
local_epochs=100
learning_rate=5e-5
datasets=()
mode="both"
unlearn_client=0
eval_dataset="mrpc"
checkpoint_file=""

# Summary log file
summary_log_file="logs/_experiments_summary.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev-mode)
            dev_mode="--dev-mode"
            shift
            ;;
        --dataset)
            datasets+=("$2")
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --unlearn-client)
            unlearn_client="$2"
            shift 2
            ;;
        --checkpoint-file)
            checkpoint_file="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to log experiment start
log_experiment_start() {
    local start_time=$1
    
    {
        echo "================================================================================"
        echo "Experiment Summary for Multi-Dataset Federated Learning"
        echo "================================================================================"
        echo "Start Time: $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
        echo "Hyperparameters:"
        echo "  Prompt Length: $prompt_length"
        echo "  Batch Size: $batch_size"
        echo "  Number of Rounds: $num_rounds"
        echo "  Number of Clients: $num_clients"
        echo "  Client Fraction: $client_fraction"
        echo "  Local Epochs: $local_epochs"
        echo "  Learning Rate: $learning_rate"
        echo "  Dev Mode: ${dev_mode:+Enabled}"
        echo "  Mode: $mode"
        echo "  Unlearn Client: $unlearn_client"
        echo "  Checkpoint File: ${checkpoint_file:-N/A}"
        echo "Status: Running"
    } >> "$summary_log_file"
}

# Function to log experiment end
log_experiment_end() {
    local start_time=$1
    local end_time=$2
    local duration=$((end_time - start_time))
    
    # Find the last occurrence of "Experiment Summary for Multi-Dataset Federated Learning"
    local line_number=$(tac "$summary_log_file" | grep -m 1 -n "Experiment Summary for Multi-Dataset Federated Learning" | cut -d: -f1)
    line_number=$(($(wc -l < "$summary_log_file") - line_number + 1))
    
    # Use sed to replace the "Status: Running" line and append end time and duration
    sed -i "${line_number},\$s/Status: Running/Status: Completed\nEnd Time: $(date -d @$end_time '+%Y-%m-%d %H:%M:%S')\nDuration: $(date -u -d @$duration '+%H:%M:%S')\n================================================================================/g" "$summary_log_file"
    
    echo "" >> "$summary_log_file"
}

# Function to run the experiment
run_experiment() {
    echo "Running multi-dataset federated learning experiment"
    start_time=$(date +%s)
    
    # Log experiment start
    log_experiment_start "$start_time"
    
    # Create a temporary file for output
    output_file=$(mktemp)
    
    echo "Running Python script..."
    python -m torch.distributed.launch --nproc_per_node=$nproc_per_node \
        adaptive_prompt_FU_multi_gpu.py \
        --prompt_length $prompt_length \
        --batch_size $batch_size \
        --num_rounds $num_rounds \
        --local_epochs $local_epochs \
        --learning_rate $learning_rate \
        --device "$device" \
        --mode $mode \
        --unlearn_client $unlearn_client \
        --eval_dataset $eval_dataset \
        ${checkpoint_file:+--checkpoint_file "$checkpoint_file"} \
        $dev_mode \
        2>&1 | tee $output_file
    
    end_time=$(date +%s)
    
    # Log experiment end
    log_experiment_end "$start_time" "$end_time"
    
    echo "Experiment completed"
    echo "----------------------------------------"
    echo "Python script output:"
    cat $output_file
    echo "----------------------------------------"
    
    # Clean up
    rm $output_file
}

# Ensure the logs directory exists
mkdir -p "$(dirname "$summary_log_file")"

# Run the experiment
run_experiment

echo "Experiment completed."
