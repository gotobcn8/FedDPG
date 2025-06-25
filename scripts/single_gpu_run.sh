#!/bin/bash

# Array of all available datasets
#  "cola" "rte" "mrpc" "mnli"  can be added
all_datasets=("agnews" "sst2" "yelp")

# Default values
dev_mode=""
model_name="roberta-base" # "xlm-roberta-large"
prompt_lengths=(1 5 10)
batch_size=256
num_rounds=100
num_clients=100
client_fractions=(0.1 0.2 0.5)
local_epochs=5
learning_rate=5e-5
datasets=()
use_non_iid=""
alpha_split=1.0
alpha_device=5.0
gpu_id=0

# Summary log file
summary_log_file=""

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
        --prompt-length)
            prompt_lengths=("$2")
            shift 2
            ;;
        --client-fraction)
            client_fractions=("$2")
            shift 2
            ;;
        --use-non-iid)
            use_non_iid="--use_non_iid"
            shift
            ;;
        --alpha-split)
            alpha_split="$2"
            shift 2
            ;;
        --alpha-device)
            alpha_device="$2"
            shift 2
            ;;
        --gpu)
            gpu_id="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If no datasets were specified, use all datasets
if [ ${#datasets[@]} -eq 0 ]; then
    datasets=("${all_datasets[@]}")
fi

# Function to log experiment start
log_experiment_start() {
    local dataset=$1
    local prompt_length=$2
    local client_fraction=$3
    local start_time=$4
    
    {
        echo "================================================================================"
        echo "Experiment Summary for Dataset: $dataset, Prompt Length: $prompt_length, Client Fraction: $client_fraction"
        echo "================================================================================"
        echo "Start Time: $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
        echo "Hyperparameters:"
        echo "  LLM: $model_name"
        echo "  Prompt Length: $prompt_length"
        echo "  Batch Size: $batch_size"
        echo "  Number of Rounds: $num_rounds"
        echo "  Number of Clients: $num_clients"
        echo "  Client Fraction: $client_fraction"
        echo "  Local Epochs: $local_epochs"
        echo "  Learning Rate: $learning_rate"
        echo "  Dev Mode: ${dev_mode:+Enabled}"
        echo "  Non-IID: ${use_non_iid:+Enabled}"
        if [ -n "$use_non_iid" ]; then
            echo "  Alpha Split: $alpha_split"
            echo "  Alpha Device: $alpha_device"
        fi
        echo "Status: Running"
    } >> "$summary_log_file"
}

# Function to log experiment end
log_experiment_end() {
    local dataset=$1
    local prompt_length=$2
    local client_fraction=$3
    local start_time=$4
    local end_time=$5
    local duration=$((end_time - start_time))
    
    # Find the last occurrence of the specific experiment summary
    local line_number=$(tac "$summary_log_file" | grep -m 1 -n "Experiment Summary for Dataset: $dataset, Prompt Length: $prompt_length, Client Fraction: $client_fraction" | cut -d: -f1)
    line_number=$(($(wc -l < "$summary_log_file") - line_number + 1))
    
    # Use sed to replace the "Status: Running" line and append end time and duration
    sed -i "${line_number},\$s/Status: Running/Status: Completed\nEnd Time: $(date -d @$end_time '+%Y-%m-%d %H:%M:%S')\nDuration: $(date -u -d @$duration '+%H:%M:%S')\n================================================================================/g" "$summary_log_file"
    
    echo "" >> "$summary_log_file"
}

# Function to run the experiment
run_experiment() {
    dataset=$1
    prompt_length=$2
    client_fraction=$3
    
    # Update the summary log file name to include the dataset
    summary_log_file="logs/_experiments_summary_${dataset}.log"
    
    echo "Running experiment for dataset: $dataset, prompt length: $prompt_length, client fraction: $client_fraction"
    start_time=$(date +%s)
    
    # Log experiment start
    log_experiment_start "$dataset" "$prompt_length" "$client_fraction" "$start_time"

    python adaptive_prompt_fl.py \
        --dataset "$dataset" \
        --model_name "$model_name" \
        --prompt_length "$prompt_length" \
        --batch_size "$batch_size" \
        --num_rounds "$num_rounds" \
        --num_clients "$num_clients" \
        --client_fraction "$client_fraction" \
        --local_epochs "$local_epochs" \
        --learning_rate "$learning_rate" \
        --device "cuda:$gpu_id" \
        $dev_mode \
        $use_non_iid \
        --alpha-split "$alpha_split" \
        --alpha_device "$alpha_device"
    
    end_time=$(date +%s)
    
    # Log experiment end
    log_experiment_end "$dataset" "$prompt_length" "$client_fraction" "$start_time" "$end_time"
    
    echo "Experiment completed for dataset: $dataset, prompt length: $prompt_length, client fraction: $client_fraction"
    echo "----------------------------------------"
}

# Ensure the logs directory exists
mkdir -p "$(dirname "$summary_log_file")"

# Log information about multiple values
{
    echo "================================================================================"
    echo "Experiment Configuration Summary"
    echo "================================================================================"
    echo "Datasets to be tested: ${datasets[*]}"
    echo "Prompt lengths to be tested: ${prompt_lengths[*]}"
    echo "Client fractions to be tested: ${client_fractions[*]}"
    echo "Total number of experiments: $((${#datasets[@]} * ${#prompt_lengths[@]} * ${#client_fractions[@]}))"
    echo "Non-IID: ${use_non_iid:+Enabled}"
    if [ -n "$use_non_iid" ]; then
        echo "Alpha Split: $alpha_split"
        echo "Alpha Device: $alpha_device"
    fi
    echo "================================================================================"
    echo ""
} >> "$summary_log_file"

# Run experiments for each combination of dataset, prompt_length, and client_fraction
for dataset in "${datasets[@]}"; do
    # Update the summary log file name for each dataset
    summary_log_file="logs/_experiments_summary_${dataset}.log"
    
    # Ensure the logs directory exists
    mkdir -p "$(dirname "$summary_log_file")"
    
    # Log information about multiple values
    {
        echo "================================================================================"
        echo "Experiment Configuration Summary for $dataset"
        echo "================================================================================"
        echo "Prompt lengths to be tested: ${prompt_lengths[*]}"
        echo "Client fractions to be tested: ${client_fractions[*]}"
        echo "Total number of experiments: $((${#prompt_lengths[@]} * ${#client_fractions[@]}))"
        echo "Non-IID: ${use_non_iid:+Enabled}"
        if [ -n "$use_non_iid" ]; then
            echo "Alpha Split: $alpha_split"
            echo "Alpha Device: $alpha_device"
        fi
        echo "================================================================================"
        echo ""
    } >> "$summary_log_file"
    
    for prompt_length in "${prompt_lengths[@]}"; do
        for client_fraction in "${client_fractions[@]}"; do
            run_experiment $dataset $prompt_length $client_fraction
        done
    done
done

echo "All experiments completed."