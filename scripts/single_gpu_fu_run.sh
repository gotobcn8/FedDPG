#!/bin/bash

# Array of all available datasets
#  "cola" "rte" "mrpc" "mnli"  can be added
all_datasets=("agnews") #  "sst2" "yelp"

# Default values
dev_mode=""
model_name="roberta-base" # "xlm-roberta-large"
prompt_length=10
batch_size=256
num_rounds=10
num_clients=20
client_fraction=0.1
local_epochs=20
learning_rate=5e-5
datasets=()
use_non_iid=""
alpha_split=1.0
alpha_device=5.0
gpu_id=0
unlearning_client_id=0
portion_unlearn=(0.1) # 0.05 0.1 0.2  # Multiple values for portion_unlearn
metric="accuracy"
seed=42
num_eval_clients=5
checkpoint_path="checkpoints/12_final_agnews/adaptive_prompt_fl_agnews_full_c100_f0.1_r100_e5_p10_20241007_001944.pt"

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
            prompt_length="$2"
            shift 2
            ;;
        --client-fraction)
            client_fraction="$2"
            shift 2
            ;;
        --use-non-iid)
            use_non_iid="--use-non-iid"
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
        --unlearning-client-id)
            unlearning_client_id="$2"
            shift 2
            ;;
        --portion-unlearn)
            IFS=',' read -r -a portion_unlearn <<< "$2"
            shift 2
            ;;
        --metric)
            metric="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --num-eval-clients)
            num_eval_clients="$2"
            shift 2
            ;;
        --checkpoint-path)
            checkpoint_path="$2"
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
    local portion=$4
    local start_time=$5
    
    {
        echo "================================================================================"
        echo "Experiment Summary for Dataset: $dataset, Prompt Length: $prompt_length, Client Fraction: $client_fraction, Portion Unlearn: $portion"
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
        echo "  Unlearning Client ID: $unlearning_client_id"
        echo "  Portion of Data to Unlearn: $portion"
        echo "  Metric: $metric"
        echo "  Seed: $seed"
        echo "  Number of Evaluation Clients: $num_eval_clients"
        echo "  Checkpoint Path: ${checkpoint_path:-None}"
        echo "Status: Running"
    } >> "$summary_log_file"
}

# Function to log experiment end
log_experiment_end() {
    local dataset=$1
    local prompt_length=$2
    local client_fraction=$3
    local portion=$4
    local start_time=$5
    local end_time=$6
    local duration=$((end_time - start_time))
    
    # Find the last occurrence of the specific experiment summary
    local line_number=$(tac "$summary_log_file" | grep -m 1 -n "Experiment Summary for Dataset: $dataset, Prompt Length: $prompt_length, Client Fraction: $client_fraction, Portion Unlearn: $portion" | cut -d: -f1)
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
    portion=$4
    
    # Update the summary log file name to include the dataset
    summary_log_file="logs/_unlearning_experiments_summary_${dataset}.log"
    
    echo "Running experiment for dataset: $dataset, prompt length: $prompt_length, client fraction: $client_fraction, portion unlearn: $portion"
    start_time=$(date +%s)
    
    # Log experiment start
    log_experiment_start "$dataset" "$prompt_length" "$client_fraction" "$portion" "$start_time"

    python fed_dpg_unlearning.py \
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
        --use-non-iid \
        --alpha-split "$alpha_split" \
        --alpha_device "$alpha_device" \
        --unlearning_client_id "$unlearning_client_id" \
        --portion_unlearn "$portion" \
        --metric "$metric" \
        --seed "$seed" \
        --num_eval_clients "$num_eval_clients" \
        --checkpoint_path "$checkpoint_path"
    
    end_time=$(date +%s)
    
    # Log experiment end
    log_experiment_end "$dataset" "$prompt_length" "$client_fraction" "$portion" "$start_time" "$end_time"
    
    echo "Experiment completed for dataset: $dataset, prompt length: $prompt_length, client fraction: $client_fraction, portion unlearn: $portion"
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
    echo "Prompt length to be tested: $prompt_length"
    echo "Client fraction to be tested: $client_fraction"
    echo "Portions to be unlearned: ${portion_unlearn[*]}"
    echo "Total number of experiments: $((${#datasets[@]} * ${#portion_unlearn[@]}))"
    echo "Non-IID: ${use_non_iid:+Enabled}"
    if [ -n "$use_non_iid" ]; then
        echo "Alpha Split: $alpha_split"
        echo "Alpha Device: $alpha_device"
    fi
    echo "================================================================================"
    echo ""
} >> "$summary_log_file"

# Run experiments for each combination of dataset and portion_unlearn
for dataset in "${datasets[@]}"; do
    # Update the summary log file name for each dataset
    summary_log_file="logs/_unlearning_experiments_summary_${dataset}.log"
    
    # Ensure the logs directory exists
    mkdir -p "$(dirname "$summary_log_file")"
    
    # Log information about multiple values
    {
        echo "================================================================================"
        echo "Experiment Configuration Summary for $dataset"
        echo "================================================================================"
        echo "Prompt length to be tested: $prompt_length"
        echo "Client fraction to be tested: $client_fraction"
        echo "Portions to be unlearned: ${portion_unlearn[*]}"
        echo "Total number of experiments: ${#portion_unlearn[@]}"
        echo "Non-IID: ${use_non_iid:+Enabled}"
        if [ -n "$use_non_iid" ]; then
            echo "Alpha Split: $alpha_split"
            echo "Alpha Device: $alpha_device"
        fi
        echo "================================================================================"
        echo ""
    } >> "$summary_log_file"
    
    for portion in "${portion_unlearn[@]}"; do
        run_experiment $dataset $prompt_length $client_fraction $portion
    done
done

echo "All experiments completed."