# python get_results_utils.py --dir_path logs/ --mode learning


import os
import re
import csv
import argparse

def extract_validation_metrics(log_file_path, mode):
    # Prepare the CSV file name
    csv_file_path = log_file_path.replace('.log', f'_{mode}.csv')
    
    with open(log_file_path, 'r') as log_file, open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        if mode == 'learning':
            writer.writerow(['round', 'accuracy'])  # Write header for learning
            for line in log_file:
                # Match the line containing validation accuracy for learning
                match = re.search(r'Round (\d+)/\d+ - Validation accuracy: ([\d.]+)', line)
                if match:
                    round_num = match.group(1)
                    accuracy = match.group(2)
                    writer.writerow([round_num, accuracy])  # Write data
        elif mode == 'unlearning':
            writer.writerow(['client', 'accuracy_before', 'accuracy_after'])  # Write header for unlearning
            client_data = {}
            global_accuracy_before = None
            global_accuracy_after = None
            
            for line in log_file:
                # Match the line containing client validation accuracy before unlearning
                before_match = re.search(r'Performance Impact - Before Unlearning - Client (\d+) Validation accuracy: ([\d.]+)', line)
                if before_match:
                    client_id = before_match.group(1)
                    accuracy_before = before_match.group(2)
                    client_data[client_id] = {'before': accuracy_before, 'after': None}
                
                # Match the line containing client validation accuracy after unlearning
                after_match = re.search(r'Performance Impact - After Unlearning - Client (\d+) Validation accuracy: ([\d.]+)', line)
                if after_match:
                    client_id = after_match.group(1)
                    accuracy_after = after_match.group(2)
                    if client_id in client_data:
                        client_data[client_id]['after'] = accuracy_after
                
                # Match the line containing global model validation accuracy before unlearning
                global_before_match = re.search(r'Global Model Validation accuracy Before Unlearning: ([\d.]+)', line)
                if global_before_match:
                    global_accuracy_before = global_before_match.group(1)
                
                # Match the line containing global model validation accuracy after unlearning
                global_after_match = re.search(r'Global Model Validation accuracy After Unlearning: ([\d.]+)', line)
                if global_after_match:
                    global_accuracy_after = global_after_match.group(1)
            
            # Write client data to CSV
            for client_id, accuracies in client_data.items():
                writer.writerow([client_id, accuracies['before'], accuracies['after']])
            
            # Write global model accuracy to CSV
            if global_accuracy_before is not None and global_accuracy_after is not None:
                writer.writerow(['global', global_accuracy_before, global_accuracy_after])

def process_directory(directory, mode):
    for filename in os.listdir(directory):
        if filename.endswith('.log') and filename != "_experiments_summary.log":
            log_file_path = os.path.join(directory, filename)
            extract_validation_metrics(log_file_path, mode)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--dir_path', required=True, help='Path to the folder containing log files')
    parser.add_argument('--mode', required=True, choices=['learning', 'unlearning'], help='Mode of extraction: learning or unlearning')
    args = parser.parse_args()
    dir_path = args.dir_path
    mode = args.mode
    process_directory(dir_path, mode)
