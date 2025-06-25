#!/bin/bash
#SBATCH --job-name=adaptive_pfl_yelp
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --account=OD-234762

# Application specific commands:
module load cuda
module load miniconda3

cd ..

conda env create -f environment.yml
conda activate www

# run dev 
python adaptive_prompt_fl.py --dataset yelp --num_clients 100 --client_fraction 0.2 --num_rounds 10 --local_epochs 100 --prompt_length 10 --dev_mode True

# run full
# python adaptive_prompt_fl.py --dataset yelp --num_clients 100 --client_fraction 0.2 --num_rounds 10 --local_epochs 100 --prompt_length 10 