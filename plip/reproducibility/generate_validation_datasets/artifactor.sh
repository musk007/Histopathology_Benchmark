#! /bin/bash
#SBATCH --job-name=artification # Job name
#SBATCH --output=/home/roba.majzoub/Histopathology_Benchmark/plip/reproducibility/generate_validation_datasets/slurm_out/output_.%A.txt # Standard output and error.
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=40G
#SBATCH --cpus-per-task=64 
#SBATCH --gres=gpu:1 
#SBATCH -p cscc-gpu-p 
#SBATCH --time=12:00:00 
#SBATCH -q cscc-gpu-qos 

python artifact_creator.py