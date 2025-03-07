#! /bin/bash
#SBATCH --job-name=benchmarker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_out/slurm-%N-%j.out
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos

models=("clip" "plip" "quilt" "biomedclip" "mi_zero_pubmedbert" "mi_zero_bioclinicalbert" "conch")

datasets=("CRC_100K")

# text_errors=("remove" "replace" "swap")
text_errors=("None")



# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for text_error in "${text_errors[@]}"; do
      python3 scripts/zero_shot_evaluation.py --model_name=$model --dataset=$dataset --text_error=$text_error 
    done
  done
done