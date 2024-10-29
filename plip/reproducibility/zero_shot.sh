#! /bin/bash
#SBATCH --job-name=benchmarker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/roba.majzoub/benchmark/Histopathology_Benchmark/plip/reproducibility/slurm_out/slurm-%N-%j.out
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos

### to be run in the path /home/roba.majzoub/research/new_plip/plip/reproducibilty/scripts
# models=("clip" "plip" "quilt" "biomedclip" "mi_zero_pubmedbert" "mi_zero_bioclinicalbert" "conch")
models=("clip")
datasets=("Kather")
# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}";
  do
    for dataset in "${datasets[@]}";
      do
        python3 scripts/zero_shot_evaluation.py --model_name=$model --dataset $dataset
    done
done