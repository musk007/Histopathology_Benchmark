#! /bin/bash
#SBATCH --job-name=benchmarker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/roba.majzoub/Histopathology_Benchmark/plip/reproducibility/slurm_out/slurm-%N-%j.out
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos

### to be run in the path /home/roba.majzoub/research/new_plip/plip/reproducibilty/scripts
# models=("clip")
models=("conch")
# models=("clip" "plip" "quilt" "biomedclip" "mi_zero_pubmedbert" "mi_zero_bioclinicalbert" "conch")
# models=("clip" "plip")
datasets=("l4_thread")

# datasets=("TCGA_Uniform_Fold_1" "TCGA_Uniform_Fold_2" \
#         "TCGA_Uniform_Fold_3" "TCGA_Uniform_Fold_4" "TCGA_Uniform_Fold_5" "WSSS4LUAD")

# text_errors=("remove" "replace" "swap")
text_errors=("None")


# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for text_error in "${text_errors[@]}"; do
      # for adverse in "${adversarial[@]}"; do
        # python3 scripts/zero_shot_evaluation.py --model_name=$model --dataset=$dataset --text_error=$text_error --adversarial=$adverse
      python3 scripts/zero_shot_evaluation.py --model_name=$model --dataset=$dataset --text_error=$text_error 
      # done
    done
  done
done