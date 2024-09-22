source /apps/local/anaconda2023/conda_init.sh
conda activate quilt
### to be run in the path /home/roba.majzoub/research/new_plip/plip/reproducibilty/scripts
models=("clip" "plip" "quilt" "biomedclip" "mi_zero_pubmedbert" "mi_zero_bioclinicalbert" "conch")
datasets=("Kather5K")
# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}";
  do
    for dataset in "${datasets[@]}";
      do
        python3 scripts/zero_shot_evaluation.py --model_name=$model --dataset $dataset
    done
done