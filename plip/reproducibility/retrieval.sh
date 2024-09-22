source /apps/local/anaconda2023/conda_init.sh
conda activate quilt
### to be run in the path /home/roba.majzoub/research/new_plip/plip/reproducibilty/scripts
models=("clip" "plip" "quilt")
# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}";
  do
    python3 scripts/retrieval_evaluation.py --model_name=$model --dataset $1
done


