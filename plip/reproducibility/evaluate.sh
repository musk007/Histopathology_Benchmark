source /apps/local/anaconda2023/conda_init.sh
conda activate quilt
### to be run in the path /home/roba.majzoub/research/new_plip/plip/reproducibilty/scripts


# evaluating fine-tuned models

models=("clip" "plip" "quilt")
# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}";
  do
    python3 scripts/evaluation_finetune.py --model_name=$model --finetune_test $1 --train_dataset $2 --test_dataset $3 
done

