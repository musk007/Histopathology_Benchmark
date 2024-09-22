source /apps/local/anaconda2023/conda_init.sh
conda activate quilt
### to be run in the path /home/roba.majzoub/research/new_plip/plip/reproducibilty/scripts
models=("clip" "plip" "quilt")
# Iterating over available models
# Reproducing zero-shot experiments

# for model in "${models[@]}";
#   do
  python3 scripts/fine_tuning_train.py --model_name $1 --dataset $2 
# done


