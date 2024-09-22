source /apps/local/anaconda2023/conda_init.sh
conda activate quilt

python3 scripts/zero_shot_evaluation.py --model_name="biomedclip" --dataset $1



