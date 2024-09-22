import sys
sys.path.insert(1, '/home/roba.majzoub/research/new_plip/plip')
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np
import logging
from reproducibility.embedders.factory import EmbedderFactory
from reproducibility.evaluation.zero_shot.zero_shot import ZeroShotClassifier
import pandas as pd

from dotenv import load_dotenv
import os
from reproducibility.utils.results_handler import ResultsHandler
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def config():
    load_dotenv("/home/roba.majzoub/research/new_plip/plip/reproducibility/config_example.env")
    os.environ["PC_DEFAULT_BACKBONE"] = "/l/users/roba.majzoub/plip/pytorch_model.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str)
    parser.add_argument("--caption_column", default="text_style_4", type=str,
                        help="text_style_4 serves as the most intuitive prompt formulation for describing the image: An H&E image of XXX. On the other hand, text_style_0 simply acts as a categorical label for XXX.")
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--train_dataset", default="kather", type=str)
    parser.add_argument("--test_dataset", default="kather", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--finetune_test", default=True, type=bool)
    parser.add_argument("--checkpoint", default='./final_mode.pt', type=str)

    ## Probe hparams
    parser.add_argument("--alpha", default=0.01, type=float)
    return parser.parse_args()


if __name__ == "__main__":


    args = config()

    np.random.seed(args.seed)

    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]
    print(f"Testing {args.test_dataset} on {args.model_name} model trained on {args.train_dataset}\n","*"*50)

    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]

    test_dataset_name = args.test_dataset + "_test.csv"
    args.checkpoint = os.path.join("/l/users/roba.majzoub/plip_finetune",args.train_dataset,'train_ratio=1.0',args.model_name, "final_model.pt")

    test_dataset = pd.read_csv(os.path.join(data_folder, test_dataset_name))
    embedder = EmbedderFactory().factory(args)
    
    test_x = embedder.image_embedder(test_dataset["image"].tolist(),
                                     additional_cache_name=test_dataset_name, batch_size=512)

    labels = test_dataset["label"].unique().tolist()

    # embeddings are generated using the selected caption, not the labels
    test_y = embedder.text_embedder(test_dataset[args.caption_column].unique().tolist(),
                                    additional_cache_name=test_dataset_name, batch_size=512)

    prober = ZeroShotClassifier()

    results = prober.zero_shot_classification(test_x, test_y,
                                              unique_labels=labels, target_labels=test_dataset["label"].tolist(), model_name = args.model_name, train_ds_name = args.train_dataset, test_ds_name = args.test_dataset, finetune_test= args.finetune_test)

    additional_parameters = {'dataset': args.dataset, 'seed': args.seed,
                             'model': args.model_name, 'backbone': args.backbone}

    rs = ResultsHandler(args.dataset, "zero_shot", args.model_name, additional_parameters)
    rs.add(results)
