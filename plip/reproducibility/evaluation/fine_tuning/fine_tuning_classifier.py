import sys
sys.path.insert(1, '/home/roba.majzoub/research/new_plip/plip')
import logging
from typing import List
from dotenv import load_dotenv
import argparse
import numpy as np
import logging
import os
import pandas as pd
from reproducibility.embedders.factory import EmbedderFactory
from reproducibility.fine_tuning import finetune
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def config():
    load_dotenv("/home/roba.majzoub/research/new_plip/plip/reproducibility/config_example.env")
    os.environ["PC_DEFAULT_BACKBONE"] = "/l/users/roba.majzoub/plip/pytorch_model.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str)
    parser.add_argument("--caption_column", default="text_style_4", type=str,
                        help="text_style_4 serves as the most intuitive prompt formulation for describing the image: An H&E image of XXX. On the other hand, text_style_0 simply acts as a categorical label for XXX.")
    parser.add_argument("--backbone", default='default', type=str)
    ##########################
    # parser.add_argument("--root", default="./", type=str) ########### Added
    parser.add_argument("--dataset", default="FinetuneDS", type=str) ########### Added
    parser.add_argument("--num_classes", default=2, type=int) ########### Added
    parser.add_argument("--lr", default=5e-5, type=float) ########### Added
    parser.add_argument("--weight_decay", default=0.2, type=float) ########### Added
    parser.add_argument("--warmup", default=0, type=int) ########### Added
    parser.add_argument("--optimizer", default="SGD", type=str) ########### Added
    ##########################
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)

    ## Probe hparams
    parser.add_argument("--alpha", default=0.01, type=float)
    return parser.parse_args()


if __name__ == "__main__":


    args = config()

    np.random.seed(args.seed)

    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]
    print(data_folder)

    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]

    finetune_dataset_csv = args.dataset + ".csv"
    data_path = os.path.join(data_folder, args.dataset)
    finetune_dataset = pd.read_csv(os.path.join(data_path, finetune_dataset_csv))
    # print("finetuner")
    finetuner = finetune.FineTuner(
        args,
        logging=logging,
        backbone=args.backbone,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # embedder = EmbedderFactory().factory(args)

    # test_x = embedder.image_embedder(test_dataset["image"].tolist(),
    #                                  additional_cache_name=finetune_dataset_name, batch_size=512)

    # labels = test_dataset["label"].unique().tolist()

    # # embeddings are generated using the selected caption, not the labels
    # test_y = embedder.text_embedder(test_dataset[args.caption_column].unique().tolist(),
    #                                 additional_cache_name=finetune_dataset_name, batch_size=512)

    # prober = ZeroShotClassifier()

    # results = prober.zero_shot_classification(test_x, test_y,
    #                                           unique_labels=labels, target_labels=test_dataset["label"].tolist(), model_name = args.model_name, ds_name = args.dataset)

    # additional_parameters = {'dataset': args.dataset, 'seed': args.seed,
    #                          'model': args.model_name, 'backbone': args.backbone}

    # rs = ResultsHandler(args.dataset, "zero_shot", additional_parameters)
    # rs.add(results)