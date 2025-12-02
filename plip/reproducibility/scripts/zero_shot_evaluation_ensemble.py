import sys
sys.path.insert(1, '/home/roba/Histopathology_Benchmark/plip')
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np
import logging
from reproducibility.embedders.factory import EmbedderFactory
from reproducibility.evaluation.zero_shot.zero_shot import ZeroShotClassifier
import pandas as pd
import json

from dotenv import load_dotenv
import os
from reproducibility.utils.results_handler import ResultsHandler
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import json

def config():
    load_dotenv("/home/roba/Histopathology_Benchmark/plip/reproducibility/config_example.env")

    os.environ["PC_DEFAULT_BACKBONE"] = "/l/users/roba.majzoub/plip/pytorch_model.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str)
    parser.add_argument("--caption_column", default="text_style_4", type=str,
                        help="text_style_4 serves as the most intuitive prompt formulation for describing the image: An H&E image of XXX. On the other hand, text_style_0 simply acts as a categorical label for XXX.")
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="kather", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--finetune_test", default=False, type=bool)
    parser.add_argument("--ensemble", default=True, type=bool)

    parser.add_argument("--adversarial", default=False, type=bool)
    parser.add_argument("--text_error", default='remove', type=str)   # options include remove,replace,swap

    ## Probe hparams
    parser.add_argument("--alpha", default=0.01, type=float)
    return parser.parse_args()


if __name__ == "__main__":


    args = config()
    if args.ensemble:
        mode = "ensemble"
    else:
        mode = "single caption"

    np.random.seed(args.seed)

    
    print(f"\nAdeversarial mode ........ {args.adversarial}")


    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]
    
    print("\n\n")

    print(f"Testing {args.dataset} on {args.model_name} model in {mode} mode\n","*"*50)

    print("\n\n")

    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]
    test_dataset_name = args.dataset + "_test.csv"
    if args.ensemble:
        f = open(os.path.join(data_folder, "text_files", args.dataset+"_ensemble.json"), 'r')
        args.ensemble_components = json.load(f)
        f.close()
    else:
        args.ensemble_components = {}

    
    test_dataset_name = args.dataset + "_test.csv"

    if args.ensemble:
        f = open(os.path.join(data_folder, "text_files", args.dataset+"_ensemble.json"), 'r')
        args.ensemble_components = json.load(f)
        f.close()
    else:
        args.ensemble_components = {}

        
    test_dataset = pd.read_csv(os.path.join(data_folder,"text_files", test_dataset_name))
    num_classes= len(test_dataset[args.caption_column].unique().tolist())
    embedder = EmbedderFactory().factory(args,num_classes)

    test_x = embedder.image_embedder(test_dataset["image"].tolist(),
                                     additional_cache_name=test_dataset_name, batch_size=args.batch_size)

    labels = test_dataset["label"].unique().tolist()
   

    # embeddings are generated using the selected caption, not the labels
    test_y = embedder.text_embedder(test_dataset[args.caption_column].unique().tolist(),
                                    additional_cache_name=test_dataset_name, batch_size=args.batch_size, ensemble_components=args.ensemble_components)

    prober = ZeroShotClassifier()
    model, preprocesses = embedder.get_model()
    grad_images = [test_dataset["image"].tolist()][:10]
    results = prober.zero_shot_classification(test_x, test_y,
                                              unique_labels=labels, target_labels=test_dataset["label"].tolist(), model_name = args.model_name, test_ds_name = args.dataset, preprocess=preprocesses, mode=mode, model=model, text_error=args.text_error, adversarial=args.adversarial)


    additional_parameters = {'dataset': args.dataset, 'seed': args.seed,
                             'model': args.model_name, 'backbone': args.backbone}

    rs = ResultsHandler(args.dataset, "zero_shot", args.model_name, additional_parameters)
    rs.add(results)

