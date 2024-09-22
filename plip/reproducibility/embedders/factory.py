import sys
sys.path.insert(1, '/home/roba.majzoub/research/new_plip/')
import torch
import clip
from reproducibility.embedders.plip import CLIPEmbedder
from reproducibility.embedders.mudipath import build_densenet
from torchvision import transforms
from reproducibility.embedders.mudipath import DenseNetEmbedder
import os
from plip import PLIP
import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from src.models.factory import create_model 
import torch.nn as nn
import pandas as pd

import CONCH.conch.open_clip_custom as concher
import timm



_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

_IMAGE_TRANSFORM_TRAIN = [
    'pad_to_minimum',
    'horizontal_flip',
    'vertical_flip',
    'normalize'
]

_IMAGE_TRANSFORM_VAL = [
    'pad_to_minimum',
    'normalize'
]


class EmbedderFactory:

    def __init__(self):
        pass
    
    def factory(self, args):
        
        name = args.model_name
        path = args.backbone

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "plip":
            model = PLIP("vinid/plip")
            
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
            ])
            if args.finetune_test == True:
                model.model.load_state_dict(torch.load(args.checkpoint))
                print(f"loaded {args.model_name} checkpoint ..........\n")
              
            # model, preprocess = clip.load(os.environ["PC_CLIP_ARCH"], device=device)
            if device == 'cuda':
                # model.load_state_dict(torch.load(path))
                model.model.to(device)
            elif device == 'cpu':
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                
            model.model.eval()
            return CLIPEmbedder(model, preprocess, name, path)

        elif name == "clip":
            model, preprocess = clip.load(os.environ["PC_CLIP_ARCH"], device=device)
            model.eval()
            if args.finetune_test == True:
                model.load_state_dict(torch.load(args.checkpoint))
                print(f"loaded {args.model_name} checkpoint ..........\n")
           
            return CLIPEmbedder(model, preprocess, name, path)
        
        elif name == "quilt":

            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')

            if args.finetune_test == True:
                model.load_state_dict(torch.load(args.checkpoint))
                print(f"loaded {args.model_name} checkpoint ..........\n")
             
            model.to(device)
            model.eval()

            return CLIPEmbedder(model, preprocess_val, name, path)
        
        elif name == "biomedclip":

            model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            
            
            
            model.to(device)
            model.eval()

            return CLIPEmbedder(model, preprocess, name, path)
        
        elif name == "conch":
            # model = timm.create_model("hf_hub:MahmoodLab/CONCH", pretrained=True)
            model, preprocess = concher.create_model_from_pretrained('conch_ViT-B-16', "/l/users/roba.majzoub/models/conch/pytorch_model.bin")
            # preprocess = transforms.Compose([
            #     transforms.Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn),
            #     transforms.CenterCrop(size=(224, 224)),
            #     transforms.Lambda(lambda img: img.convert('RGB')),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            # ])
            
            
            
            
            model.to(device)
            model.eval()

            return CLIPEmbedder(model, preprocess, name, path)



      
        elif "mi_zero" in name:
            text_encoder = name.split("_")[-1]
            if text_encoder == "bioclinicalbert":
                encoder_name = "bioclinicalbert"
            elif text_encoder == "pubmedbert":
                encoder_name = "pubmedbert"

            ### using bioclinicalbert
            model_checkpoint = f"/l/users/roba.majzoub/models/MI-zero/ctranspath_448_{encoder_name}/checkpoints/epoch_50.pt"
            img_size = 224
            def clean_state_dict_ctranspath(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if 'attn_mask' in k:
                        continue
                    new_state_dict[k.replace('module.', '')] = v
                return new_state_dict
            def get_transforms_ctranspath(img_size=224, 
                              mean = (0.485, 0.456, 0.406), 
                              std = (0.229, 0.224, 0.225)):
                trnsfrms = transforms.Compose(
                                [
                                transforms.Resize(img_size),
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std)
                                ]
                            )
                return trnsfrms

            model_config = name
            model = create_model(model_config, device=device, override_image_size=None)
            if model_checkpoint is not None: # load PPTCLIP checkpoint if applicable
                if os.path.exists(model_checkpoint):
                    state_dict = torch.load(model_checkpoint, map_location='cpu')['state_dict']
                    state_dict = clean_state_dict_ctranspath(state_dict)
                    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
                    assert pd.Series(missing_keys).str.contains('attn_mask').all() # only modules with attn_mask are not loaded
                    print(f'Checkpoint {model_checkpoint} loaded successfully')
                else:
                    print(f'Cannot find model checkpoint {model_checkpoint}')
                    return 1
            
            preprocess = get_transforms_ctranspath(img_size = img_size)
            model.to(device)
            model.eval()
                


            return CLIPEmbedder(model, preprocess, name, path)


        elif name == "mudipath":
            backbone = build_densenet(download_dir="/oak/stanford/groups/jamesz/pathtweets/models/",
                                      pretrained="mtdp")  # TODO fixed path
            backbone.num_feats = backbone.n_features()
            backbone.forward_type = "image"
            backbone = backbone.to(device)
            backbone.eval()
            image_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
            ])
            return DenseNetEmbedder(backbone, image_preprocess, name, path)