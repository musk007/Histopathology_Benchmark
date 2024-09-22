import sys
sys.path.insert(1, '/home/roba.majzoub/research/new_plip/')
import clip
import tqdm
import numpy as np
import torch
from reproducibility.embedders.internal_datasets import *
from torch.utils.data import DataLoader
from reproducibility.utils.cacher import cache_hit_or_miss, cache_numpy_object, cache_hit_or_miss_raw_filename, cache_numpy_object_raw_filename
import open_clip
from open_clip import get_tokenizer

from src.zeroshot_utils import zeroshot_path
from transformers import AutoTokenizer
import torch.nn.functional as F
import CONCH.conch.open_clip_custom as concher
# from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

class CLIPEmbedder:

    def __init__(self, model, preprocess, name, backbone):
        self.model = model
        self.preprocess = preprocess
        self.name = name
        self.backbone = backbone
        
        
    def image_embedder(self, list_of_images, device="cuda", num_workers=1, batch_size=32, additional_cache_name=""):
        hit_or_miss = cache_hit_or_miss_raw_filename(self.name + "img" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_images(list_of_images, device=device, num_workers=num_workers, batch_size=batch_size)
            cache_numpy_object_raw_filename(hit, self.name + "img" + additional_cache_name, self.backbone)
            return hit

    def text_embedder(self, list_of_labels, device="cuda", num_workers=1, batch_size=32, additional_cache_name=""):
        hit_or_miss = cache_hit_or_miss(self.name + "txt" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_text(list_of_labels, device=device, num_workers=num_workers, batch_size=batch_size)
            cache_numpy_object(hit, self.name + "txt" + additional_cache_name, self.backbone)
            return hit

    def embed_images(self, list_of_images, device="cuda", num_workers=1, batch_size=32):
        train_dataset = CLIPImageDataset(list_of_images, self.preprocess)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

        image_embeddings = []

        total = len(list_of_images) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for images in dataloader:
                images = images.to(device)
                if self.name in ["clip", "quilt", "biomedclip"]:
                    image_embeddings.extend(self.model.encode_image(images).detach().cpu().numpy()) # for clip
                
                elif self.name == "plip":
                    image_embeddings.extend(self.model.model.get_image_features(images).detach().cpu().numpy()) # for plip
                
                elif "mi_zero" in self.name:
                    image_embeddings.extend(self.model.visual(images).detach().cpu().numpy()) # for mi-zero
                
                elif "conch" in self.name:
                    image_embeddings.extend(self.model.encode_image(images).detach().cpu().numpy()) # for mi-zero
                
                pbar.update(1)
            pbar.close()

        image_embeddings = np.array(image_embeddings)
        
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        return image_embeddings

    def embed_text(self, list_of_labels, device="cuda", num_workers=1, batch_size=32):
        train_dataset = CLIPCaptioningDataset(list_of_labels)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        text_embeddings = []
        total = len(list_of_labels) // batch_size

        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for captions in dataloader:
                if self.name == "quilt":
                    tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
                    idx = tokenizer(captions).to(device)
                    # idx = tokenizer.tokenize(captions, truncate=True).to(device)
                    text_embeddings.extend(self.model.encode_text(idx).detach().cpu().numpy()) # for quilt
                elif self.name == "clip":
                    idx = clip.tokenize(captions, truncate=True).to(device)
                    print(f"idx shape is : {idx.shape}")
                    text_embeddings.extend(self.model.encode_text(idx).detach().cpu().numpy()) # for clip
                        
                elif self.name == "plip":
                    idx = clip.tokenize(captions, truncate=True).to(device)
                    text_embeddings.extend(self.model.model.get_text_features(idx).detach().cpu().numpy()) # for plip
                
                elif self.name == "biomedclip":
                    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    idx = tokenizer(captions).to(device)
                    text_embeddings.extend(self.model.encode_text(idx).detach().cpu().numpy())
                
                elif "mi_zero" in self.name:
                    encoder_name = self.name.split("_")[-1]

                    def load_pretrained_tokenizer(encoder_name):
                        if 'clinicalbert' in encoder_name:
                            model_name = 'emilyalsentzer/Bio_ClinicalBERT'
                            tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
                        elif 'pubmed' in encoder_name:
                            model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                            tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
                        else:
                            raise NotImplementedError
                        
                        return tokenizer

                    def tokenize(tokenizer, texts):
                        tokens = tokenizer.batch_encode_plus(texts, 
                                                            max_length = 64,
                                                            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                                            return_token_type_ids=False,
                                                            truncation = True,
                                                            padding = 'max_length',
                                                            return_attention_mask=True)
                        return tokens['input_ids'], tokens['attention_mask']

                 
                    tokenizer = load_pretrained_tokenizer(encoder_name)
                    texts, attention_mask = tokenize(tokenizer, captions)
                 
                    texts = torch.from_numpy(np.array(texts)).to(device)
                    attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
                    class_embeddings = self.model.encode_text(texts, attention_mask=attention_mask)
                    ### The following were not used here since they are usually used for the WSI classification task
                    # class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    # class_embedding /= class_embedding.norm()
                    # idx = tokenizer(captions).to(device)
                  
                    text_embeddings.extend(class_embeddings.detach().cpu().numpy())

                elif "conch" in self.name:
                    # pass
                    tokenizer = concher.get_tokenizer()
                    tokenized_prompts = concher.tokenize(texts=captions, tokenizer=tokenizer).to(device)
                    text_embeddings.extend(self.model.encode_text(tokenized_prompts).detach().cpu().numpy())
                    
                    # text_embeddings.extend(self.model.encode_text(idx).detach().cpu().numpy())
                pbar.update(1)

            pbar.close()
        text_embeddings = np.array(text_embeddings)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        return text_embeddings


