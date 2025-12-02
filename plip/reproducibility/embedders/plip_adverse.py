import sys
sys.path.insert(1, '/home/roba/Histopathology_Benchmark')

import clip
import tqdm
import numpy as np
import torch
from reproducibility.embedders.internal_datasets import *
from torch.utils.data import DataLoader
from reproducibility.utils.cacher import cache_hit_or_miss, cache_numpy_object, cache_hit_or_miss_raw_filename, cache_numpy_object_raw_filename
import open_clip
from open_clip import get_tokenizer
import torch.nn as nn

from src.zeroshot_utils import zeroshot_path
from transformers import AutoTokenizer
import torch.nn.functional as F
import CONCH.conch.open_clip_custom as concher
# from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer



'''
def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        orig_images = images.clone().detach().to(self.device)
 
        with torch.no_grad():
            orig_output = self.model.forward(orig_images)
        
        perturbations = torch.randn_like(images).uniform_(-self.eps, self.eps)

        adv_images = images + perturbations
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True
 
        outputs = self.model.forward(adv_images)
 
        # Calculate loss
        cost = nn.functional.cosine_similarity(orig_output, outputs)
        
 
        # Update adversarial images
        grad = torch.autograd.grad(
            -cost, adv_images, retain_graph=False, create_graph=False
        )[0]
 
        adv_images = adv_images + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
 
        return adv_images
'''


class AdversarialAttack:
    def __init__(self, model, epsilon=8, device="cuda"):

        """
        Initializes the adversarial attack class.
        
        Args:
            model (nn.Module): The model to attack. This should be an instance of a class with a defined `forward` method.
            epsilon (float): The perturbation magnitude for generating adversarial examples.
            device (str): The device to run the attack on (e.g., "cuda" or "cpu").
        """
        self.model = model
        self.epsilon = epsilon / 255.0  # Convert epsilon to the range [0, 1]
        self.device = device
        self.model.eval()  # Set the model to evaluation mode to avoid updating parameters
        self.count = 0
        self.alpha = 2 / 255.0

   

    def generate_adversarial_example(self, image, target_label=None):
        """
        Generates an adversarial image using the Fast Gradient Sign Method (FGSM).
        
        Args:
            image (torch.Tensor): The input image with shape (1, C, H, W).
            target_label (torch.Tensor, optional): The target label for a targeted attack. If None, the attack is untargeted.
        
        Returns:
            torch.Tensor: The adversarially perturbed image.
        """
        

        # Calculate min and max
        min_val = image.min()
        max_val = image.max()

        # Normalize the image to [0, 1]
        image = (image - min_val) / (max_val - min_val)
        # print(f"Image values: {image.min()} , {image.max()}")

        images = image.clone().detach().to(self.device)
        orig_images = image.clone().detach().to(self.device)
 
        with torch.no_grad():
            orig_output = self.model.forward(orig_images)
        
        perturbations = torch.randn_like(images).uniform_(-self.epsilon, self.epsilon)

        adv_images = images + perturbations
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True
 
        outputs = self.model.forward(adv_images)
 
        # Calculate loss
        cost = -nn.functional.cosine_similarity(orig_output, outputs)
        
        # print(f"Original loss ... {cost}")
        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]
 
        adv_images = adv_images + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        new_cost = -nn.functional.cosine_similarity(orig_output, self.model.forward(adv_images))
        # print(f"Loss after the perturbations ... {new_cost}")

        


        return adv_images

class AdverseCLIPEmbedder(nn.Module):

    def __init__(self, model, preprocess, name, backbone, ensemble, device, batch_size, num_workers):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.name = name
        self.backbone = backbone
        self.ensemble = ensemble
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def get_model(self):
        return self.model, self.preprocess
            
    def image_embedder(self, list_of_images, device="cuda", num_workers=1, batch_size=32, additional_cache_name=""):
        hit_or_miss = cache_hit_or_miss_raw_filename(self.name + "img" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_images(list_of_images, device=device, num_workers=num_workers, batch_size=batch_size)
            cache_numpy_object_raw_filename(hit, self.name + "img" + additional_cache_name, self.backbone)
            return hit

    def text_embedder(self, list_of_labels, device="cuda", num_workers=1, batch_size=32,ensemble_components=None, additional_cache_name=""):
        hit_or_miss = cache_hit_or_miss(self.name + "txt" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_text(list_of_labels, device=device, num_workers=num_workers, batch_size=batch_size, ensemble_components=ensemble_components)
            cache_numpy_object(hit, self.name + "txt" + additional_cache_name, self.backbone)
            return hit

    def embed_images(self, list_of_images, device="cuda", num_workers=1, batch_size=32):
        # Pass epsilon as input-space "in 255", e.g., 8 for 8/255
        attack = AdversarialAttack(self, epsilon=51, device=device)

        train_dataset = CLIPImageDataset(list_of_images, self.preprocess)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

        
        image_embeddings = []
        total = len(list_of_images) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        for images in dataloader:
            images = images.to(device)
            adv_images = []
            for image in images:
                adv_image = attack.generate_adversarial_example(image.unsqueeze(0))
                adv_images.append(adv_image)
            adv_images = torch.cat(adv_images)
            with torch.no_grad():
                embeddings = self.forward(adv_images)  # Use the custom forward method
                image_embeddings.extend(embeddings.detach().cpu().numpy())

                
                pbar.update(1)
            pbar.close()

        image_embeddings = np.array(image_embeddings)
        
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        return image_embeddings



    def forward(self,image):
        # print("in the forward function .... ")
        if self.name in ["clip", "quilt", "biomedclip"]:
            image_embeddings = self.model.encode_image(image) # for clip
        
        elif self.name == "plip":
            image_embeddings = self.model.model.get_image_features(image) # for plip
        
        elif "mi_zero" in self.name:
            image_embeddings = self.model.visual(image) # for mi-zero
        
        elif "conch" in self.name:
            image_embeddings = self.model.encode_image(image) # for mi-zero
        
        elif "keep" in self.name:
            image_embeddings = self.model.encode_image(image) # for keep
        
        elif "pathclip" in self.name:
            image_embeddings = self.model.encode_image(image) # for pathclip

        return image_embeddings
    
    
    
    def embed_text(self, list_of_labels, device="cuda", num_workers=1, batch_size=32, ensemble_components={}):
        train_dataset = CLIPCaptioningDataset(list_of_labels)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        text_embeddings = []
        total = len(list_of_labels) // batch_size

        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for captions in dataloader:
                ###### creating ensemble predictions
                if self.ensemble == True:
                    
                    classnames = ensemble_components['0']["classnames"]
                    templates = ensemble_components['0']["templates"]
                    
                    if self.name == "clip":
                        for caption in captions:
                            clss = caption.split("An H&E image patch of ")[-1].split(" tissue")[0]
                            zeroshot_weights = []
                            for classnames_for_class in classnames.keys():
                                embeddings_for_class = []
                                for classname in classnames[classnames_for_class]:
                                    texts = [template.replace("CLASSNAME", classname) for template in templates]
                                    token_ids = clip.tokenize(texts, truncate=True).to(device) # Tokenize with custom tokenizer
                                    classname_embeddings = self.model.encode_text(token_ids) # for clip
                                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
                                class_embedding = torch.stack(embeddings_for_class, dim=0)
                                class_embedding = class_embedding.mean(dim=(0, 1))
                                class_embedding /= class_embedding.norm()
                                zeroshot_weights.append(class_embedding.detach().cpu().numpy())
                        text_embeddings.extend(zeroshot_weights) # for clip




                    elif self.name == "plip":
                        for caption in captions:
                            clss = caption.split("An H&E image patch of ")[-1].split(" tissue")[0]
                            zeroshot_weights = []
                            for classnames_for_class in classnames.keys():
                                embeddings_for_class = []
                                for classname in classnames[classnames_for_class]:
                                    texts = [template.replace("CLASSNAME", classname) for template in templates]
                                    token_ids = clip.tokenize(captions, truncate=True).to(device)
                                    classname_embeddings = self.model.model.get_text_features(token_ids) # for plip
                                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
                                class_embedding = torch.stack(embeddings_for_class, dim=0)
                                class_embedding = class_embedding.mean(dim=(0, 1))
                                class_embedding /= class_embedding.norm()
                                zeroshot_weights.append(class_embedding.detach().cpu().numpy())
                        text_embeddings.extend(zeroshot_weights) # for plip


                    elif self.name == "quilt":
                        for caption in captions:
                            clss = caption.split("An H&E image patch of ")[-1].split(" tissue")[0]
                            zeroshot_weights = []
                            for classnames_for_class in classnames.keys():
                                embeddings_for_class = []
                                for classname in classnames[classnames_for_class]:
                                    texts = [template.replace("CLASSNAME", classname) for template in templates]
                                    tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
                                    token_ids = tokenizer(texts).to(device)
                                    classname_embeddings = self.model.encode_text(token_ids)
                                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1)) 
                                class_embedding = torch.stack(embeddings_for_class, dim=0)
                                class_embedding = class_embedding.mean(dim=(0, 1))
                                class_embedding /= class_embedding.norm()
                                zeroshot_weights.append(class_embedding.detach().cpu().numpy())
                        text_embeddings.extend(zeroshot_weights) # for clip

                    elif self.name == "biomedclip":
                        for caption in captions:
                            clss = caption.split("An H&E image patch of ")[-1].split(" tissue")[0]
                            zeroshot_weights = []
                            for classnames_for_class in classnames.keys():
                                embeddings_for_class = []
                                for classname in classnames[classnames_for_class]:
                                    texts = [template.replace("CLASSNAME", classname) for template in templates]
                                    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                                    token_ids = tokenizer(texts).to(device)
                                    classname_embeddings = self.model.encode_text(token_ids)
                                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
                                class_embedding = torch.stack(embeddings_for_class, dim=0)
                                class_embedding = class_embedding.mean(dim=(0, 1))
                                class_embedding /= class_embedding.norm()
                                zeroshot_weights.append(class_embedding.detach().cpu().numpy())
                        text_embeddings.extend(zeroshot_weights) # for clip
                    
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

                        for caption in captions:
                            clss = caption.split("An H&E image patch of ")[-1].split(" tissue")[0]
                            zeroshot_weights = []
                            for classnames_for_class in classnames.keys():
                                embeddings_for_class = []
                                for classname in classnames[classnames_for_class]:
                                    texts = [template.replace("CLASSNAME", classname) for template in templates]
                                    tokenizer = load_pretrained_tokenizer(encoder_name)
                                    n_texts, attention_mask = tokenize(tokenizer, texts)
                                    n_texts = torch.from_numpy(np.array(n_texts)).to(device)
                                    attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
                                    classname_embeddings = self.model.encode_text(n_texts, attention_mask=attention_mask)
                                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
                                class_embedding = torch.stack(embeddings_for_class, dim=0)
                                class_embedding = class_embedding.mean(dim=(0, 1))
                                class_embedding /= class_embedding.norm()
                                zeroshot_weights.append(class_embedding.detach().cpu().numpy())
                        text_embeddings.extend(zeroshot_weights) # for clip
            


                    elif "conch" in self.name:
                        for caption in captions:
                            clss = caption.split("An H&E image patch of ")[-1].split(" tissue")[0]
                            zeroshot_weights = []
                            for classnames_for_class in classnames.keys():
                                embeddings_for_class = []
                                for classname in classnames[classnames_for_class]:
                                    texts = [template.replace("CLASSNAME", classname) for template in templates]
                                    tokenizer = concher.get_tokenizer()
                                    token_ids = concher.tokenize(texts=texts, tokenizer=tokenizer).to(device)
                                    classname_embeddings = self.model.encode_text(token_ids) # for clip
                                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
                                class_embedding = torch.stack(embeddings_for_class, dim=0)
                                class_embedding = class_embedding.mean(dim=(0, 1))
                                class_embedding /= class_embedding.norm()
                                zeroshot_weights.append(class_embedding.detach().cpu().numpy())
                        text_embeddings.extend(zeroshot_weights) # for clip
                        



                if self.ensemble == False:
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

                       
                    elif self.name == "pathclip":

                        tokenizer = open_clip.get_tokenizer('ViT-B-16')
                        tokenized_prompts = tokenizer(texts=captions).to(device)
                        text_embeddings.extend(self.model.encode_text(tokenized_prompts).detach().cpu().numpy())


                    elif "keep" in self.name:

                        tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
                        tokenized_prompts = tokenizer(captions, max_length=256,padding='max_length',truncation=True, return_tensors='pt').to(device)
                        text_embeddings.extend(self.model.encode_text(tokenized_prompts).detach().cpu().numpy())    
                        
                    pbar.update(1)

                pbar.close()
        text_embeddings = np.array(text_embeddings)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        return text_embeddings

