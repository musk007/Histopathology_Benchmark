import sys
sys.path.insert(1, '/home/roba.majzoub/research/new_plip/plip')

from torch import nn
from torch import optim
import clip
import tqdm
import numpy as np
import torch
from reproducibility.embedders.internal_datasets import CLIPImageLabelDataset
from reproducibility.embedders.scheduler import cosine_lr
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from torch.cuda.amp import autocast
from dotenv import load_dotenv
import os
from plip import PLIP
from transformers import AutoProcessor
import open_clip
from torchvision import transforms
import json

from sklearn.metrics import f1_score

# Define a linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        # Convert input matrix to the same data type as self.weight
        x = x.to(self.fc.weight.dtype)
        out = self.fc(x)
        return out
        
def convert_models_to_fp32(model, model_name):
    if model_name == "plip":
        for p in model.model.parameters():
            p.data = p.data.float()
            if p.grad is not None:
                p.grad.data = p.grad.data.float()
    else:    
        for p in model.parameters():
            p.data = p.data.float()
            if p.grad is not None:
                p.grad.data = p.grad.data.float()

class FineTuner:

    def __init__(self,
                args=None,
                logging=None,
                backbone=None,
                num_classes=None,
                lr=5e-5,
                weight_decay=0.2,
                warmup=0,
                comet_tracking=None,
                comet_tags=None
                ):
        load_dotenv("/home/roba.majzoub/research/new_plip/plip/reproducibility/config_example.env")
        
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logging = logging
        self.save_directory = args.save_directory
        self.warmup = warmup
        self.hyper_params = {
            "lr": lr,
            "weight_decay": weight_decay
        }
        self.min_loss = 10000
        

        ##########################
        # Step 1. Model switch
        ##########################
        # Get preprocess regardless if it is CLIP backbone or not
        if args.model_name == "clip":
            self.model, self.preprocess = clip.load(os.environ["PC_CLIP_ARCH"], device="cuda")
        elif args.model_name == "plip":
            self.model = PLIP("vinid/plip")
            # processor = AutoProcessor.from_pretrained("vinid/plip")
            # print(processor)
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
            ])
        elif args.model_name == "quilt":

            self.model, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32') 
        # self.model, self.preprocess = clip.load(model_type,
        #                                         device=self.device,
        #                                         jit=False)  # Must set jit=False for training


        if self.args.model_name in ['plip', 'clip', 'quilt']:
            # TODO this is hard coded
            input_size = 512
            self.linear_classifier = LinearClassifier(input_size, num_classes)
            self.linear_classifier = self.linear_classifier.to(self.device)


            # parameters to be back-propagated.
            if args.model_name == "plip":
                bp_params = list(self.model.model.parameters()) + list(self.linear_classifier.parameters())
            else:
                bp_params = list(self.model.parameters()) + list(self.linear_classifier.parameters())

        elif self.args.model_name.startswith('resnet'):
            model_version = int(self.args.model_name.split('resnet')[1])
            self.model = None
            if model_version == 18:
                from torchvision.models import resnet18, ResNet18_Weights
                self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif model_version == 50:
                from torchvision.models import resnet50, ResNet50_Weights
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            elif model_version == 101:
                from torchvision.models import resnet101, ResNet101_Weights
                self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            # Modify the last fully connected layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.to(self.device)
            # parameters to be back-propagated.
            bp_params = self.model.parameters()

        elif self.args.model_name.startswith('vit'):
            self.model = None
            if self.args.model_name == 'vit_b_16':
                from torchvision.models import vit_b_16, ViT_B_16_Weights
                self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            elif self.args.model_name == 'vit_b_32':
                from torchvision.models import vit_b_32, ViT_B_32_Weights
                self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
            # Modify the last fully connected layer
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
            self.model.to(self.device)
            # parameters to be back-propagated.
            bp_params = self.model.parameters()
        else:
            raise Exception('No such model.')

        if self.device == "cpu":
            self.model.float()
        else:
            if self.args.model_name in ['clip', 'quilt']:
                clip.model.convert_weights(self.model)
                self.model.to(self.device)

            elif self.args.model_name == "plip":
                    for l in self.model.model.named_modules():
                        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                            l.weight.data = l.weight.data.half()
                            if l.bias is not None:
                                l.bias.data = l.bias.data.half()

                        if isinstance(l, nn.MultiheadAttention):
                            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                                tensor = getattr(l, attr)
                                if tensor is not None:
                                    tensor.data = tensor.data.half()

                        for name in ["text_projection", "proj"]:
                            if hasattr(l, name):
                                attr = getattr(l, name)
                                if attr is not None:
                                    attr.data = attr.data.half()
                    self.model.model.to(self.device)
        
        ##########################
        # Step 2. Optimizer
        ##########################
        
        self.classification_criterion = nn.CrossEntropyLoss()
        
        if self.args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=args.weight_decay)
        elif self.args.optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adagrad(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=args.weight_decay)
        elif self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=args.weight_decay)

        self.logging.info(f"Initialized finetuner")

    def _load_plip_checkpoint(self,
                                path=None,
                                ):
        if path is None:
            raise Exception('No path provided.')
        self.model.load_state_dict(torch.load(path))


    def calculate_f1_score(self, outputs, labels, average='weighted'):
        # Convert tensor outputs and labels to numpy arrays
        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        # Convert outputs to predicted labels by selecting the index of the maximum value
        predicted_labels = np.argmax(outputs, axis=1)
        # Calculate the F1 score
        f1 = f1_score(labels, predicted_labels, average=average)
        return f1

    def forward_pass(self, images):
        with torch.autocast(device_type = "cuda"):
            if self.args.model_name in ['clip', 'quilt']:
                image_features = self.model.encode_image(images)
                ### classifier
                outputs = self.linear_classifier(image_features)
            elif self.args.model_name == "plip":
                image_features = self.model.model.get_image_features(images)
                outputs = self.linear_classifier(image_features)
            else:
                with autocast():
                
                    outputs = self.model(images)
            return outputs


    def valid_evaluation(self, dataloader, pbar, pbar_description="Currently Validating"):
        valid_loss_this_epoch = 0
        
        outputs_list = []
        labels_list = []
        if self.args.model_name == "plip":
            self.model.model.eval()
        else:
            self.model.eval()
        if self.args.model_name in ['plip', 'clip', 'quilt']:
            self.linear_classifier.eval()

        for batch in dataloader:
            pbar.set_description(pbar_description)
            with torch.autocast(device_type = "cuda"):
                with torch.no_grad():
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self.forward_pass(images)
                    
                    # Append the output and label tensors to the lists
                    outputs_list.append(outputs)
                    labels_list.append(labels)

                    # Compute the loss
                    total_loss = self.classification_criterion(outputs, labels)
                    valid_loss_this_epoch += total_loss.cpu().data.numpy()

        # Concatenate output and label tensors
        outputs_all = torch.cat(outputs_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        f1_weighted = self.calculate_f1_score(outputs_all, labels_all, average='weighted')
        f1_macro = self.calculate_f1_score(outputs_all, labels_all, average='macro')
        if self.args.model_name =="plip":
            self.model.model.train()
        else:
            self.model.train()
        if self.args.model_name in ['plip', 'clip', 'quilt']:
            self.linear_classifier.train()

        return valid_loss_this_epoch, f1_weighted, f1_macro
    
        

    def tuner(self,
                train_dataframe,
                validation_dataframe,
                test_dataframe=None,
                save_directory="./",
                batch_size=4,
                epochs=5,
                evaluation_steps=500,
                num_workers=1
                ):

        start_time = str(datetime.now())


        # Regardless the model_type, we will use the same CLIP Image Label Dataset loader.
        train_dataset = CLIPImageLabelDataset(train_dataframe, self.preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        validation_dataset = CLIPImageLabelDataset(validation_dataframe, self.preprocess)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
        if test_dataframe is not None:
            test_dataset = CLIPImageLabelDataset(test_dataframe, self.preprocess)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        num_batches_per_epoch = len(train_dataloader)
        total_steps = len(train_dataloader) * epochs
        scheduler = cosine_lr(self.optimizer, self.hyper_params["lr"], self.warmup, total_steps)
        if self.args.model_name == "plip":
            self.model.model.train()
        else:
            self.model.train()
        # if self.args.model_name in ['plip', 'clip', 'quilt']:
        self.linear_classifier.train()

        performance_df = pd.DataFrame(index=np.arange(epochs), columns=['epoch','loss','f1_weighted','f1_macro'])
        for epoch in range(epochs):
            pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
            pbar.set_description(f"{epoch}/{epochs}")

            train_loss_this_epoch = 0
            for i, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                step = num_batches_per_epoch * epoch + i
                scheduler(step)

                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.forward_pass(images)
                

                # Compute the loss
                
                # Check if the tensor has one dimension
                if len(outputs.shape) == 1:
                    #print("Tensor has one dimension, unsqueeze it.")
                    outputs = outputs.unsqueeze(0)
                else:
                    pass
                total_loss = self.classification_criterion(outputs, labels)
                
                total_loss.backward()
                new_lr = scheduler(step)

                train_loss_this_epoch += total_loss.cpu().data.numpy()
                self.logging.info(f'[Train - this batch] epoch: {epoch}, batch: {i}, new learning rate: {new_lr}')
                #self.experiment.log_metric("learning_rate", new_lr, step=step)

                if self.device == "cpu":
                    self.optimizer.step() 
                else:
                    convert_models_to_fp32(self.model, self.args.model_name)
                    self.optimizer.step()
                    if self.args.model_name == "plip":
                        
                        for l in self.model.model.named_modules():
                            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                                l.weight.data = l.weight.data.half()
                                if l.bias is not None:
                                    l.bias.data = l.bias.data.half()

                            if isinstance(l, nn.MultiheadAttention):
                                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                                    tensor = getattr(l, attr)
                                    if tensor is not None:
                                        tensor.data = tensor.data.half()

                            for name in ["text_projection", "proj"]:
                                if hasattr(l, name):
                                    attr = getattr(l, name)
                                    if attr is not None:
                                        attr.data = attr.data.half()
                        # sys.exit()
                        # clip.model.convert_weights(self.model.model)
                    else:
                        clip.model.convert_weights(self.model)
                pbar.update(1)

                if evaluation_steps == 0:
                    pass
                else:
                    if step % evaluation_steps == 0:
                        valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(validation_dataloader, pbar, pbar_description="Currently Validating")
                        pbar.set_description(f"{epoch}/{epochs}")
                        self.logging.info(f'[Validation - this batch] epoch: {epoch}, batch: {i}, total loss: {valid_loss_this_epoch}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}')

            self.logging.info(f'[Train - final] epoch: {epoch}, total loss: {train_loss_this_epoch}')

            # Validation at the end of each epoch
            valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(validation_dataloader, pbar, pbar_description="Currently Validating")
            pbar.set_description(f"{epoch}/{epochs}")
            self.logging.info(f'[Validation - final] epoch: {epoch}, total loss: {valid_loss_this_epoch}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}')

            performance_df.loc[epoch, 'epoch'] = epoch
            performance_df.loc[epoch, 'loss'] = valid_loss_this_epoch
            performance_df.loc[epoch, 'f1_weighted'] = f1_weighted
            performance_df.loc[epoch, 'f1_macro'] = f1_macro
            if epoch == (epochs - 1):
                # self.min_loss = valid_loss_this_epoch
                # print("model saved ... ")
                # print("current loss is ", self.min_loss)
                if self.args.model_name == "plip":
                    torch.save(self.model.model.state_dict(), f"{save_directory}/epoch_{epoch}_{start_time}_{self.args.model_name}.pt")
                else:
                    torch.save(self.model.state_dict(), f"{save_directory}/epoch_{epoch}_{start_time}_{self.args.model_name}.pt")
               
                
                performance_df_path = os.path.join(self.save_directory,f"{self.args.model_name}_performance_at_{self.args.dataset}_{epoch}.csv")
                performance_df.to_csv(performance_df_path, index=False)
               

            # This is just for debug only:
            # TODO: remove it.
            if test_dataframe is not None:
                valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(test_dataloader, pbar, pbar_description="Currently Testing")
                performance_df.loc[epoch, 'f1_test_weighted'] = f1_weighted
                performance_df.loc[epoch, 'f1_test_macro'] = f1_macro


            pbar.close()

        performance_df['f1_weighted'] = performance_df['f1_weighted'].astype(float)
        performance_df['f1_macro'] = performance_df['f1_macro'].astype(float)
        performance_df_path = os.path.join(self.save_directory,f"{self.args.model_name}_overall_performance_at_{self.args.dataset}_{epoch}.csv")
        performance_df.to_csv(performance_df_path, index=False)
        print("*"*100)
        print(f"saved performance df in {performance_df_path}")
        print("*"*100)
        return performance_df
    