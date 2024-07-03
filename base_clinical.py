
"""
2値分類

"""
import hydra
from omegaconf import DictConfig, OmegaConf
import sys,gc,os,random,time,math,glob
import matplotlib.pyplot as plt
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
from  torch.cuda.amp import autocast, GradScaler 
import timm
import cv2

from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.metrics import log_loss
from functools import partial
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,f1_score,log_loss
from  sklearn.metrics import accuracy_score as acc
import torch
import torch.nn as nn
from torch.optim import Adam, SGD,AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,RandomGamma, RandomRotate90,GaussNoise,Cutout,RandomBrightnessContrast,RandomContrast,Resize
from albumentations.pytorch import ToTensorV2
from timm.utils.agc import adaptive_clip_grad
import transformers as T
import albumentations as A

from torch.optim.swa_utils import AveragedModel, SWALR

### my utils
from code_factory.pooling import GeM,AdaptiveConcatPool2d
from code_factory.augmix import RandomAugMix
from code_factory.gridmask import GridMask
from code_factory.fmix import *
from code_factory.loss_func import *
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold

###

all_df = pd.read_csv("/home/abe/KidneyM/hubmap2021/MATUI_bbxo/final_pas.csv")

all_df = all_df[all_df["DM"]!=-1].reset_index(drop=True)
print(all_df.shape)


all_df["label"] =  all_df["DM"] 
print(all_df["label"].value_counts())
print(all_df["WSI"].nunique())
print(all_df.groupby("label")["WSI"].nunique())


print(all_df.shape)

n_target = all_df["label"].nunique()
print("n_target",n_target)
N_fold = 5



sgkf = StratifiedGroupKFold(n_splits=N_fold,random_state=2021,shuffle=True)
for fold, ( _, val_) in enumerate(sgkf.split(X=all_df, y=all_df.label.to_numpy(),groups=all_df.WSI)):
    all_df.loc[val_ , "fold"] = fold
    
    val_df = all_df[all_df["fold"]==fold]
        #print(val_df["label"].nunique())








import logging
#from mylib.
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


from timm.data.transforms import RandomResizedCropAndInterpolation


#### dataset ==============
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df,train=True,transform1=None):
        self.paths = df["path"].to_numpy()
        self.labels = df["label"].to_numpy()
        self.transform = transform1

        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]#+".jpg"
        image = cv2.imread(file_path)[:,:,::-1]
        
        image = self.transform(image=image)["image"]
        image = torch.from_numpy(image.transpose((2,0,1)))

        label = torch.tensor(self.labels[idx]).long()
        return image,label




#### dataset ==============

#### augmentation ==============


def get_transforms(*, data,CFG):
    if data == 'train':
        return Compose([
            #A.augmentations.crops.transforms.CenterCrop(1024*0.8,1024*0.8),
            A.crops.transforms.RandomResizedCrop(CFG.preprocess.size,CFG.preprocess.size),
            #A.crops.transforms.RandomCrop(CFG.preprocess.size,CFG.preprocess.size),
            #Resize(512,512),
            #A.crops.transforms.RandomResizedCrop(448,448),
            A.HorizontalFlip(p=CFG.aug.HorizontalFlip.p),
            A.VerticalFlip(p=CFG.aug.VerticalFlip.p),
            A.RandomRotate90(p=CFG.aug.RandomRotate90.p),
            A.ShiftScaleRotate(
                shift_limit=CFG.aug.ShiftScaleRotate.shift_limit,
                scale_limit=CFG.aug.ShiftScaleRotate.scale_limit,
                rotate_limit=CFG.aug.ShiftScaleRotate.rotate_limit,
                p=CFG.aug.ShiftScaleRotate.p),
            A.RandomBrightnessContrast(
                brightness_limit=CFG.aug.RandomBrightnessContrast.brightness_limit,
                contrast_limit=CFG.aug.RandomBrightnessContrast.contrast_limit,
                p=CFG.aug.RandomBrightnessContrast.p),
            A.CLAHE(
                clip_limit=(1,4),
                p=CFG.aug.CLAHE.p),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3)], p=CFG.aug.one_of_Distortion.p),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(),
                ], p=CFG.aug.one_of_Blur_Gnoise.p),
            #Resize(CFG.preprocess.size,CFG.preprocess.size),
            A.OneOf([
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=CFG.aug.compress.p),
            GridMask(
                num_grid=CFG.aug.GridMask.num_grid,p=CFG.aug.GridMask.p),
            A.CoarseDropout(max_holes=CFG.aug.CoarseDropout.max_holes, max_height=CFG.aug.CoarseDropout.max_height, max_width=CFG.aug.CoarseDropout.max_width, p=CFG.aug.CoarseDropout.p),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'valid':
        return Compose([
            #Resize(512,512),
            Resize(CFG.preprocess.size,CFG.preprocess.size),
            #A.augmentations.crops.transforms.CenterCrop(448,448),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
def get_transforms_1(*, data,CFG):
    if data == 'train':
        return Compose([
            Resize(512,512),
            #A.augmentations.crops.transforms.CenterCrop(1024*0.8,1024*0.8),
            A.crops.transforms.RandomResizedCrop(448,448),
            #A.crops.transforms.RandomCrop(CFG.preprocess.size,CFG.preprocess.size),
            A.HorizontalFlip(p=CFG.aug.HorizontalFlip.p),
            A.VerticalFlip(p=CFG.aug.VerticalFlip.p),

            A.RandomRotate90(p=CFG.aug.RandomRotate90.p),
            A.ShiftScaleRotate(
                shift_limit=CFG.aug.ShiftScaleRotate.shift_limit,
                scale_limit=CFG.aug.ShiftScaleRotate.scale_limit,
                rotate_limit=CFG.aug.ShiftScaleRotate.rotate_limit,
                p=CFG.aug.ShiftScaleRotate.p),
            A.RandomBrightnessContrast(
                brightness_limit=CFG.aug.RandomBrightnessContrast.brightness_limit,
                contrast_limit=CFG.aug.RandomBrightnessContrast.contrast_limit,
                p=CFG.aug.RandomBrightnessContrast.p),

            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'valid':
        return Compose([
            Resize(CFG.preprocess.size,CFG.preprocess.size),
            A.augmentations.crops.transforms.CenterCrop(448,448),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

def transform_qishen(*, data,CFG):
    if data == 'train':
        return Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                ], p=0.75),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
            A.CoarseDropout(max_holes=1, max_height=int(CFG.preprocess.size * 0.4), max_width=int(CFG.preprocess.size * 0.4), p=0.75),
            #Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'valid':
        return Compose([
            Resize(CFG.preprocess.size,CFG.preprocess.size),
            #Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

#### augmentation ==============

#### model ================
#### model ================
SEQ_POOLING = {
    'gem': GeM(dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': nn.AdaptiveAvgPool2d(1),
    'max': nn.AdaptiveMaxPool2d(1)
}

class Model_iafoss(nn.Module):
    def __init__(self, base_model='tf_efficientnet_b0_ns',pool="avg",pretrain=True):
        super(Model_iafoss, self).__init__()
        self.base_model = base_model 
        if self.base_model=="dino_vit_s":
            checkpoint_key = "teacher"
            pretrained_weights = "/home/abebe9849/MAYO/dino/exp000/checkpoint0320.pth"
            #self.model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for _, p in self.model.head.named_parameters():
                p.requires_grad = True

            freeze =1
            for n, p in self.model.blocks.named_parameters():
                if int(n.split(".")[0])>=(12-freeze):
                    p.requires_grad = True
            avgpool_patchtokens = 0
            n_last_blocks = 4
            self.n_last_blocks = n_last_blocks
            nc = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
            self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)

            nc*=64
            self.head = nn.Sequential(nn.Linear(nc,512),
                            nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,1))
            self.exam_predictor = nn.Linear(512*2, 1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            
        else:
            self.model = timm.create_model(self.base_model, pretrained=True, num_classes=0,in_chans=3)
            #self.model.conv_stem = nn.Conv2d(2, 32, kernel_size=3, padding=1, stride=1, bias=False)
            nc = self.model.num_features
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(nc,512),
                            nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,8))

            self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
            self.exam_predictor = nn.Linear(512*2, 8)
            self.pool = nn.AdaptiveAvgPool1d(1)

        
    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]
        #print(input1.size())#torch.Size([2, 107, 3, 256, 256])
        input1 = input1.reshape(-1,shape[2],shape[3],shape[4])

        if "dino" in self.base_model:
            intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
            x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
            #x = x.view(batch_size,x.shape[1]*n)
            #y = self.head(x)
           
            return y
        else:
            x = self.model.forward_features(input1) #bs*num_tile,embed_dim,h,w 
            shape = x.size()
            x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
            y = self.head(x)
            #x =  self.model(input1)
            #embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            #embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            #y = self.exam_predictor(embeds)
            return y

#### model ================

import vision_transformer as vits

###  loss =============


###  loss =============


###  metric =============
from sklearn.metrics import roc_auc_score
def AUC_(true,predict):
    auc_score = []
    for i in range(predict.shape[1]):
        try:
            auc_score.append(roc_auc_score(true[:,i], predict[:,i]))
        except:
            print(true[:,i].mean())
    return auc_score,sum(auc_score)/len(auc_score)






def train_fn(CFG,fold,folds):

    #nvnn_transform = A.load("/home/u094724e/NIH/siim2021_nvnn/pipeline1/configs/aug/s_0220/0220_hf_cut_sm2_0.75_384_v1.yaml", data_format='yaml')

    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"### fold: {fold} ###")
    
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    val_folds = folds.loc[val_idx].reset_index(drop=True)
    tra_folds = folds.loc[trn_idx]
    train_dataset = TrainDataset(tra_folds,train=True, 
                                 transform1=get_transforms(data='train',CFG=CFG))#get_transforms(data='train',CFG=CFG)
    valid_dataset = TrainDataset(val_folds,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG))#


    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=8,pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=8,pin_memory=True)



    ###  model select ============
    if CFG.model.name=="dino_vit_B":
        model = vits.__dict__["vit_base"](patch_size=16)
        pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_exp001/checkpoint0600.pth"
        #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/checkpoint0600.pth"
        state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        model.head = nn.Linear(768*CFG.model.Ncat,n_target)
    elif CFG.model.name=="dino_vit_s":
        model = vits.__dict__["vit_small"](patch_size=16)
        pretrained_weights = "/home/abe/KidneyM/dino/src/vit256_small_dino.pth"
        #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/checkpoint0600.pth"
        state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        model.head = nn.Linear(384*CFG.model.Ncat,n_target)
    elif CFG.model.name=="lunit":
        from timm.models.vision_transformer import VisionTransformer


        def get_pretrained_url(key):
            URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
            model_zoo_registry = {
                "DINO_p16": "dino_vit_small_patch16_ep200.torch",
                "DINO_p8": "dino_vit_small_patch8_ep200.torch",
            }
            pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
            return pretrained_url


        def vit_small(pretrained, progress, key, **kwargs):
            patch_size = kwargs.get("patch_size", 16)
            model = VisionTransformer(
                img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
            )
            if pretrained:
                pretrained_url = get_pretrained_url(key)
                verbose = model.load_state_dict(
                    torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
                )
                print(verbose)
            return model
        model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        model.head = nn.Linear(384,n_target)
    elif CFG.model.name=="Dino_vit_B":
        model = vits.__dict__["vit_base"](patch_size=16)
        pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_exp001/checkpoint0600.pth"
        #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/checkpoint0600.pth"
        state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        model.head = nn.Linear(768,n_target)
        CFG.model.linear = False
    elif "vit_base" in CFG.model.name or "vit_large" in CFG.model.name:
        model = timm.create_model(CFG.model.name, pretrained=True,num_classes=n_target,in_chans=3,img_size=CFG.preprocess.size)
    else:
        model = timm.create_model(CFG.model.name, pretrained=True,num_classes=n_target,in_chans=3) 
    #exit()
    if CFG.model.linear:
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
    model.to(device)
    # ============


    ###  optim select ============
    if CFG.train.optim=="adam":
        optimizer = Adam(model.parameters(), lr=CFG.train.lr, amsgrad=False)
    elif CFG.train.optim=="adamw":
        optimizer = AdamW(model.parameters(), lr=CFG.train.lr,weight_decay=5e-5)


    ###  scheduler select ============
    if CFG.train.scheduler.name=="cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs, eta_min=CFG.train.scheduler.min_lr)
    elif CFG.train.scheduler.name=="cosine_warmup":
        scheduler =T.get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=len(train_loader)*CFG.train.scheduler.warmup,
        num_training_steps=len(train_loader)*CFG.train.epochs)

    # ============

    ###  loss select ============
    criterion=nn.CrossEntropyLoss()

    print(criterion)
    ###  loss select ============

    softmax = nn.Softmax(dim=-1)

    scaler = torch.cuda.amp.GradScaler()
    best_score = 0
    best_loss = np.inf
    best_preds = None
        
    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            optimizer.zero_grad()
            images = images.to(device).float()
            labels = labels.to(device)

            ### mix系のaugumentation=========
            rand = np.random.rand()
            ##mixupを終盤のepochでとめる
            if epoch+1 >=CFG.train.without_hesitate:
                rand=0

            if CFG.augmentation.mix_p>rand and CFG.augmentation.do_mixup:
                images, y_a, y_b, lam = mixup_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            elif CFG.augmentation.mix_p>rand and CFG.augmentation.do_cutmix:
                images, y_a, y_b, lam = cutmix_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            elif CFG.augmentation.mix_p>rand and CFG.augmentation.do_resizemix:
                images, y_a, y_b, lam = resizemix_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            elif CFG.augmentation.mix_p>rand and CFG.augmentation.do_fmix:
                images, y_a, y_b, lam = fmix_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            ### mix系のaugumentation おわり=========

            
            with autocast(enabled=CFG.train.amp):
                if CFG.model.name=="dino_vit_B" or CFG.model.name=="dino_vit_s":
                    y_preds = model.get_intermediate_forward(images,CFG.model.Ncat)
                else:
                    y_preds = model(images)                
                if CFG.augmentation.mix_p>rand:
                    loss = mixup_criterion(criterion, y_preds, y_a, y_b, lam)
                else:
                    loss = criterion(y_preds,labels)

            scaler.scale(loss).backward()

            if (i+1)%CFG.train.ga_accum==0 or i==-1:
                scaler.step(optimizer)
                scaler.update()        
                if CFG.train.scheduler.name=="cosine_warmup":
                    scheduler.step()
            if CFG.train.scheduler.name=="cosine":
                scheduler.step()


            avg_loss += loss.item() / len(train_loader)
        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:
            images = images.to(device).float()
            labels = labels.to(device)
            with torch.no_grad():
                if CFG.model.name=="dino_vit_B" or CFG.model.name=="dino_vit_s":
                    y_preds = model.get_intermediate_forward(images,CFG.model.Ncat)
                else:
                    y_preds = model(images)
                loss =  criterion(y_preds,labels)
                y_preds = softmax(y_preds)

            valid_labels.append(labels.to('cpu').numpy())
            preds.append(y_preds.to('cpu').numpy())
            avg_val_loss += loss.item() / len(valid_loader)
        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)



        elapsed = time.time() - start_time
        AUC_score = roc_auc_score(valid_labels, preds[:,1])

        log.info(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.5f} AUC_score {AUC_score:.4f}  time: {elapsed:.0f}s')

        #if best_loss>avg_val_loss:#pr_auc best
        #     best_loss = avg_val_loss
            #best_preds = preds
        #    log.info(f'  Epoch {epoch+1} - Save Best loss: {best_loss:.4f}')
        #    torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_best_loss.pth')

        if AUC_score>best_score:#pr_auc best
            best_score = AUC_score
            best_preds = preds
            log.info(f'  Epoch {epoch+1} - Save Best score: {AUC_score:.4f}')
            torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_best_AUC.pth')

    for c in range(n_target):
        col = f"pred_{c}"
        val_folds[col]=best_preds[:,c]

    return best_preds, valid_labels,val_folds


def eval_func(model, valid_loader, device,CFG):
    model.to(device) 
    model.eval()
    softmax = nn.Softmax(dim=-1)
    scaler = torch.cuda.amp.GradScaler()

    valid_labels = []
    preds = []

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, (images, labels) in tk1:
        images = images.to(device).float()
        labels = labels.to(device)
        with torch.no_grad():
            with autocast():
                if CFG.model.name=="dino_vit_B" or CFG.model.name=="dino_vit_s":
                    y_preds = model.get_intermediate_forward(images,CFG.model.Ncat)
                else:
                    y_preds = model(images)               
                y_preds = softmax(y_preds)

        valid_labels.append(labels.to('cpu').numpy())
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    return preds#,valid_labels


def submit(test_df,folds,CFG):
    valid_dataset = TrainDataset(test_df,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG))#

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=8,pin_memory=True)
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds_ = []
    for fold in range(folds["fold"].nunique()):
        if CFG.model.name=="dino_vit_B":
            model = vits.__dict__["vit_base"](patch_size=16)
            pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_exp001/checkpoint0600.pth"
            #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/checkpoint0600.pth"
            state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

            model.head = nn.Linear(768*CFG.model.Ncat,n_target)
        elif CFG.model.name=="dino_vit_s":
            model = vits.__dict__["vit_small"](patch_size=16)
            #pretrained_weights = "/home/abe/KidneyM/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
            #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/checkpoint0600.pth"
            state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

            model.head = nn.Linear(384*CFG.model.Ncat,n_target)
        elif CFG.model.name=="lunit":
            from timm.models.vision_transformer import VisionTransformer


            def get_pretrained_url(key):
                URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
                model_zoo_registry = {
                    "DINO_p16": "dino_vit_small_patch16_ep200.torch",
                    "DINO_p8": "dino_vit_small_patch8_ep200.torch",
                }
                pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
                return pretrained_url


            def vit_small(pretrained, progress, key, **kwargs):
                patch_size = kwargs.get("patch_size", 16)
                model = VisionTransformer(
                    img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
                )
                if pretrained:
                    pretrained_url = get_pretrained_url(key)
                    verbose = model.load_state_dict(
                        torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
                    )
                    print(verbose)
                return model
            model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
            
        elif "vit_base" in CFG.model.name or "vit_large" in CFG.model.name:
            model = timm.create_model(CFG.model.name, pretrained=True,num_classes=n_target,in_chans=3,img_size=CFG.preprocess.size)
        else:
            model = timm.create_model(CFG.model.name, pretrained=True,num_classes=n_target,in_chans=3) 
        model.load_state_dict(torch.load(f'fold{fold}_{CFG.general.exp_num}_best_AUC.pth'),strict=False)

        #model.load_state_dict(torch.load(f'/home/abe/KidneyM/dino/src/outputs/2023-09-28/14-16-10/fold{fold}_{CFG.general.exp_num}_best_AUC.pth'))

        model.to(device)
        
        preds = eval_func(model, valid_loader, device,CFG)
        for c in range(n_target):
            col = f"pred_fold{fold}_{c}"
            test_df[col]=preds[:,c]
            
        preds_.append(preds)
    preds_ = np.mean(np.stack(preds_),axis=0)
    for c in range(n_target):
        col = f"pred_mean_{c}"
        test_df[col]=preds_[:,c]
        
        
    
    
    return test_df,preds_
    
        
    
 


log = logging.getLogger(__name__)
@hydra.main(config_path="/home/abe/KidneyM/dino/src/",config_name="base")
#@hydra.main(config_path="/home/abe/KidneyM/dino/src/outputs/2023-11-27/09-08-47/.hydra/",config_name="config")
def main(CFG : DictConfig) -> None:

    CFG.general.exp_num+="_"+CFG.task+"_"
    if CFG.model.name=="dino_vit_B":
        CFG.general.exp_num+="dino"
        
    elif CFG.model.name=="vit_base_patch16_224" and CFG.model.linear==False:
        CFG.general.exp_num+="imnet_FT"
    else:
        CFG.general.exp_num+="imnet_linear"
        
    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")
    
    #if CFG.task =="alb":
    #
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__Alb_over3.0/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__Alb_over3.0/sub.csv")
    #elif CFG.task =="dbp":
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__dBP_over90/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__dBP_over90/sub.csv")
    #elif CFG.task =="DM":
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__DM_seed2021/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__DM_seed2021/sub.csv")
    #elif CFG.task =="sbp":
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__sBP_seed2021_over140/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__sBP_seed2021_over140/sub.csv")
    #elif CFG.task =="hep":
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__血尿_over1/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__血尿_over1/sub.csv")
    #elif CFG.task =="egfr":
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__eGFR_over60/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__eGFR_over60/sub.csv")
    #elif CFG.task =="upc":
    #    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__UPC_seed2023_over3.5/oof.csv")
    #    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__UPC_seed2023_over3.5/sub.csv")

    all_df = pd.read_csv("/home/abe/KidneyM/data/final_2_pas.csv")
    dic_ = {"egfr":60,"alb":3,"uprot":3.5,"DM":1,"DBP":90,"収縮期血圧":140,"血尿":1}
    COL = CFG.task
    CONF = dic_[COL]
    all_df = all_df[all_df[COL]!=-1].reset_index(drop=True)

    all_df["label"] =  np.where(all_df[COL] >=CONF,1,0)

    N_fold = 5

    n_target = all_df["label"].nunique()
    sgkf = StratifiedGroupKFold(n_splits=N_fold,random_state=2025,shuffle=True)
    for fold, ( _, val_) in enumerate(sgkf.split(X=all_df, y=all_df.label.to_numpy(),groups=all_df.WSI)):
        all_df.loc[val_ , "fold"] = fold

        val_df = all_df[all_df["fold"]==fold]
            #print(val_df["label"].nunique())                                                             
    folds = all_df[all_df["fold"]!=N_fold-1].reset_index(drop=True)
    test_df = all_df[all_df["fold"]==N_fold-1].reset_index(drop=True)
    
    
    #os.chdir("/home/abe/KidneyM/dino/src/outputs/2023-11-27/09-08-47")
    #time.sleep(20*15*3)

    preds = []
    valid_labels = []
    oof = pd.DataFrame()
    #"""


    for fold in range(folds["fold"].nunique()):
        seed_torch(seed=CFG.general.seed)
        _preds, _valid_labels,_oof_val = train_fn(CFG,fold,folds)
        preds.append(_preds)
        valid_labels.append(_valid_labels)
        oof = pd.concat([oof,_oof_val])
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    AUC_score = roc_auc_score(valid_labels, preds[:,1])

    log.info(f"OOF")
    log.info(f"AUC_mean :{AUC_score}")
    oof.to_csv(f"oof_{CFG.general.exp_num}.csv",index=False)
    #"""

    test_df,pred_ = submit(test_df,folds,CFG)
    test_df.to_csv(f"sub_{CFG.general.exp_num}.csv",index=False)

    AUC_score = roc_auc_score(test_df["label"], pred_[:,1])
    log.info(f"TEST")
    log.info(f"AUC_mean :{AUC_score}")
    sub_g = test_df.groupby("WSI")[["label","pred_mean_1","pred_fold1_1"]].mean()

    AUC_score = roc_auc_score(sub_g["label"],sub_g["pred_mean_1"])
    log.info(f"TEST groupby")
    log.info(f"AUC_mean :{AUC_score}")


    


if __name__ == "__main__":
    main()
