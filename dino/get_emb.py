# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from DINO library:
https://github.com/facebookresearch/dino
"""
import pandas as pd
import os,glob
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import json
import copy
import torch
import torch.backends.cudnn as cudnn
import utils
import vision_transformer as vits
from torchvision import models as torchvision_models

import sklearn.metrics as metrics
from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms

from PIL import Image
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import numpy as np


class embDataset(torch.utils.data.Dataset):
    def __init__(self, paths,transform=None):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]
        image = Image.open(file_path)
        image = self.transform(image)
        return image

def eval_linear(args):
    #utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    # fix the seed for reproducibility 
    utils.fix_random_seeds(args.seed)

    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(512, interpolation=3),
        pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    
    # ============ building network ... ============
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        model.cuda()
        model.eval()
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    # if the network is a XCiT
    elif "imnet" == args.arch:
        model = timm.create_model("vit_base_patch16_224", pretrained=True,num_classes=0,in_chans=3,img_size=448)
        model.cuda()
        model.eval()

    df = pd.read_csv("oof.csv")
    test_df = pd.read_csv("test.csv")

    dataset_val = embDataset(test_df["path"].to_numpy(), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset_val,batch_size=128,num_workers=10,pin_memory=True,)
    emb = validate_network(val_loader, model)
    np.save(f"{args.output_dir}/{args.arch}_test.npy",emb)
    
    for fold in range():
        tra_df = df[df["fold"]!=fold].reset_index(drop=True)
        val_df = df[df["fold"]==fold].reset_index(drop=True)
        

        dataset_val = embDataset(tra_df["path"].to_numpy(), transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset_val,batch_size=128,num_workers=10,pin_memory=True,)
        emb = validate_network(val_loader, model)
        np.save(f"{args.output_dir}/{args.arch}_tra_fold{fold}.npy",emb)
        
        dataset_val = embDataset(val_df["path"].to_numpy(), transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset_val,batch_size=128,num_workers=10,pin_memory=True,)
        emb = validate_network(val_loader, model)
        np.save(f"{args.output_dir}/{args.arch}_val_fold{fold}.npy",emb)





@torch.inference_mode()
def validate_network(val_loader, model):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    preds = []
    for inp in metric_logger.log_every(val_loader, 50, header):
        inp = inp.cuda(non_blocking=True)
        output = model(inp).to("cpu").numpy()
        preds.append(output)
    preds = np.concatenate(preds)

    return preds



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_base', type=str, choices=['imnet', 'vit_base', 'vit_large'] ,help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='checkpoint0600.pth', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--output_dir", default="./", type=str)

    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    #args.output_dir = os.path.join(args.output_dir,str(args.fold))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
