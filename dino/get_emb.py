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
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths,train=True,transform=None):
        self.paths = paths
        self.transform = transform

        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]
        image = Image.open(file_path)
        
        image = self.transform(image)
        

        label = torch.tensor(0).long()
        return image,label

def eval_linear(args):
    #utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    # fix the seed for reproducibility 
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    if args.arch == 'dalle_encoder':
        train_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(112),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(128, interpolation=3),
            pth_transforms.CenterCrop(112),
            pth_transforms.ToTensor(),
        ])
    else:
        train_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(512, interpolation=3),
            pth_transforms.CenterCrop(448),
            #pth_transforms.RandomGrayscale(p=1),
            pth_transforms.RandomEqualize(p=1),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    paths_ =  glob.glob("/home/abe/KidneyM/hubmap2021/yolov5/SAHI/runs/predict/PAS_/croped/*")
    df = pd.read_csv("/home/abe/KidneyM/hubmap2021/MATUI_bbxo/final_pas.csv")
    df["h_w"]=df["H"]/df["W"]
    df = df[df["h_w"]>0.5]
    df = df[df["h_w"]<2].reset_index(drop=True)
    
    paths_ = df["path"].values
    dataset_val = TrainDataset(paths_, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    
    if 'swin' in args.arch:
        num_features = []
        for i, d in enumerate(model.depths):
            num_features += [int(model.embed_dim * 2 ** i)] * d
        feat_dim = sum(num_features[-args.n_last_blocks:])
    else:
        feat_dim = embed_dim * (args.n_last_blocks * int(args.avgpool_patchtokens != 1) + \
            int(args.avgpool_patchtokens > 0))



    emb = validate_network(val_loader, model)
    
    np.save(f"{args.output_dir}/finalG_emb.npy",emb)









@torch.inference_mode()
def validate_network(val_loader, model):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    preds = []
    valid_labels = []
    softmax= nn.Softmax(dim=1)
    for inp, target in metric_logger.log_every(val_loader, 50, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        intermediate_output = model.get_intermediate_layers(inp, 4)

        output = [x[:, 0] for x in intermediate_output]
        
        output = torch.cat(output, dim=-1).to("cpu").numpy()
        

        preds.append(output)

    preds = np.concatenate(preds)

    return preds


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base/large-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--arch', default='vit_large', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101', 'dalle_encoder'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/checkpoint0600.pth', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
        help='Please specify path to the ImageNet data.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/home/abe/KidneyM/dino/pas_glomerulus_wbf_L", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument("--fold", default=1, type=int)

    args = parser.parse_args()
    #args.output_dir = os.path.join(args.output_dir,str(args.fold))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
