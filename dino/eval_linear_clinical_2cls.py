# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json,time
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import matplotlib.pyplot as plt
import utils
import vision_transformer as vits

import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df,train=True,transform=None):
        self.paths = df["path"].to_numpy()
        self.labels = df["label"].to_numpy()
        self.transform = transform

        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]#+".jpg"
        image = Image.open(file_path)
        
        image = self.transform(image)
        

        label = torch.tensor(self.labels[idx]).long()
        return image,label


all_df = pd.read_csv("final_pas.csv")

def sel(df,SEED=42):
    np.random.seed(SEED)
    tmps = []
    for label in df["label"].unique():
        tmp = df[df["label"]==label]
        if tmp["WSI"].nunique()==1:
            tmp = tmp.sample(frac=0.25,random_state=SEED)
        else:
            tmp_WSI = list(tmp["WSI"].unique())
            tmp_WSI = np.random.choice(tmp_WSI, size=len(tmp_WSI)//4, replace=False)
            tmp = tmp[tmp["WSI"].isin(tmp_WSI)]
        tmps.append(tmp)
        
    df = pd.concat(tmps,axis=0).reset_index(drop=True)
    return df
        
      
#exit()

from sklearn.metrics import roc_auc_score
def eval_linear(args):
    #utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    CONF = args.CONF
    COL = args.COL
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

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
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
    # load weights to evaluate
    args.num_labels = n_target
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize((512,512), interpolation=3),
        #pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    FOLD = int(args.fold)
    val_df = folds[folds["fold"]==FOLD].reset_index(drop=True)
    dataset_val = TrainDataset(val_df, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    tra_df = folds[folds["fold"]!=FOLD].reset_index(drop=True)
    
    dataset_test = TrainDataset(test_df, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop((512,512)),
        #pth_transforms.RandomApply(
        #        [pth_transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.1)],
        #        p=0.2
        #    ),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.RandomVerticalFlip(),
        #pth_transforms.RandomApply(
        #        [pth_transforms.GaussianBlur(7, sigma=(0.1, 2.0))],
        #        p=0.2
        #    ),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    #tra_df = sel(tra_df)    
    dataset_train = TrainDataset(tra_df, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint

    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    best_state=None
    best_score=0
    best_preds=None
    patient=15
    
    for epoch in range(start_epoch, args.epochs):

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats,preds,valid_labels = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            AUC_score = roc_auc_score(valid_labels, preds[:,1])
            print(f"AUC_score  {AUC_score}")
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            if AUC_score>best_score:
                best_score=AUC_score
                best_preds=preds
                patient=15
                torch.save(linear_classifier.state_dict(),os.path.join(args.output_dir,"linear_w.pth"))
            else:
            	patient-=1
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) +f"AUC_score:{AUC_score}"+ "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
        if patient==0:break

    state_dict = torch.load(os.path.join(args.output_dir,"linear_w.pth"))
    linear_classifier.load_state_dict(state_dict, strict=False)
    linear_classifier.eval()
    test_stats,preds,valid_labels = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
    for i in range(args.num_labels):
    	col=f"pred_{i}"
    	val_df[col]=preds[:,i]
    val_df.to_csv(os.path.join(args.output_dir,f"oof_fold{FOLD}_dino.csv"),index=False)
    
    test_stats,preds,valid_labels = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
    
    for i in range(args.num_labels):
    	col=f"pred_{i}"
    	test_df[col]=preds[:,i]
    test_df.to_csv(os.path.join(args.output_dir,f"sub_fold{FOLD}_dino.csv"),index=False)
    
    

def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 40, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    labels = []
    preds=[]
    softmax=nn.Softmax(dim=1)
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        preds.append(softmax(output).to("cpu").numpy())
        labels.append(target.to("cpu").numpy())
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    preds = np.concatenate(preds)
    labels= np.concatenate(labels)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},preds,labels


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
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/home/abe/KidneyM/dino/pas_glomerulus_exp001/checkpoint0600.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/home/abe/KidneyM/dino/FIX/B600_", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--COL', default="egfr", type=str, help='')
    parser.add_argument('--CONF', default=60, type=int, help='')

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    ROOT = f"{args.output_dir}_{args.COL}_over_{args.CONF}/"
    os.makedirs(ROOT,exist_ok=True)
    args.output_dir = ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    
    args.fold = 1
    args.output_dir =ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    
    args.fold = 2
    args.output_dir = ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    
    args.fold = 3
    args.output_dir =ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    
    
    cols = [f"pred_{i}" for i in range(args.num_labels)]
    
    
    sub_pred_ = []
    oof_ = []
    for fold in [0,1,2,3]:
    
        sub = pd.read_csv(os.path.join(ROOT+str(fold),f"sub_fold{fold}_dino.csv"))
        sub_pred_.append(sub[cols].to_numpy())
        oof_.append(pd.read_csv(os.path.join(ROOT+str(fold),f"oof_fold{fold}_dino.csv")))
    sub_pred_ = np.mean(np.stack(sub_pred_),axis=0)
    oof_ = pd.concat(oof_,axis=0)
    oof_.to_csv(f"{ROOT}oof.csv",index=False)
    
    for i in range(args.num_labels):
        col = f"pred_mean_{i}"
        sub[col]=sub_pred_[:,i]
    sub.to_csv(f"{ROOT}sub.csv",index=False)
    
    
    
        
    
    
    

