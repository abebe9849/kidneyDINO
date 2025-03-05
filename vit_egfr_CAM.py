import pandas as pd
import timm 
import os
import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=28, width=28):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

import pandas as pd 
#path,WSI,target,label,BMI,DBP,DM,age,alb,egfr,uprot,収縮期血圧,血尿,H,W,h/w,kouka_or_not,fold,pred_fold0_0,pred_fold0_1,pred_fold1_0,pred_fold1_1,pred_fold2_0,pred_fold2_1,pred_fold3_0,pred_fold3_1,pred_mean_0,pred_mean_1
root="/home/abe/KidneyM/dino/src/EGFR_cam/low_1"

import torch.nn as nn
import vision_transformer as vits
import glob
imnet = pd.read_csv("/home/abe/KidneyM/dino/src/outputs/2024-04-12/egfr_FT/sub_exp__egfr_imnet_FT.csv")

dino = pd.read_csv("/home/abe/KidneyM/dino/src/EGFR_cam/low_egfr_imnet_C.csv")
paths =dino["path"].values
#)imnet["pred"]=np.where(imnet["pred_mean_1"]>0.5,1,0)
#dpath"].isin(dino_e["path"].unique())]
for path in paths[10:]:
    args = get_args()
    path_c = args.image_path.split("/")[-1].split("__0.")[0]
    args.image_path =path  #glob.glob(f"{ROOT}/CCE_OU_01_12*")[0]
    if os.path.exists(f'{root}/{args.method}_{path_c}_dino_above_imnet.jpg'):continue
    path_c = args.image_path.split("/")[-1].split("__0.")[0]

    methods = {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,"eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}



    model2 = vits.__dict__["vit_base"](patch_size=16)
    #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerulus_exp001/checkpoint0600.pth"
    #pretrained_weights = "/home/abe/KidneyM/dino/pas_glomerut
    model = timm.create_model("vit_base_patch16_224", pretrained=True,num_classes=2,in_chans=3,img_size=448)
    
    for fold in range(4):
        #model.load_state_dict(torch.load(f"/home/abe/KidneyM/dino/src/outputs/2024-04-12/upc_FT/fold{fold}_exp__uprot_imnet_FT_best_AUC.pth"))
        #model2.load_state_dict(torch.load(f"/home/abe/KidneyM/dino/src/outputs/2024-04-19/11-45-38/ld{fold}_exp__uprot_imnet_linear_best_AUC.pth"),strict=False) 
        model.load_state_dict(torch.load(f"/home/abe/KidneyM/dino/src/outputs/2024-04-12/egfr_FT/fold{fold}_exp__egfr_imnet_FT_best_AUC.pth"))
        model2.load_state_dict(torch.load(f"/home/abe/KidneyM/dino/src/outputs/2024-04-18/18-30-35/fold{fold}_exp__egfr_imnet_linear_best_AUC.pth"),strict=False) 
        model.eval()
        model2.eval()
        if args.use_cuda:
            model = model.cuda()
        target_layers = [model.blocks[-1].norm1]
        #if args.method not in methods:raise Exception(f"Method {args.method} not implemented")
        #if args.method == "ablationcam":cam = methods[args.method](use_cuda=args.hape_transform=reshape_transform,lationLayerVit())#else:
        cam = methods[args.method](model=model,target_layers=target_layers,use_cuda=args.use_cuda,
                                    reshape_transform=reshape_transform)
        rgb_img_ = cv2.imread(args.image_path, 1)[:, :, ::-1]
        print(rgb_img_.shape)
        rgb_img_ = cv2.resize(rgb_img_, (512, 512))[32:-32, 32:-32]
        rgb_img = np.float32(rgb_img_) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        targets = None
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        if fold ==0:
            cam_image__ = np.hstack([rgb_img_,cam_image])
        else:
            cam_image__ = np.hstack([cam_image__,cam_image])
        #cv2.imwrite(f'{root}/{args.method}_{path_c}_nor_inet_fold{fold}.jpg', cam_image__)
        target_layers = [model2.blocks[-1].norm1]
        cam = methods[args.method](model=model2,target_layers=target_layers,use_cuda=args.use_cuda,
                                    reshape_transform=reshape_transform)
        rgb_img_ = cv2.imread(args.image_path, 1)[:, :, ::-1]
        print(rgb_img_.shape)
        rgb_img_ = cv2.resize(rgb_img_, (512, 512))[32:-32, 32:-32]
        rgb_img = np.float32(rgb_img_) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        targets = None
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)
        # Here grayscale_cam has only one image in the batch                                                          
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        if fold ==0:
            cam_image__2 = np.hstack([rgb_img_,cam_image])
        else:
            cam_image__2 = np.hstack([cam_image__2,cam_image])
    save =np.vstack([cam_image__2,cam_image__])
    cv2.imwrite(f'{root}/{args.method}_{path_c}_dino_above_imnet.jpg', save)

    
