# kidneyDINO
[[paper]()]

Official implementation of the paper "Self-supervised learning for feature extraction from glomerular images and disease classification with minimal annotations ".

######## be going to attach figure 1 here after it is adopted.


### Key points

1. Self-supervised learning extracts meaningful glomerular features without teacher labels
2. DINO outperforms conventional supervised learning in disease and clinical classification 
3. DINO enables deep learning on small datasets, reducing annotation efforts


### installation
```
conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia rapids=24.06 python=3.11 cuda-version=11.2 #rapids-24.06
conda env create -f=kidneySSL.yml #kidneySSL
```

### pretraining

``` python
conda activate kidneySSL
python -m torch.distributed.launch --nproc_per_node=3 main_dino.py ##600epoch dino training
```

### download dino-vit-B pretrained weights

https://www.kaggle.com/datasets/abebe9849/sslglomerular-images-weights 
or
https://www.kaggle.com/datasets/niioka/dino-ssl/data/checkpoint.pth


### embed image
```python
conda activate kidneySSL
cd dino
python get_emb.py --arch vit_base #dino glo pretrained ViT-B
python get_emb.py --arch imnet #supervised ImageNet pretrained ViT-B
```
###  train&evaluate kNN model 

 
```python
conda activate rapids-24.06
##### disease 4cls 
python train_knn.py vit_base 
python train_knn.py imnet
##### clinical parameter 2cls
python train_knn_2cls.py dino/{task_name} 
python train_knn_2cls.py imnet/{task_name} 

```
###  train&evaluate linear model

```python

conda activate kidneySSL

##### disease 4cls 

python dino/eval_linear_e4cls.py #lableled 100%
python dino/eval_linear_e4cls_25per.py #lableled 25%,5seed for select 25%
##### clinical parameter 2cls
python base_clinical.py 
```

### PCA analyze 

```python

python dino/dino-vit-features/pca.py --model_type dino_vitb16
python dino/dino-vit-features/pca.py --model_type vit_base_patch16_224
#### for supp fig2 rgb histgram analyze
python dino/dino-vit-features/RGB_analyze.py

```

### attention map of last blocks(each head visalize)
```python
python dino/visualize_attention.py 
```

### Issues
Please open new issue threads specifying the issue with the codebase or report issues directly to masamasa20001002@gmail.com . 

### Citation


### License

The source code for the site is licensed under the MIT license






