# kidneyDINO
[[paper]()]

Official implementation of the paper "Self-supervised learning for feature extraction from glomerular images and disease classification with minimal annotations ".

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

### embed image
```python
conda activate kidneySSL
cd dino
python get_emb.py --arch vit_base #dino glo pretrained ViT-B
python get_emb.py --arch imnet #supervised ImageNet pretrained ViT-B
```
###  train&evaluate kNN model 

##### 4cls 
```
conda activate rapids-24.06 
python train_knn.py vit_base 
python train_knn.py imnet
```
###  train&evaluate linear model
```
conda activate kidneySSL

##### 4cls 

python dino/eval_linear_e4cls.py #lableled 100%
python dino/eval_linear_e4cls_25per.py #lableled 25%,5seed for select 25%
```

### 





