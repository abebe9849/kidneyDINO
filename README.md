# kidneyDINO
[[paper]()]

Official implementation of the paper "Self-supervised learning for feature extraction from glomerular images and disease classification with minimal annotations ".

<details>
  <summary>
	  <b>Key Points </b>
  </summary>

1. **Self-supervised learning extracts meaningful glomerular features without teacher labels:** 
2. **DINO outperforms conventional supervised learning in disease and clinical classification :** 
3. **DINO enables deep learning on small datasets, reducing annotation efforts :**
</details>

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

 
```python
conda activate rapids-24.06
##### disease 4cls 
python train_knn.py vit_base 
python train_knn.py imnet
##### clinical parameter 2cls


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





