

### installation


### in order to use cuml/cupy
conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia rapids=24.06 python=3.11 cuda-version=11.2

kidneySSL

### pretraining
conda activate kidneySSL
python -m torch.distributed.launch --nproc_per_node=3 main_dino.py ##600epoch dino training

### embed image
conda activate kidneySSL
cd dino
python get_emb.py --arch vit_base #dino glo pretrained ViT-B
python get_emb.py --arch imnet #supervised ImageNet pretrained ViT-B

###  train&evaluate kNN model 

##### 4cls 
conda activate rapids-24.06 
python train_knn.py vit_base 
python train_knn.py imnet

###  train&evaluate linear model

### 





