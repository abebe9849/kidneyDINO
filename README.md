

### pretraining

python -m torch.distributed.launch --nproc_per_node=3 main_dino.py ##600epoch dino training

### embed image
cd dino
python get_emb.py --arch vit_base #dino glo pretrained ViT-B
python get_emb.py --arch imnet #supervised ImageNet pretrained ViT-B

### 





