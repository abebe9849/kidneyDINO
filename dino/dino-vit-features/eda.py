##%%
import glob,cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

images_paths = glob.glob("/home/abe/KidneyM/dino/dino-vit-features/forPCA/*")
images_wsi = ["_".join(i.split("/")[-1].split("__")[0].split("_")[:-1]) for i in images_paths]
df = pd.read_csv("/home/abe/KidneyM/hubmap2021/MATUI_bbxo/final_pas.csv")

pca_image_ = np.load("/home/abe/KidneyM/dino/dino-vit-features/accum_data_1024/pca_per_image.npy")

K = int(pca_image_.shape[1]**0.5)
n_components = pca_image_.shape[-1]

pca_image_ = pca_image_.reshape((len(pca_image_),K,K,n_components))

print(pca_image_.shape)

def func(comp):
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    return comp_img

def func2(comp):
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    return (comp_img*255).astype(np.uint8)
"""
for i in range(len(pca_image_)):
    
    new_ = np.zeros((512*3,512*3,3))
    imgs = []
    pca_image = pca_image_[i]
    for comp_idx in range(6):
        comp = pca_image[:, :, comp_idx]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        imgs.append((np.stack([comp_img,comp_img,comp_img],axis=-1)*255).astype(np.uint8))
    comp = pca_image[:, :, :3]
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    imgs.append((comp_img*255).astype(np.uint8))
    comp = pca_image[:, :,-3:]
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    imgs.append((comp_img*255).astype(np.uint8))
    imgs.append(cv2.imread(images_paths[i]))
    
    for n in range(3):
        for m in range(3):
            img = imgs[n*3+m]
            print(img.shape)
            img = cv2.resize(img,(512,512))
            print(new_[n*512:n*512+512,m*512:m*512+512,:].shape,n,m)
            new_[n*512:n*512+512,m*512:m*512+512,:]=img
    path = images_paths[i].split("/")[-1]
    cv2.imwrite(f"/home/abe/KidneyM/dino/dino-vit-features/accum_data_1024/DIM6/{path}",new_)
"""        



for i in range(len(pca_image_)):
    pca_image = pca_image_[i]

    #plt.imshow(func(pca_image[:,:,0]),cmap="gray")
    #plt.show()
    #plt.imshow(func(pca_image[:,:,1]),cmap="gray")
    #plt.show()

    img_r = func(pca_image[:,:,1])-func(pca_image[:,:,0])
    #plt.imshow(img_r,cmap="gray")
    #plt.show()

    #plt.imshow(func(pca_image[:,:,2]),cmap="gray")
    #plt.show()
    img_g = func(pca_image[:,:,2])-func(pca_image[:,:,0])
    #plt.imshow(img_g,cmap="gray")
    #plt.show()

    #plt.imshow(func(pca_image[:,:,3]),cmap="gray")
    #plt.show()
    img_b = func(pca_image[:,:,3])-func(pca_image[:,:,0])
    #plt.imshow(img_b,cmap="gray")
    #plt.show()
    img = np.stack([func(img_r),func(img_g),func(img_b)],axis=-1)[:,:,::-1]
    #plt.imshow(img,cmap="gray")
    #plt.show()

    comp = pca_image[:, :, -3:]
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    #plt.imshow(img,cmap="gray")
    #plt.show()
    ##dim1は背景除去ではっきりする　dim2は基底膜黒　が見えなくなる
    img = (np.concatenate([img,comp_img],axis=1)*255.).astype(np.uint8)[:,:,::-1]
    path = images_paths[i].split("/")[-1].split("__")[0]
    cv2.imwrite(f"/home/abe/KidneyM/dino/dino-vit-features/accum_data_1024/diff/{path}_diff.png",img)






# %%
