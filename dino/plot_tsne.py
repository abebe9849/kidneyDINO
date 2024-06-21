"""
Embedding the glomerular image using a self-supervised trained model, compressing the dimension of the embedding using tsne, umap, etc. and displaying it along with the corresponding glomerular image on two-dimensional coordinates. 
Not used in the paper.

"""
import cv2
import pathlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, image_list, ax=None, zoom=0.5):
    if ax is None:
        ax = plt.gca()
    im_list = [OffsetImage(cv2.imread(p)[:,:,::-1], zoom=zoom) for p in image_list]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, im_list):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
import glob
df = pd.read_csv("all_df.csv")
df["h_w"]=df["H"]/df["W"]
df = df[df["h_w"]>0.5]
df = df[df["h_w"]<2].reset_index(drop=True)
path_l = df["path"].values
from PIL import Image
from functools import reduce
from skimage.transform import resize
def plot_tiles(imgs, emb, grid_units=50, pad=2):


    imgs = [cv2.resize(cv2.imread(p)[:,:,::-1],(256,256)) for p in imgs]
    # roughly 1000 x 1000 canvas
    cell_width = 15000 // grid_units
    s = grid_units * cell_width

    nb_imgs = len(imgs)

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx]               
                
                resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                #print(resized_tile.shape)
                
                
                #exit()
                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = resized_tile
                
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict

#canvas, img_idx_dict = plot_tiles(path_l, kills_reduced, grid_units=30)

#print(type(canvas))
plt.figure(figsize=(25,25))
#plt.imshow(canvas)
#plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/TSNE_1.png")
#plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/TSNE_2.png")

def plot_tiles_color(imgs, emb,cls_, grid_units=50, pad=2):

    imgs = [cv2.imread(p)[:,:,::-1] for p in imgs]
    # roughly 1000 x 1000 canvas
    cell_width = 10000 // grid_units
    s = grid_units * cell_width
    nb_imgs = len(imgs)
    print(nb_imgs)

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.full((s, s, 3),255, dtype=np.uint8)  # Initialize canvas as white image

    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx]
                cls_img = cls_[img_idx]
                
                

                # Resize the tile
                resized_tile = cv2.resize(tile,(cell_width - 2 * pad, cell_width - 2 * pad))
                # Create a border by setting the outer 2 pixels to green
                border_color = cls_2_color[cls_img]
                resized_tile[:pad, :] = border_color  # Top border
                resized_tile[-pad:, :] = border_color  # Bottom border
                resized_tile[:, :pad] = border_color  # Left border
                resized_tile[:, -pad:] = border_color  # Right border
                
                
                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width + pad:s - y - pad, x + pad:x + cell_width - pad] = resized_tile

                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict
    
def target2_label(x):
    if x in ["MGA","MCN","TBM"]:
        return 0
    elif x in ["IGA","MSP","PRU"]:
        return 1
    elif x in ["MEN","LUE"]:
        return 2
    elif x in ["AMY"]:
        return 7
    elif x in ["MPG"]:
        return 8
    elif x in ["ANC","GBM"]:
        return 4
    elif x in ["END"]:
        return 9
    elif x in ["TIN","ATN","ATI","IGG"]:
        return 6
    elif x in ["BNS"]:
        return 5
    elif x in ["FGS","OBE"]:
        return 9
    elif x in ["DMN"]:
        return 3
    elif x in ["FAB"]:
        return 11
    elif x in ["LUD"]:
        return 12
    elif x in ["LCH"]:
        return 13#5
    elif x in ["SLC","SCL"]:
        return 14#4

    
    elif x in ["MNS"]:
        return 15#4
    elif x in ["ALP"]:
        return 16#4
    elif x in ["TMA"]:
        return 17#3


    elif x in ["PRE"]:
        return 18
    elif x in ["CCE"]:
        return 19 #1
    elif x in ["OTH"]:
        return 20 #1
    elif x in ["CNI"]:
        return 21 #1
    
    elif x in ["LUC"]:
        return 22 #1
df["label"] =  np.vectorize(target2_label)(
    df["target"].to_numpy())
df["label"]=np.where(df["label"]>4,4,df["label"])

test_labels = df["label"].values
targets_cm = ["MGA_MCN_TBM","IGA_MSP_PRU","MEN_LUE","DMN","other"]
clist = ["orange","red","blue","black","green"]
clist_ = [[255, 165, 0],[255, 0, 0],[0, 0, 255]
         ,[0, 0, 0],[0, 255, 0]]
cls_2_color = dict(zip(range(len(clist_)),clist_))
n_neighbors = 30
seed = 1000

kills_reduced = np.load(f"/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/test_umap_seed{seed}_n{n_neighbors}.npy")


df["tsen_0"]=kills_reduced[:,0]
df["tsen_1"]=kills_reduced[:,1]
plt.figure(figsize=(25,25))

for idx,(cls_name,color) in enumerate(zip(targets_cm,clist)):
    tmp = df[df["label"]==idx].reset_index(drop=True)
    x = tmp["tsen_0"].values
    y = tmp["tsen_1"].values
    plt.scatter(x,y,c=color,label=cls_name,s=25)
    
#plt.savefig("/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/tsne_p30_scatter.jpg")
plt.savefig(f"/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/umap_seed{seed}_n{n_neighbors}_scatter.jpg")

canvas, img_idx_dict = plot_tiles_color(path_l, kills_reduced,test_labels, grid_units=30)
plt.figure(figsize=(25,25))
plt.imshow(canvas)
ax = plt.gca()

for cls_name,color in zip(targets_cm,clist):
    ax.plot([], [], marker="s", color=color, label=cls_name, linestyle="none")

ax.legend(frameon=True,loc='lower left', handletextpad=0, ncol=1, columnspacing=1,fontsize=25,markerscale=4)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.title("Title ",fontsize=40)
plt.savefig(f"/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/umap_seed{seed}_n{n_neighbors}.jpg")
#plt.savefig("/home/abe/KidneyM/dino/pas_glomerulus_wbf_L/tsne_p30.jpg")
