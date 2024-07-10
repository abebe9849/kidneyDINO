import openslide

import openslide
from PIL import Image
import pandas as pd
import numpy as np
import cv2
# WSIファイルのパスを指定

ROOT = "/home/user/20240707_KPMP_PAS_with_MASK/"


def read_crop(slide,xyxy):
    margin_ratio = 0.1
    ymin,xmin,ymax,xmax = xyxy
    width = xmax - xmin
    height = ymax - ymin
    max_side = max(width, height)

    # 正方形の中心を計算
    center_x = xmin + width / 2
    center_y = ymin + height / 2

    # 新しい正方形のbboxを計算
    new_xmin = center_x - max_side / 2
    new_ymin = center_y - max_side / 2
    new_xmax = center_x + max_side / 2
    new_ymax = center_y + max_side / 2

    # マージンを計算
    margin = max_side * margin_ratio

    # マージンを追加したbboxを計算
    new_xmin_with_margin = int(new_xmin - margin)
    new_ymin_with_margin = int(new_ymin - margin)
    new_xmax_with_margin = int(new_xmax + margin)
    new_ymax_with_margin = int(new_ymax + margin)


    return slide.read_region((new_xmin_with_margin,new_ymin_with_margin),0,(new_xmax_with_margin-new_xmin_with_margin,new_ymax_with_margin-new_ymin_with_margin))

import glob,os,tqdm
size_ = []
from multiprocessing import Pool
def get_image_height(wsi_path):
    img = cv2.imread(wsi_path)
    try:

        return img.shape[0]
    except:
        print(wsi_path,img)
        return 512
    


pool = Pool()
    # 画像ファイルのリストを取得
wsi_paths = glob.glob("*png")

    # 並列処理を実行
#size_ = list(tqdm.tqdm(pool.imap(get_image_height, wsi_paths), total=len(wsi_paths)))

import matplotlib.pyplot as plt
#plt.hist(size_,bins=50)
#plt.savefig("/home/user/ABE/kidneyDINO/KPMP_his_half.png")

#exit()

for wsi_path in tqdm.tqdm(glob.glob(ROOT+"*.svs")[:]):
    file_ID = wsi_path.split("/")[-1].split(".")[0]

    excel_file = wsi_path.replace(".svs","_features.xlsx")
    maskpath = wsi_path.replace(".svs",".ome.tif")
    dfs= pd.read_excel(excel_file, sheet_name=None)


    df = dfs['non_gs_glomeruli']
    df1 = dfs['gs_glomeruli']
    slide = openslide.OpenSlide(wsi_path)
    for i in range(df.shape[0]):
        xyxy = df.iloc[i][["x1","y1","x2","y2"]].values
        if os.path.exists(f"crop__{file_ID}__non_gs_{i}.png"):
            continue
        c = read_crop(slide,xyxy)
        region = c.convert("RGB")
        region.save(f"crop__{file_ID}__non_gs_{i}.png", 'PNG')

    for i in range(df1.shape[0]):
        xyxy = df1.iloc[i][["x1","y1","x2","y2"]].values
        if os.path.exists(f"crop__{file_ID}__gs_{i}.png"):
            continue
        c = read_crop(slide,xyxy)
        region = c.convert("RGB")
        region.save(f"crop__{file_ID}__gs_{i}.png", 'PNG')

