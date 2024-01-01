import glob,cv2
import cv2,tqdm
import matplotlib.pyplot as plt
import numpy as np

imnet = glob.glob("/home/abe/KidneyM/dino/dino-vit-features/size1024_imnet/*")

histogram_R_total = np.zeros((256, 1))
histogram_G_total = np.zeros((256, 1))
histogram_B_total = np.zeros((256, 1))

# 各画像に対して処理を行う
for img_path in imnet:
    # 画像を読み込む
    img = cv2.imread(img_path)
    img = img[:1024,2048:,:]

    # OpenCVはBGRフォーマットで画像を読み込むので、RGBに変換する
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RGBチャンネルを分割
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    # 各チャンネルのヒストグラムを計算し、累積する
    histogram_R = cv2.calcHist([R], [0], None, [256], [0, 256])
    histogram_G = cv2.calcHist([G], [0], None, [256], [0, 256])
    histogram_B = cv2.calcHist([B], [0], None, [256], [0, 256])

    histogram_R_total += histogram_R
    histogram_G_total += histogram_G
    histogram_B_total += histogram_B

rgb = np.stack([histogram_R_total,histogram_G_total,histogram_B_total])
print(rgb.shape,histogram_R_total.shape)
np.save("/home/abe/KidneyM/dino/dino-vit-features/imnet_rgb_histogram.npy",rgb)

# 各チャンネルの累積ヒストグラムを描画
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(histogram_R_total, color='red')
plt.title('Red Histogram')

plt.subplot(1, 3, 2)
plt.plot(histogram_G_total, color='green')
plt.title('Green Histogram')

plt.subplot(1, 3, 3)
plt.plot(histogram_B_total, color='blue')
plt.title('Blue Histogram')

plt.savefig("/home/abe/KidneyM/dino/dino-vit-features/imnet_rgb.png")




imnet = glob.glob("/home/abe/KidneyM/dino/dino-vit-features/size1024_dino/*")




# 各チャンネルのヒストグラムを累積するための配列を初期化
histogram_R_total = np.zeros((256, 1))
histogram_G_total = np.zeros((256, 1))
histogram_B_total = np.zeros((256, 1))

# 各画像に対して処理を行う
for img_path in imnet:
    # 画像を読み込む
    img = cv2.imread(img_path)
    img = img[:1024,2048:,:]

    # OpenCVはBGRフォーマットで画像を読み込むので、RGBに変換する
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RGBチャンネルを分割
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    # 各チャンネルのヒストグラムを計算し、累積する
    histogram_R = cv2.calcHist([R], [0], None, [256], [0, 256])
    histogram_G = cv2.calcHist([G], [0], None, [256], [0, 256])
    histogram_B = cv2.calcHist([B], [0], None, [256], [0, 256])

    histogram_R_total += histogram_R
    histogram_G_total += histogram_G
    histogram_B_total += histogram_B

rgb = np.stack([histogram_R_total,histogram_G_total,histogram_B_total])
print(rgb.shape,histogram_R_total.shape)
np.save("/home/abe/KidneyM/dino/dino-vit-features/dino_rgb_histogram.npy",rgb)

# 各チャンネルの累積ヒストグラムを描画
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(histogram_R_total, color='red')
plt.title('Red Histogram')

plt.subplot(1, 3, 2)
plt.plot(histogram_G_total, color='green')
plt.title('Green Histogram')

plt.subplot(1, 3, 3)
plt.plot(histogram_B_total, color='blue')
plt.title('Blue Histogram')

plt.savefig("/home/abe/KidneyM/dino/dino-vit-features/dino_rgb.png")


imnet = np.load("/home/abe/KidneyM/dino/dino-vit-features/imnet_rgb_histogram.npy")

dino = np.load("/home/abe/KidneyM/dino/dino-vit-features/dino_rgb_histogram.npy")

for i in range(3):
    histogram_R_total = dino[i]
    histogram_G_total = imnet[i]
    variance_R = np.var(histogram_R_total)
    std_dev_R = np.sqrt(variance_R)

    variance_G = np.var(histogram_G_total)
    std_dev_G = np.sqrt(variance_G)

    # 結果を出力
    print(f"dino  - Variance: {variance_R}, Standard Deviation: {std_dev_R}")
    print(f"imnet  - Variance: {variance_G}, Standard Deviation: {std_dev_G}")
    
    print(f"dino/imnet  - Variance: {variance_R/variance_G}, Standard Deviation: {std_dev_R/std_dev_G}")
    
