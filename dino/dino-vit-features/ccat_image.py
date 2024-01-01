import glob,cv2
import numpy as np 

for i in glob.glob("/home/abe/KidneyM/dino/dino-vit-features/size1024_/*.png"):
    dino = cv2.imread(i)
    path = i.replace("size1024_","size1024___imnet")
    imnet  = cv2.imread(path)
    p = path.split("/")[-1]
    ccat = np.concatenate([dino,imnet],axis=1)
    cv2.imwrite(p,ccat)