import cuml
import cudf,glob
from cuml.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import cupy as cp
import pandas as pd
import sys,os,time
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold


model_type = "dino/upc"
model_type =sys.argv[1]
task = model_type.split("/")[-1]



all_df = pd.read_csv("/home/abe/KidneyM/data/final_2_pas.csv")
#dic_ = {"egfr":60,"alb":3,"uprot":3.5,"DM":1,"DBP":90,"収縮期血圧":140,"血尿":1}
COL = task
if COL=="sbp":
    COL="収縮期血圧"
elif COL=="hep":
    COL="血尿"
test_label = test_df["label"].to_numpy()


test_data = cp.load(f"/home/abe/KidneyM/dino/pas_glomerulus_exp001/KNN/{model_type}/dino_test.npy")
test_predictions_ = [[],[],[],[]]

k_ = [5,10,20,50]
for i in range(4):
    tra = folds[folds["fold"]!=i].reset_index(drop=True)
    val = folds[folds["fold"]==i].reset_index(drop=True)
    train_data = cp.load(f"/home/abe/KidneyM/dino/pas_glomerulus_exp001/KNN/{model_type}/dino_fold{i}_tra.npy")
    val_data = cp.load(f"/home/abe/KidneyM/dino/pas_glomerulus_exp001/KNN/{model_type}/dino_fold{i}_val.npy")
    train_labels = tra["label"].values
    val_labels = val["label"].values
    
    for i,k in enumerate(k_):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_labels)


        val_predictions = knn.predict_proba(val_data)[:,1]
        test_predictions = knn.predict_proba(test_data)[:,1]
        test_predictions_[i].append(test_predictions)
        
        val_auc = roc_auc_score(cp.asnumpy(val_labels),cp.asnumpy(val_predictions))
    #print(val_auc)
    
    
print(task)


from sklearn.metrics import precision_recall_curve, auc

def PRAUC(label, pred_):
    precision, recall, _ = precision_recall_curve(label, pred_)
    return auc(recall, precision)

for i in range(len(test_predictions_)):
    pred = cp.stack(test_predictions_[i],axis=-1)
    pred = cp.asnumpy(cp.mean(pred,axis=-1))
    
    test_df["pred_mean_1"]=pred
    
    sub_g = test_df.groupby("WSI")[["label","pred_mean_1"]].mean()
        
    test_auc = roc_auc_score(sub_g["label"],sub_g["pred_mean_1"])
    pr_auc =PRAUC(sub_g["label"],sub_g["pred_mean_1"])
    print(pr_auc)



    




