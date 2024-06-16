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
#if task =="alb":

#    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__Alb_over3.0/oof.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__Alb_over3.0/sub.csv")
#elif task =="dbp":
#    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__dBP_over90/oof.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__dBP_over90/sub.csv")
#elif task =="DM":
#    folds = pd.read_csv("/home/abe/KidneyM/dino/src/outputs/2024-04-12/DM_FT/oof_exp__DM_imnet_FT.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/src/outputs/2024-04-12/DM_FT/sub_exp__DM_imnet_FT.csv")
#elif task =="sbp":
#    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__sBP_seed2021_over140/oof.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__sBP_seed2021_over140/sub.csv")
#elif task =="hep":
#    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__血尿_over1/oof.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__血尿_over1/sub.csv")
#elif task =="egfr":
#    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__eGFR_over60/oof.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__eGFR_over60/sub.csv")
#elif task =="upc":
#    folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__UPC_seed2023_over3.5/oof.csv")
#    test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__UPC_seed2023_over3.5/sub.csv")


if 1:
    if task =="alb":

        #folds = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__Alb_over3.0/oof.csv")                                                        
        #test_df = pd.read_csv("/home/abe/KidneyM/dino/FIX/B600__Alb_over3.0/sub.csv")                                                      
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/alb_dino/oof*")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/alb_dino/sub*")[0])
    elif task =="dbp":
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/dbp_dino/oof*")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/dbp_dino/sub*")[0])
    elif task =="DM":
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/DM_dino/oof*")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/DM_dino/sub*")[0])
    elif task =="sbp":
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-14/sbp_dino/oof*csv")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-14/sbp_dino/sub*csv")[0])
    elif task =="hep":
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-14/hep_dino/oof*csv")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-14/hep_dino/sub*csv")[0])
    elif task =="egfr":
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/egfr_dino/oof*")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/egfr_dino/sub*")[0])
    elif task =="upc":
        folds = pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/upc_dino/oof*")[0])
        test_df =  pd.read_csv(glob.glob("/home/abe/KidneyM/dino/src/outputs/2024-04-13/upc_dino/sub*")[0])








all_df = pd.read_csv("/home/abe/KidneyM/data/final_2_pas.csv")
#dic_ = {"egfr":60,"alb":3,"uprot":3.5,"DM":1,"DBP":90,"収縮期血圧":140,"血尿":1}
COL = task
if COL=="sbp":
    COL="収縮期血圧"
elif COL=="hep":
    COL="血尿"
#CONF = dic_[COL]
#all_df = all_df[all_df[COL]!=-1].reset_index(drop=True)

#all_df["label"] =  np.where(all_df[COL] >=CONF,1,0)

#N_fold = 5

#n_target = all_df["label"].nunique()
#sgkf = StratifiedGroupKFold(n_splits=N_fold,random_state=2025,shuffle=True)
#for fold, ( _, val_) in enumerate(sgkf.split(X=all_df, y=all_df.label.to_numpy(),groups=all_df.WSI)):
#    all_df.loc[val_ , "fold"] = fold

#    val_df = all_df[all_df["fold"]==fold]
            #print(val_df["label"].nunique())                                                             
#folds = all_df[all_df["fold"]!=N_fold-1].reset_index(drop=True)
#test_df = all_df[all_df["fold"]==N_fold-1].reset_index(drop=True)

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



    




