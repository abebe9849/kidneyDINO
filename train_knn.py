import cuml
import cudf
from cuml.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import cupy as cp
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

def multiclass_prauc(y_true, y_pred, n_classes):
    """
    Calculate the Precision-Recall AUC for each class in a multi-class classification.

    Args:
    y_true (array-like): True class labels, array of shape (n_samples,)
    y_pred (array-like): Predicted probabilities, array of shape (n_samples, n_classes)
    n_classes (int): Number of classes

    Returns:
    dict: A dictionary with class indices as keys and corresponding PRAUC scores as values
    """
    # Initialize dictionary to store PRAUC for each class
    prauc_scores = 0
    
    # Convert labels to one-hot encoded format
    y_true_encoded = np.eye(n_classes)[y_true]
    
    for i in range(n_classes):
        # Extract true labels and predicted probabilities for class i
        true_class = y_true_encoded[:, i]
        pred_probs = y_pred[:, i]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(true_class, pred_probs)
        
        # Calculate AUC for the precision-recall curve
        pr_auc = auc(recall, precision)
        
        # Store the PRAUC score for the class
        prauc_scores+=pr_auc

    return prauc_scores/4

# Example usage:
# y_true = [0, 1, 2, 0, 1, 2]  # True labels
# y_pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9],
#                    [0.85, 0.1, 0.05], [0.05, 0.8, 0.15], [0.1, 0.2, 0.7]])  # Predicted probabilities
# n_classes = 3  # Number of classes

# prauc_scores = multiclass_prauc(y_true, y_pred, n_classes)
# print(prauc_scores)


# データの準備
# train_data, train_labels, val_data, val_labels をここで定義します。
# 例: train_data = cudf.DataFrame(...), train_labels = cudf.Series(...)

# kNNモデルの訓練

df = cudf.read_csv("/home/abe/KidneyM/dino/FIX/WO_kouka/B600__4cls/oof.csv")


sub = pd.read_csv("/home/abe/KidneyM/dino/FIX/WO_kouka/B600__4cls/sub.csv")
test_label = sub["label"].to_numpy()
model_type = "dino"
test_data = cp.load(f"/home/abe/KidneyM/dino/pas_glomerulus_exp001/KNN/OLD/{model_type}/dino_test.npy")
test_predictions_ = []

for k in [5,10,20,50]:
    test_predictions_ = []
    for i in range(4):
        tra = df[df["fold"]!=i].reset_index(drop=True)
        val = df[df["fold"]==i].reset_index(drop=True)
        train_data = cp.load(f"/home/abe/KidneyM/dino/pas_glomerulus_exp001/KNN/OLD/{model_type}/dino_fold{i}_tra.npy")
        val_data = cp.load(f"/home/abe/KidneyM/dino/pas_glomerulus_exp001/KNN/OLD/{model_type}/dino_fold{i}_val.npy")
        train_labels = tra["label"].values
        val_labels = val["label"].values
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_labels)


        val_predictions = knn.predict_proba(val_data)
        test_predictions = knn.predict_proba(test_data)
        test_predictions_.append(test_predictions)
    
        val_auc = roc_auc_score(cp.asnumpy(val_labels),cp.asnumpy(val_predictions),multi_class="ovr")
        #print(val_auc)
    
    test_predictions_ = cp.stack(test_predictions_,axis=-1)
    test_predictions_ = cp.mean(test_predictions_,axis=-1)
    
    test_auc = roc_auc_score(test_label,cp.asnumpy(test_predictions_),multi_class="ovr")
    #print(test_auc)

    pr =multiclass_prauc(test_label,cp.asnumpy(test_predictions_),4)
    print(pr)

