

import numpy as np
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable
import japanize_matplotlib
from scipy import stats

def calc_auc_s(tests):
    y_true = tests[0]["label"]
    aucs = np.array([roc_auc_score(y_true, test["pred_mean_1"].to_numpy()) for test in tests])
    mean_auc = np.mean(aucs)
    

    # 標準偏差  
    std_auc = np.std(aucs)

    # 95%信頼区間の計算
    confidence = 0.95

    lower, upper = stats.norm.interval(confidence, loc=mean_auc, scale=std_auc/np.sqrt(len(aucs)))
    ci_95 = "[{:.3f} - {:.3f}]".format(lower, upper)
    print(f"wo Gby AUC: {mean_auc:.3f} {ci_95}")
    
    ##groupby
    
    tests = [t.groupby("WSI")[["label","pred_mean_1"]].mean() for t in tests]
    
    y_true = tests[0]["label"]
    aucs = np.array([roc_auc_score(y_true, test["pred_mean_1"].to_numpy()) for test in tests])
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 95%信頼区間の計算
    confidence = 0.95

    lower, upper = stats.norm.interval(confidence, loc=mean_auc, scale=std_auc/np.sqrt(len(aucs)))
    ci_95 = "[{:.3f} - {:.3f}]".format(lower, upper)
    print(f"G Avg AUC: {mean_auc:.3f} {ci_95}")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,title__ = None,
                          cmap=plt.cm.Blues):
    """
    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=40)
    plt.yticks(tick_marks, fontsize=40)
    #plt.xlabel('Predicted label',fontsize=25)
    #plt.ylabel('True label', fontsize=25)
    #plt.title(title, fontsize=30)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0.15)
    if normalize:
        im.set_clim(0,1)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title__,
           #ylabel='True label',
           #xlabel='Predicted label'
           )
    ax.set_title(title__, fontsize=50)
    ax.set_ylabel('True label', fontsize=45)
    ax.set_xlabel('Predicted label', fontsize=45)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontsize=45,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    fig.tight_layout()
    title = title.replace(" ","_")
    fig.savefig(f"{title}.png",dpi=280)
    return ax

from sklearn.metrics import roc_auc_score, roc_curve

def plot_AUC(targets_cm,y_true, y_pred,save_name):
    auc_val = roc_auc_score(y_true, y_pred, multi_class="ovr")

    # ROC曲線の計算
    y_true = np.eye(4)[y_true]

    # クラスごとのROC曲線計算
    fpr = {}
    tpr = {}
    auc_ = []
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc_.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        
    # プロット    
    plt.figure(figsize=(5,5))
    for i in range(y_pred.shape[1]):
        label = f"{targets_cm[i]} :{auc_[i]:.3f}"
        plt.plot(fpr[i], tpr[i], label=label)
    plt.legend(loc="lower right",fontsize=12) # 凡例 
    s = save_name.split("/")[-1].split(".")[0]
    plt.title(f"{s}={auc_val:.3f}",fontsize=20)   
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    
    plt.savefig(save_name)
    

def plot_AUC_each(targets_cm,y_true, y_pred,y_pred_dino,save_name):
    auc_val = roc_auc_score(y_true, y_pred, multi_class="ovr")
    auc_dino = roc_auc_score(y_true, y_pred_dino, multi_class="ovr")

    # ROC曲線の計算
    y_true = np.eye(4)[y_true]

    # クラスごとのROC曲線計算
    fpr = {}
    tpr = {}
    auc_ = []
    for i in range(4):
        
        auc_dino=  roc_auc_score(y_true[:, i], y_pred_dino[:, i])
        auc_imnet=  roc_auc_score(y_true[:, i], y_pred[:, i])
        
        plt.figure(figsize=(5,5))
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_dino[:, i])
        label = f"DINO pretrained :{auc_dino:.3f}"
        plt.plot(fpr, tpr, label=label)
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        label = f"ImageNet pretrained :{auc_imnet:.3f}"
        plt.plot(fpr, tpr, label=label)

        plt.legend(loc="lower right",fontsize=13) # 凡例 
        s = save_name.split("/")[-1].split(".")[0]
        plt.title(f"{targets_cm[i]}",fontsize=20)   
        plt.xlabel("FPR")
        plt.ylabel("TPR")
    
        plt.savefig(save_name+f"_{targets_cm[i]}.png")
    
def calc_auc_4(tests):
    y_true = tests[0]["label"]
    cols = [f"pred_mean_{i}" for i in range(4)]
    aucs = np.array([roc_auc_score(y_true, test[cols].to_numpy(), multi_class="ovr") for test in tests])
    mean_auc = np.mean(aucs)
    

    # 標準偏差  
    std_auc = np.std(aucs)

    # 95%信頼区間の計算
    confidence = 0.95

    lower, upper = stats.norm.interval(confidence, loc=mean_auc, scale=std_auc/np.sqrt(len(aucs)))
    ci_95 = "[{:.3f} - {:.3f}]".format(lower, upper)
    print(f"Average AUC: {mean_auc:.3f} {ci_95}")
    
def calc_F1_4(tests):
    y_true = tests[0]["label"]
    cols = [f"pred_mean_{i}" for i in range(4)]
    aucs = np.array([f1_score(y_true, np.argmax(test[cols].to_numpy(),axis=-1),average="macro") for test in tests])
    mean_auc = np.mean(aucs)
    

    # 標準偏差  
    std_auc = np.std(aucs)

    # 95%信頼区間の計算
    
    confidence = 0.95

    lower, upper = stats.norm.interval(confidence, loc=mean_auc, scale=std_auc/np.sqrt(len(aucs)))
    ci_95 = "[{:.3f} - {:.3f}]".format(lower, upper)
    print(f"Average F1: {mean_auc:.3f} {ci_95}")

def calc_F1_2(tests):
    y_true = tests[0]["label"]
    aucs = np.array([f1_score(y_true, np.where(test["pred_mean_1"].to_numpy()>0.5,1,0)) for test in tests])
    mean_auc = np.mean(aucs)    # 標準偏差  
    std_auc = np.std(aucs)
    confidence = 0.95

    lower, upper = stats.norm.interval(confidence, loc=mean_auc, scale=std_auc/np.sqrt(len(aucs)))
    ci_95 = "[{:.3f} - {:.3f}]".format(lower, upper)
    print(f"wo Gby F1: {mean_auc:.3f} {ci_95}")
    
    ##groupby
    
    tests = [t.groupby("WSI")[["label","pred_mean_1"]].mean() for t in tests]
    
    y_true = tests[0]["label"]
    aucs = np.array([f1_score(y_true, np.where(test["pred_mean_1"].to_numpy()>0.5,1,0)) for test in tests])
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # 95%信頼区間の計算
    confidence = 0.95

    lower, upper = stats.norm.interval(confidence, loc=mean_auc, scale=std_auc/np.sqrt(len(aucs)))
    ci_95 = "[{:.3f} - {:.3f}]".format(lower, upper)
    print(f"G Avg F1: {mean_auc:.3f} {ci_95}")

import os,glob

targets_cm = ["MGA_MCN_TBM","IGA_MSP_PRU","MEN_LUE","DMN"]
targets_cm = ["微小糸球体病変","メサンギウム増殖性糸球体腎炎","膜性腎症","糖尿病性腎症"]


cols = [f"pred_mean_{i}" for i in range(4)]


def calc_4cls(dir_):
    tests = glob.glob(f"{dir_}*sub*csv")
    print(tests)
    tests = [pd.read_csv(i) for i in tests]
    calc_F1_4(tests)
    calc_auc_4(tests)
    
def calc_2cls(dir_):
    tests = glob.glob(f"{dir_}*sub*csv")
    tests = [pd.read_csv(i) for i in tests]
    calc_auc_s(tests)
    #calc_F1_2(tests)
    

calc_4cls("{DIR}")
