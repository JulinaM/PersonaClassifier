import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import re,os, glob, traceback, nltk, logging, sys

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc

def generate_auroc(a_output, model, ckpt, savefig=True):      
    plt.figure(figsize=(8, 6))
    for trait, (y_true, y_pred, y_scores) in a_output.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_index = tpr - fpr
        optimal_idx = youden_index.argmax()
        optimal_threshold = thresholds[optimal_idx]
        logging.info(f"Optimal Threshold: {model}: {optimal_threshold}")
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{trait} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guessing")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Multiple (Trait) Classifiers using {model}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if savefig: plt.savefig(f'{ckpt}/{model}_auroc.png')
    plt.show()


def generate_cm(a_output, model, ckpt, savefig=True):
    n_classifiers = len(a_output)
    fig, axes = plt.subplots(1, n_classifiers, figsize=(5 * n_classifiers, 5))
    results = []
    for ax, (trait, (y_true, y_pred, _)) in zip(axes, a_output.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        ax.set_title(trait)

        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # Avoid division by zero
        results.append( {
            "Model": model,
            "Classifier": trait,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Specificity": specificity,
            "False Positive Rate": false_positive_rate,
            "Confusion Matrix": cm 
        })
    plt.tight_layout()
    if savefig: plt.savefig(f"{ckpt}/{model}_cm.png")
    return results

    
def display_metrics(savefig=True):
    logging.info(f'generating metrics and confusion matrix ..')
    performance_records = [] 
    for model in self.models:
        logging.info(15*'='+f" {model} "+ 15*'=')
        a_output = self.all_outputs[model]
        n_classifiers = len(a_output)
        
    performance_df = pd.DataFrame(performance_records)
    logging.info(f"Performance metrics dataframe created with shape: {performance_df.shape}")
    performance_df.to_csv(f"{ckpt}/performance.csv")
    for col in self.traits:
        s = performance_df[performance_df['Classifier'] ==col]
        best_model_row = s.loc[s['Accuracy'].idxmax()]
        logging.info(f'For {best_model_row["Classifier"]}, {best_model_row["Model"]},  {best_model_row["Accuracy"]}')


#TODO
def explain_SHAP(self, savefig=True):
    import shap
    shap.initjs()
    explainer = shap.Explainer(self.lr_model, self.X_train)
    shap_values = explainer(self.X_val)
    shap.plots.beeswarm(shap_values)
    if savefig: plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight') 
    plt.show()