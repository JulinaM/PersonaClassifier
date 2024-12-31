import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import re,os, glob, traceback, nltk, logging, sys
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.calibration import calibration_curve

def display_calibration(y_test, y_prob, target, filepath=None):
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel(f'Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve for {target}')
    plt.legend()
    if filepath: plt.savefig(filepath)
    plt.show()

def display_auroc(y_test, y_prob, target, filepath=None):
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) for {target}')
    plt.legend(loc="lower right")
    if filepath: plt.savefig(filepath)
    plt.show()


def generate_auroc(a_output, model, filepath=None):      
    plt.figure(figsize=(8, 6))
    for trait, (y_true, _, y_scores) in a_output.items():
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_index = tpr - fpr
        optimal_idx = youden_index.argmax()
        optimal_threshold = thresholds[optimal_idx]
        logging.info(f"Optimal Threshold: {model}:{trait}: {optimal_threshold}")
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
    if filepath: plt.savefig(filepath)
    plt.show()


def generate_cm(a_output, filepath=None):
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
            # "Model": model,
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
    if filepath: plt.savefig(filepath)
    return results


#TODO
def explain_SHAP(self, savefig=True):
    import shap
    shap.initjs()
    explainer = shap.Explainer(self.lr_model, self.X_train)
    shap_values = explainer(self.X_val)
    shap.plots.beeswarm(shap_values)
    if savefig: plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight') 
    plt.show()