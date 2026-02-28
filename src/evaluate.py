import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import torch.nn.functional as F

def compute_ood_scores(logits, method='max_logit'):
    """
    Compute OOD scores (higher score means more likely to be IN-DISTRIBUTION, 
    lower score means more likely OOD).
    """
    if method == 'max_logit':
        return torch.max(logits, dim=1)[0]
    elif method == 'energy':
        # E = -T * logsumexp(logits / T)
        # We want higher score for ID, so we use logsumexp directly
        return torch.logsumexp(logits, dim=1)
    elif method == 'softmax':
        probs = F.softmax(logits, dim=1)
        return torch.max(probs, dim=1)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

def evaluate_model(model, dataloader, device, class_to_idx):
    model.eval()
    all_logits = []
    all_labels = []
    all_ood_flags = []
    
    with torch.no_grad():
        for batch in dataloader:
            sar = batch['sar'].to(device)
            labels = batch['label'].to(device)
            ood_flags = batch['ood_flag']
            
            _, logits = model(None, sar, mode='sar_only')
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_ood_flags.append(ood_flags)
            
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_ood_flags = torch.cat(all_ood_flags, dim=0).numpy()
    
    # 1. ID Accuracy
    id_mask = (all_ood_flags == 0)
    id_logits = all_logits[id_mask]
    id_labels = all_labels[id_mask]
    
    preds = torch.argmax(id_logits, dim=1).numpy()
    acc = accuracy_score(id_labels, preds)
    
    # 2. AUROC (OOD Detection)
    # Binary labels for AUROC: 1 for ID, 0 for OOD
    bin_labels = (all_ood_flags == 0).astype(int)
    scores = compute_ood_scores(all_logits, method='max_logit').numpy()
    
    auroc = roc_auc_score(bin_labels, scores)
    
    # 3. FPR@95TPR (Standard OOD metric)
    fpr95 = compute_fpr95(bin_labels, scores)
    
    # Final Competition Score
    final_score = 0.75 * acc + 0.25 * auroc
    
    return {
        'acc': acc,
        'auroc': auroc,
        'fpr95': fpr95,
        'final_score': final_score
    }

def compute_fpr95(bin_labels, scores):
    """
    Compute False Positive Rate at 95% True Positive Rate.
    bin_labels: 1 for ID, 0 for OOD
    scores: higher for ID
    """
    id_scores = scores[bin_labels == 1]
    ood_scores = scores[bin_labels == 0]
    
    if len(ood_scores) == 0:
        return 0.0
        
    threshold = np.percentile(id_scores, 5) # 5th percentile of ID scores
    fpr = np.sum(ood_scores >= threshold) / len(ood_scores)
    return fpr

def normalize_logits(logits):
    """
    Apply z-score normalization across the batch for potentially better calibration.
    As seen in Rank 1 approach.
    """
    mean = logits.mean(dim=0, keepdim=True)
    std = logits.std(dim=0, keepdim=True)
    return (logits - mean) / (std + 1e-5)
