import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        return loss.mean()

class FeatureMatchingLoss(nn.Module):
    """
    Knowledge Distillation loss between Teacher (EO) and Student (SAR) features.
    """
    def __init__(self, loss_type='mse'):
        super(FeatureMatchingLoss, self).__init__()
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'cosine':
            self.criterion = lambda x, y: 1 - F.cosine_similarity(x, y).mean()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, sar_feat, eo_feat):
        return self.criterion(sar_feat, eo_feat)

def label_smoothed_ce_loss(logits, labels, smoothing=0.1):
    """
    Cross Entropy Loss with Label Smoothing.
    """
    return F.cross_entropy(logits, labels, label_smoothing=smoothing)
