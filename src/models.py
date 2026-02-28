import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiModalBackbone(nn.Module):
    """
    Encoder backbone for EO and SAR modalities.
    """
    def __init__(self, model_type='resnet50', pretrained=True, in_channels=3):
        super(MultiModalBackbone, self).__init__()
        if model_type == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            if in_channels != 3:
                self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_type == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            if in_channels != 3:
                # EfficientNet-B0 first conv is features[0][0]
                self.model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.feature_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, x):
        return self.model(x)

class ModalityProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256, dropout=0.3):
        super(ModalityProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.projector(x)

class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ContrastiveHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)

class MAVIC_V2_Model(nn.Module):
    def __init__(self, num_classes=10, eo_backbone='efficientnet_b0', sar_backbone='resnet50', proj_dim=256):
        super(MAVIC_V2_Model, self).__init__()
        
        # Encoders
        self.eo_encoder = MultiModalBackbone(model_type=eo_backbone, in_channels=3)
        self.sar_encoder = MultiModalBackbone(model_type=sar_backbone, in_channels=1)
        
        # Projectors to a shared space
        self.eo_projector = ModalityProjector(self.eo_encoder.feature_dim, out_dim=proj_dim)
        self.sar_projector = ModalityProjector(self.sar_encoder.feature_dim, out_dim=proj_dim)
        
        # Contrastive Head
        self.contrastive_head = ContrastiveHead(proj_dim)
        
        # Classification Head
        # If training joint, we fuse. If inference is SAR-only, we use SAR projector.
        # To make it robust, we'll have a head that takes the projected feature.
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, eo, sar, mode='joint'):
        """
        mode: 'joint', 'eo_only', 'sar_only'
        """
        if mode == 'joint' or mode == 'eo_only':
            eo_feat = self.eo_encoder(eo)
            eo_proj = self.eo_projector(eo_feat)
        
        if mode == 'joint' or mode == 'sar_only':
            sar_feat = self.sar_encoder(sar)
            sar_proj = self.sar_projector(sar_feat)
            
        if mode == 'joint':
            # During joint training, we can return both for KD and Contrastive loss
            return eo_proj, sar_proj, self.classifier(eo_proj), self.classifier(sar_proj)
        elif mode == 'eo_only':
            return eo_proj, self.classifier(eo_proj)
        elif mode == 'sar_only':
            return sar_proj, self.classifier(sar_proj)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_contrastive_features(self, x_proj):
        return self.contrastive_head(x_proj)
