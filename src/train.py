import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MAVIC_V2_Model
from data import get_transforms, EOSARDataset, SAROnlyDataset, mixup_data, mixup_criterion
from losses import label_smoothed_ce_loss, FeatureMatchingLoss, SupConLoss
from evaluate import evaluate_model

# Configuration
CONFIG = {
    'train_sar_root': './pbvs_mavic_dataset/train/SAR_Train',
    'train_eo_root':  './pbvs_mavic_dataset/train/EO_Train',
    'val_sar_root':   './pbvs_mavic_dataset/val',
    'val_csv_path':   './pbvs_mavic_dataset/Validation_reference.csv',
    'num_classes': 10,
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-4,
    'wd': 1e-5,
    'alpha_mixup': 1.0,
    'kd_weight': 0.1,
    'con_weight': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train():
    # Verify data existence
    if not os.path.exists(CONFIG['train_sar_root']):
        print(f"ERROR: Dataset not found at {CONFIG['train_sar_root']}")
        print("Please ensure you have extracted 'pbvs_dataset.zip' into the 'pbvs_mavic_dataset' folder.")
        print("You can use 'python dataset_utils.py' if the zip is in the root directory.")
        return

    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")

    # Data
    eo_tr, sar_tr, sar_val = get_transforms()
    train_dataset = EOSARDataset(CONFIG['train_sar_root'], CONFIG['train_eo_root'], sar_tr, eo_tr)
    val_dataset = SAROnlyDataset(CONFIG['val_sar_root'], CONFIG['val_csv_path'], train_dataset.class_to_idx, sar_val)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # Model
    model = MAVIC_V2_Model(num_classes=CONFIG['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # Losses
    kd_loss_fn = FeatureMatchingLoss(loss_type='mse')
    con_loss_fn = SupConLoss(temperature=0.07)

    # Scaler for Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_score = 0.0
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch in pbar:
            sar = batch['sar'].to(device)
            eo = batch['eo'].to(device)
            labels = batch['label'].to(device)
            
            # MixUp
            sar, eo, labels_a, labels_b, lam = mixup_data(sar, eo, labels, CONFIG['alpha_mixup'])
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                # Forward
                eo_proj, sar_proj, eo_logits, sar_logits = model(eo, sar, mode='joint')
                
                # 1. Classification Loss (on both modalities)
                loss_eo = mixup_criterion(label_smoothed_ce_loss, eo_logits, labels_a, labels_b, lam)
                loss_sar = mixup_criterion(label_smoothed_ce_loss, sar_logits, labels_a, labels_b, lam)
                
                # 2. Knowledge Distillation (matching SAR to EO)
                loss_kd = kd_loss_fn(sar_proj, eo_proj.detach())
                
                # 3. Contrastive Loss (using the contrastive head)
                all_proj = torch.cat([eo_proj, sar_proj], dim=0)
                all_labels = torch.cat([labels, labels], dim=0) 
                loss_con = con_loss_fn(model.get_contrastive_features(all_proj), all_labels)
                
                # Total Loss
                total_loss = (loss_eo + loss_sar) + CONFIG['kd_weight'] * loss_kd + CONFIG['con_weight'] * loss_con
            
            # Backprop with Scaler
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})
            
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device, train_dataset.class_to_idx)
        print(f"Val Acc: {val_metrics['acc']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Score: {val_metrics['final_score']:.4f}")
        
        # Save best
        if val_metrics['final_score'] > best_score:
            best_score = val_metrics['final_score']
            torch.save(model.state_dict(), 'best_mavic_v2.pth')
            print("New best model saved!")

if __name__ == "__main__":
    train()
