import torch
from torch.utils.data import DataLoader
from models import MAVIC_V2_Model
from data import get_transforms, EOSARDataset, SAROnlyDataset
import os
import shutil
import numpy as np
from PIL import Image

def create_mini_dataset():
    base = './test_data'
    if os.path.exists(base):
        shutil.rmtree(base)
    
    os.makedirs(f'{base}/train/sar/class0', exist_ok=True)
    os.makedirs(f'{base}/train/eo/class0', exist_ok=True)
    os.makedirs(f'{base}/train/sar/class1', exist_ok=True)
    os.makedirs(f'{base}/train/eo/class1', exist_ok=True)
    os.makedirs(f'{base}/val', exist_ok=True)
    
    # Create 4 pairs
    for cls in ['class0', 'class1']:
        for i in range(2):
            sar = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            eo = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(sar).save(f'{base}/train/sar/{cls}/img{i}.png')
            Image.fromarray(eo).save(f'{base}/train/eo/{cls}/img{i}.png')
            
    # Create 2 val images
    for i in range(2):
        sar = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        Image.fromarray(sar).save(f'{base}/val/vimg{i}.png')
        
    # Create csv
    import pandas as pd
    df = pd.DataFrame({
        'image_id': ['vimg0', 'vimg1'],
        'class': ['class0', 'class1'],
        'OOD_flag': [0, 1]
    })
    df.to_csv(f'{base}/val_meta.csv', index=False)
    return base

def test_pipeline():
    print("Creating mini dataset...")
    base = create_mini_dataset()
    
    device = torch.device('cpu')
    print(f"Testing on {device}")
    
    eo_tr, sar_tr, sar_val = get_transforms()
    train_ds = EOSARDataset(f'{base}/train/sar', f'{base}/train/eo', sar_tr, eo_tr)
    val_ds = SAROnlyDataset(f'{base}/val', f'{base}/val_meta.csv', train_ds.class_to_idx, sar_val)
    
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    
    print("Initializing Model...")
    model = MAVIC_V2_Model(num_classes=2).to(device)
    
    print("Testing Forward Pass (Joint)...")
    batch = next(iter(train_loader))
    eo, sar, labels = batch['eo'].to(device), batch['sar'].to(device), batch['label'].to(device)
    eo_proj, sar_proj, eo_logits, sar_logits = model(eo, sar, mode='joint')
    
    print(f"EO Proj shape: {eo_proj.shape}") # Expected [2, 256]
    print(f"SAR Proj shape: {sar_proj.shape}")
    print(f"EO Logits shape: {eo_logits.shape}") # Expected [2, 2]
    
    assert eo_proj.shape == (2, 256)
    assert eo_logits.shape == (2, 2)
    
    print("Testing Forward Pass (Inference/SAR-only)...")
    v_batch = next(iter(val_loader))
    v_sar = v_batch['sar'].to(device)
    v_proj, v_logits = model(None, v_sar, mode='sar_only')
    print(f"Val Logits shape: {v_logits.shape}")
    assert v_logits.shape == (2, 2)
    
    print("Testing Contrastive Head...")
    con_feat = model.get_contrastive_features(eo_proj)
    print(f"Contrastive Feature shape: {con_feat.shape}") # Expected [2, 128]
    assert con_feat.shape == (2, 128)
    
    print("\n✅ End-to-end architectural test passed successfully!")

if __name__ == "__main__":
    test_pipeline()
