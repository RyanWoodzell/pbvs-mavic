import torch
import os
import shutil
import numpy as np
from PIL import Image
import pandas as pd
from train import train, CONFIG

def create_mini_dataset():
    base = './pbvs_mavic_dataset'
    if os.path.exists(base):
        shutil.rmtree(base)
    
    os.makedirs(f'{base}/train/SAR_Train/class0', exist_ok=True)
    os.makedirs(f'{base}/train/EO_Train/class0', exist_ok=True)
    os.makedirs(f'{base}/val', exist_ok=True)
    
    # Create 4 training pairs
    for i in range(4):
        sar = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        eo = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(sar).save(f'{base}/train/SAR_Train/class0/img{i}.png')
        Image.fromarray(eo).save(f'{base}/train/EO_Train/class0/img{i}.png')
            
    # Create 2 val images
    for i in range(2):
        sar = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        Image.fromarray(sar).save(f'{base}/val/vimg{i}.png')
        
    # Create csv
    df = pd.DataFrame({
        'image_id': ['vimg0', 'vimg1'],
        'class': ['class0', 'class0'],
        'OOD_flag': [0, 0]
    })
    df.to_csv(f'{base}/Validation_reference.csv', index=False)
    return base

if __name__ == "__main__":
    create_mini_dataset()
    
    # Override config for quick test
    CONFIG['num_classes'] = 1
    CONFIG['batch_size'] = 2
    CONFIG['epochs'] = 1
    CONFIG['device'] = 'cpu'
    
    print("Starting 1-epoch test train...")
    try:
        train()
        print("\nTraining/Validation loop completed successfully!")
    except Exception as e:
        print(f"\nTraining Failed: {e}")
        exit(1)
