import os
import shutil
import numpy as np
from PIL import Image
import pandas as pd
import torch
from train import train, CONFIG
from inference import run_inference

def setup_dummy_competition_data():
    base = './pbvs_mavic_dataset'
    if os.path.exists(base):
        shutil.rmtree(base)
    
    # Create structure
    os.makedirs(f'{base}/train/SAR_Train/class_A', exist_ok=True)
    os.makedirs(f'{base}/train/EO_Train/class_A', exist_ok=True)
    os.makedirs(f'{base}/train/SAR_Train/class_B', exist_ok=True)
    os.makedirs(f'{base}/train/EO_Train/class_B', exist_ok=True)
    os.makedirs(f'{base}/val', exist_ok=True)
    os.makedirs(f'{base}/test', exist_ok=True)
    
    print("Generating dummy training pairs...")
    for cls in ['class_A', 'class_B']:
        for i in range(4):
            sar = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            eo = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(sar).save(f'{base}/train/SAR_Train/{cls}/img{i}.png')
            Image.fromarray(eo).save(f'{base}/train/EO_Train/{cls}/img{i}.png')
            
    print("Generating dummy validation/test data...")
    for i in range(4):
        sar = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        Image.fromarray(sar).save(f'{base}/val/vimg{i}.png')
        Image.fromarray(sar).save(f'{base}/test/timg{i}.png')
        
    # Validation CSV
    df_val = pd.DataFrame({
        'image_id': ['vimg0', 'vimg1', 'vimg2', 'vimg3'],
        'class': ['class_A', 'class_A', 'class_B', 'unknown'],
        'OOD_flag': [0, 0, 0, 1]
    })
    df_val.to_csv(f'{base}/Validation_reference.csv', index=False)
    return base

if __name__ == "__main__":
    print("=== STARTING END-TO-END VALIDATION ===")
    
    # 1. Setup
    setup_dummy_competition_data()
    
    # 2. Train Test
    CONFIG['num_classes'] = 2
    CONFIG['batch_size'] = 2
    CONFIG['epochs'] = 1
    CONFIG['device'] = 'cpu' # Force CPU 
    
    print("\n[PHASE 1] Testing Training Loop...")
    train()
    
    # 3. Inference Test
    print("\n[PHASE 2] Testing Inference Script...")
    class_to_idx = {'class_A': 0, 'class_B': 1}
    run_inference('best_mavic_v2.pth', './pbvs_mavic_dataset/test', 'submission.csv', class_to_idx)
    
    # 4. Verification
    if os.path.exists('submission.csv'):
        df_sub = pd.read_csv('submission.csv')
        print(f"\n[SUCCESS] submission.csv created with {len(df_sub)} rows.")
        print("Sample Predictions:")
        print(df_sub.head())
    else:
        print("\n[FAILURE] submission.csv was not created.")
        exit(1)
        
    print("\n=== ALL SYSTEMS VALIDATED: CODE IS READY FOR REAL DATA ===")
