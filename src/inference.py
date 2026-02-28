import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import MAVIC_V2_Model
from data import get_transforms, SAROnlyDataset
from evaluate import compute_ood_scores, normalize_logits

def run_inference(model_path, sar_root, output_csv, class_to_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = MAVIC_V2_Model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Data
    _, _, sar_val_transform = get_transforms()
    dataset = SAROnlyDataset(sar_root, transform=sar_val_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    results = []
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            sar = batch['sar'].to(device)
            names = batch['image_id']
            
            _, logits = model(None, sar, mode='sar_only')
            
            # Apply logit normalization as seen in Rank 1 approach
            norm_logits = normalize_logits(logits)
            
            # Prediction
            preds = torch.argmax(norm_logits, dim=1).cpu().numpy()
            
            # OOD Score (Max Logit on normalized logits)
            scores = compute_ood_scores(norm_logits, method='max_logit').cpu().numpy()
            
            for name, pred, score in zip(names, preds, scores):
                # Competition usually wants image_id (without extension), class_id, and score
                image_id = os.path.splitext(name)[0]
                results.append({
                    'image_id': image_id,
                    'class_id': pred,
                    'score': score
                })
                
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # Settings
    MODEL_PATH = 'best_mavic_v2.pth'
    SAR_ROOT = './pbvs_mavic_dataset/test'
    OUTPUT_CSV = 'submission.csv'
    
    # We need class_to_idx. In a real scenario, we'd save this during training.
    # For now, let's assume standard competition classes or retrieve from train folder.
    # classes = ['bus', 'pickup_truck', 'sedan', 'truck', 'van', ...]
    # Here we use a dummy map or try to detect from the train data if available.
    
    # Assuming the user has the train folder available to recreate the mapping
    train_root = './pbvs_mavic_dataset/train/SAR_Train'
    if os.path.exists(train_root):
        classes = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        if os.path.exists(MODEL_PATH):
            run_inference(MODEL_PATH, SAR_ROOT, OUTPUT_CSV, class_to_idx)
        else:
            print(f"Model checkpoint not found at {MODEL_PATH}. Please train first.")
    else:
        print("Training directory not found. Cannot determine class mapping.")
