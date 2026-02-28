import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models import MAVIC_V2_Model
from data import get_transforms, EOSARDataset, SAROnlyDataset, mixup_data, mixup_criterion, SpeckleFilter, SARLogTransform
import torchvision.transforms.v2 as v2
from losses import label_smoothed_ce_loss, FeatureMatchingLoss, SupConLoss
from evaluate import evaluate_model
from logger import logger
import multiprocessing
import json

# Dynamic Configuration Loading
# Loads hyperparams and Rank-1 target metrics from the root config.json
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
try:
    with open(CONFIG_PATH, 'r') as f:
        _cfg = json.load(f)
    CONFIG = {**_cfg['dataset'], **_cfg['hyperparameters']}
    EARLY_STOPPING_CONFIG = _cfg['early_stopping']
except Exception as e:
    raise RuntimeError(f"Could not load config.json from {CONFIG_PATH}: {str(e)}")


def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(gpu)
        
        # Optimize memory allocator for deep learning
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress generic warnings
        # Enable CuDNN auto-tuner to find best convolution algorithms
        torch.backends.cudnn.benchmark = True
    else:
        rank = 0
        world_size = 1
        gpu = 0
    return rank, world_size, gpu

def train():
    rank, world_size, gpu = setup_ddp()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        logger.info(f"🚀 [STAG-DDP] Initializing training on {world_size} GPUs...")
        if not os.path.exists(CONFIG['train_sar_root']):
            logger.error(f"ERROR: Dataset not found. Run dataset_utils.py first.")
            return

    # Data Pipeline - Multi-threaded and Distributed
    eo_tr, sar_tr, sar_val = get_transforms()
    
    if rank == 0:
        logger.info("⚙️  Configuring On-The-Fly Data Preprocessing Pipeline:")
        logger.info("  - SAR Training : Space -> SpeckleFilter -> LogTransform -> Normalize -> MixUp")
        logger.info("  - SAR Validation: Space -> SpeckleFilter -> LogTransform -> Normalize")
        logger.info("  - EO  Training : Space -> Resize -> ColorJitter -> Normalize -> MixUp")
        
    train_dataset = EOSARDataset(CONFIG['train_sar_root'], CONFIG['train_eo_root'], sar_tr, eo_tr)
    val_dataset = SAROnlyDataset(CONFIG['val_sar_root'], CONFIG['val_csv_path'], train_dataset.class_to_idx, sar_val)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    
    # Critical Fix: CPU Bottlenecking
    # The system has 4 vCPUs. If world_size=4 and num_workers=8, we spawn 32 workers,
    # causing 100% CPU thrashing and starving the GPUs.
    # We must divide the available system CPUs evenly across the GPU processes.
    sys_cpus = multiprocessing.cpu_count()
    workers_per_gpu = max(1, sys_cpus // world_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'] // world_size,
        sampler=train_sampler,
        num_workers=workers_per_gpu,
        pin_memory=True,
        drop_last=True, # Better for Batch Norm in DDP
        prefetch_factor=2 if workers_per_gpu > 0 else None,
        persistent_workers=True if workers_per_gpu > 0 else False
    )
    
    # Validation only on rank 0 to save resources
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=min(2, sys_cpus),
            pin_memory=True
        )

    # Model - Wrapped for DDP
    model = MAVIC_V2_Model(num_classes=CONFIG['num_classes']).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # Modern AMP API (as per user warning)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    kd_loss_fn = FeatureMatchingLoss(loss_type='mse')
    con_loss_fn = SupConLoss(temperature=0.07)

    # Move Augmentations to GPU to fix 100% CPU Bottleneck
    # Using Torchvision V2 which is hyper-optimized for Batched GPU Tensors
    gpu_transforms_eo = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to(device)

    # Re-use custom classes but pass them GPU tensors
    speckle_filter = SpeckleFilter()
    log_transform = SARLogTransform()
    
    gpu_transforms_sar_train = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.Normalize(mean=[0.5], std=[0.2])
    ]).to(device)
    
    gpu_transforms_sar_val = v2.Normalize(mean=[0.5], std=[0.2]).to(device)

    best_score = 0.0
    patience = EARLY_STOPPING_CONFIG.get('patience', 7)
    epochs_no_improve = 0
    target_rank_1_score = EARLY_STOPPING_CONFIG.get('target_rank_1_score', 0.42)
    
    for epoch in range(CONFIG['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        pbar = None
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in (pbar if pbar else train_loader):
            sar = batch['sar'].to(device, non_blocking=True)
            eo = batch['eo'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # --- GPU-Accelerated Preprocessing ---
            # EO Pipeline
            eo = gpu_transforms_eo(eo)
            # SAR Pipeline
            sar = speckle_filter(sar)
            sar = log_transform(sar)
            sar = gpu_transforms_sar_train(sar)
            
            sar, eo, labels_a, labels_b, lam = mixup_data(sar, eo, labels, CONFIG['alpha_mixup'])
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # Forward - Handle DDP wrapper if needed
                if world_size > 1:
                    eo_proj, sar_proj, eo_logits, sar_logits = model.module(eo, sar, mode='joint')
                else:
                    eo_proj, sar_proj, eo_logits, sar_logits = model(eo, sar, mode='joint')
                
                loss_eo = mixup_criterion(label_smoothed_ce_loss, eo_logits, labels_a, labels_b, lam)
                loss_sar = mixup_criterion(label_smoothed_ce_loss, sar_logits, labels_a, labels_b, lam)
                loss_kd = kd_loss_fn(sar_proj, eo_proj.detach())
                
                # Use raw model for contrastive head to avoid DDP sync issues on sub-modules
                raw_model = model.module if world_size > 1 else model
                all_proj = torch.cat([eo_proj, sar_proj], dim=0)
                all_labels = torch.cat([labels, labels], dim=0) 
                loss_con = con_loss_fn(raw_model.get_contrastive_features(all_proj), all_labels)
                
                total_loss = (loss_eo + loss_sar) + CONFIG['kd_weight'] * loss_kd + CONFIG['con_weight'] * loss_con
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            if pbar:
                pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})
            
        scheduler.step()
        
        # Validation on Master Node and Broadcast Early Stop Signals
        should_stop = torch.tensor(0).to(device)
        
        if rank == 0:
            val_metrics = evaluate_model(model.module if world_size > 1 else model, val_loader, device, train_dataset.class_to_idx)
            final_sc = val_metrics['final_score']
            logger.info(f"📈 [VAL] Acc: {val_metrics['acc']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Score: {final_sc:.4f}")
            
            if final_sc > best_score:
                best_score = final_sc
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'best_mavic_v2.pth')
                logger.info("💎 New best model saved!")
                
                if final_sc >= target_rank_1_score:
                    logger.info(f"🎉🏆 TARGET REACHED! Score {final_sc:.4f} > Rank 1. Auto-stopping to save compute.")
                    should_stop = torch.tensor(1).to(device)
            else:
                epochs_no_improve += 1
                logger.info(f"⏳ No improvement for {epochs_no_improve} epochs.")
                if epochs_no_improve >= patience:
                    logger.warning(f"🛑 EARLY STOPPING triggered. No improvement for {patience} epochs.")
                    should_stop = torch.tensor(1).to(device)
                    
        # Sync the stopping decision across all GPUs
        if world_size > 1:
            dist.broadcast(should_stop, src=0)
            
        if should_stop.item() == 1:
            break

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
