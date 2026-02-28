import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models import MAVIC_V2_Model
from data import get_transforms, EOSARDataset, SAROnlyDataset, mixup_data, mixup_criterion
from losses import label_smoothed_ce_loss, FeatureMatchingLoss, SupConLoss
from evaluate import evaluate_model
from logger import logger
import multiprocessing

# Configuration - Tuned for Multi-GPU Superfast Training
CONFIG = {
    'train_sar_root': './pbvs_mavic_dataset/train/SAR_Train',
    'train_eo_root':  './pbvs_mavic_dataset/train/EO_Train',
    'val_sar_root':   './pbvs_mavic_dataset/val',
    'val_csv_path':   './pbvs_mavic_dataset/Validation_reference.csv',
    'num_classes': 10,
    'batch_size': 128,  # Increased for multi-GPU
    'epochs': 50,
    'lr': 4e-4,         # Scaled LR for larger effective batch size
    'wd': 1e-4,
    'alpha_mixup': 1.0,
    'kd_weight': 0.5,   # Increased KD for better SAR representation
    'con_weight': 0.2,  # Tightened clusters for OOD
    'num_workers': 8,   # Optimized for multi-CPU/GPU
}

def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(gpu)
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
        pin_memory=True
    )
    
    # Validation only on rank 0 to save resources
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=min(4, sys_cpus))

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

    best_score = 0.0
    patience = 7  # Epochs wait before early stopping
    epochs_no_improve = 0
    target_rank_1_score = 0.42 # Rank 1 is 0.40, we aim slightly higher before auto-stopping
    
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
