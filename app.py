import sys

# Must be first: force UTF-8 on stdout/stderr before any logging is set up.
# This prevents UnicodeEncodeError on Windows (cp1252) when printing emoji/unicode.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os

# Redirect PyTorch model cache to D drive — C drive has very limited free space.
os.environ.setdefault('TORCH_HOME', r'D:\torch_cache')
os.environ.setdefault('HF_HOME', r'D:\torch_cache\huggingface')
import torch

# Add src/ to sys.path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from logger import logger

# Remote Execution Config (Modal for Serverless GPU)
try:
    import modal
    STAG_REMOTE_ENABLED = True
except ImportError:
    STAG_REMOTE_ENABLED = False

def check_local_gpu():
    """Returns True if a valid training GPU is found."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 [LOCAL] GPU found: {device_name}")
        return True
    logger.warning("⚠️ [LOCAL] No GPU detected. Local training would be too slow for Rank 1.")
    return False

def run_local_training():
    """Runs the training pipeline on the local machine with Multi-GPU support."""
    logger.info("📦 Initializing high-performance training pipeline...")
    
    # Automated Data Check and Retrieval
    from dataset_utils import download_from_drive, extract_dataset, check_structure
    logger.info("🔍 Checking dataset availability...")
    download_from_drive()
    extract_dataset()
    if not check_structure():
        logger.error("❌ Error: Initialized data structure is invalid. Aborting.")
        return

    # Check GPU count for multi-GPU training
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logger.info(f"🔥 [LOCAL] {gpu_count} GPUs detected! Launching Distributed Training...")
        # Pre-download all model weights in the main process before spawning children.
        # mp.spawn creates fresh processes that all try to download simultaneously,
        # causing a race condition where one reads a partially-written file.
        logger.info("⬇️  Pre-fetching pretrained weights to cache before spawning workers...")
        import torchvision.models as _tv
        _tv.resnet50(weights='IMAGENET1K_V1')
        _tv.efficientnet_b0(weights='IMAGENET1K_V1')
        del _tv
        logger.info("✅ Weights cached. Spawning DDP workers...")

        # Use torch.multiprocessing.spawn — works on Windows (torchrun uses libuv TCPStore
        # which is not compiled into Windows PyTorch wheels).
        # Use a FileStore (filesystem-based rendezvous) to avoid TCP/libuv entirely.
        import tempfile, time, torch.multiprocessing as mp
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from train import train_spawn
        store_path = os.path.join(tempfile.gettempdir(),
                                  f'torch_ddp_store_{os.getpid()}_{int(time.time())}')
        mp.spawn(train_spawn, args=(gpu_count, store_path), nprocs=gpu_count, join=True)
    else:
        logger.info("🚀 [LOCAL] Single GPU detected. Running in standard mode...")
        cmd = f"{sys.executable} src/train.py"
        os.system(cmd)

def run_remote_training():
    """Triggers remote execution on a high-end GPU cluster via Modal."""
    if not STAG_REMOTE_ENABLED:
        logger.error("❌ Remote tools not installed. Please run: pip install modal")
        return

    logger.info("📡 [REMOTE] Dispatching job to StagAI Cloud Cluster...")
    logger.info("🔗 Syncing code and data. Training will start on a remote A100/L4 GPU...")
    
    # This calls the remote entry point in our project structure
    # The remote runner handles setup, training, and streaming logs back
    os.system("modal run remote_trainer.py")

if __name__ == "__main__":
    logger.info("=== StagAI MAVIC V2 Control Center ===")
    
    if check_local_gpu():
        run_local_training()
    else:
        # If no GPU locally, we automatically try remote
        if STAG_REMOTE_ENABLED:
            run_remote_training()
        else:
            logger.info("💡 Suggestion: To run training on a remote GPU with zero intervention,")
            logger.info("   sign up for Modal (modal.com) and run 'modal setup'.")
            logger.info("   Then, this script will automatically handle the rest.")
