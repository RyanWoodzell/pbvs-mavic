import os
import torch
import sys

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

    # Check GPU count for torchrun
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logger.info(f"🔥 [LOCAL] {gpu_count} GPUs detected! Launching Distributed Training...")
        # CRITICAL: Prevent OpenMP thread thrashing on CPU-starved multi-GPU instances
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        
        # Use torchrun to spawn multiple processes (one per GPU)
        cmd = f"{sys.executable} -m torch.distributed.run --nproc_per_node={gpu_count} src/train.py"
        import subprocess
        subprocess.run(cmd.split(), env=env)
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
