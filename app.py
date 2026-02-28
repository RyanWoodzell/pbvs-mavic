import os
import torch
import sys

# Add src/ to sys.path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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
        print(f"🚀 [LOCAL] GPU found: {device_name}")
        return True
    print("⚠️ [LOCAL] No GPU detected. Local training would be too slow for Rank 1.")
    return False

def run_local_training():
    """Runs the training pipeline on the local machine."""
    print("📦 Initializing local training pipeline...")
    
    # Automated Data Check and Retrieval
    from dataset_utils import download_from_drive, extract_dataset, check_structure
    print("🔍 Checking dataset availability...")
    download_from_drive()
    extract_dataset()
    if not check_structure():
        print("❌ Error: Initialized data structure is invalid. Aborting.")
        return

    from train import train
    train()

def run_remote_training():
    """Triggers remote execution on a high-end GPU cluster via Modal."""
    if not STAG_REMOTE_ENABLED:
        print("❌ Remote tools not installed. Please run: pip install modal")
        return

    print("📡 [REMOTE] Dispatching job to StagAI Cloud Cluster...")
    print("🔗 Syncing code and data. Training will start on a remote A100/L4 GPU...")
    
    # This calls the remote entry point in our project structure
    # The remote runner handles setup, training, and streaming logs back
    os.system("modal run remote_trainer.py")

if __name__ == "__main__":
    print("=== StagAI MAVIC V2 Control Center ===")
    
    if check_local_gpu():
        run_local_training()
    else:
        # If no GPU locally, we automatically try remote
        if STAG_REMOTE_ENABLED:
            run_remote_training()
        else:
            print("💡 Suggestion: To run training on a remote GPU with zero intervention,")
            print("   sign up for Modal (modal.com) and run 'modal setup'.")
            print("   Then, this script will automatically handle the rest.")
