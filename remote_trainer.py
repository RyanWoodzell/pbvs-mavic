import modal
import os

# Define the remote environment
# This matches the local stack but runs on a Linux GPU container
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pillow",
        "scikit-learn",
        "tqdm",
    )
)

stub = modal.Stub("pbvs-mavic")

# Mount your local code so Modal can see it
code_mount = modal.Mount.from_local_dir(
    os.path.dirname(__file__),
    remote_path="/root/project",
)

@stub.function(
    image=image,
    gpu="A100", # Use premium GPU for fast parameter tuning
    timeout=3600 * 4, # 4 hour limit
    mounts=[code_mount],
)
def train_remote():
    """The entry point that runs inside the remote GPU container."""
    print("🚀 Container started. GPU detected. Initializing training...")
    import sys
    sys.path.append("/root/project/src")
    
    # Trigger the actual train function
    from train import train
    train()

@stub.local_entrypoint()
def main():
    train_remote.remote()
