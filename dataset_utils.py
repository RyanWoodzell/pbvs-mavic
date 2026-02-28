import zipfile
import os
import shutil
import subprocess
import sys

# Provided Google Drive File IDs (from user request)
DATASET_ZIP_ID = "1loB3fs_iPukKH6RdRTmeIHwRI51pijAu"
VALIDATION_CSV_ID = "1DX-Y3HSCIb-LHASPSx5F0kg-n6FcmzcJ"

def install_gdown():
    """Ensure gdown is installed for automated downloads."""
    try:
        import gdown
    except ImportError:
        print("📦 Installing gdown for automated data retrieval...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    return gdown

def download_from_drive():
    """Download dataset and metadata from Google Drive automatically."""
    gdown_lib = install_gdown()
    
    # Download Zip if not exists
    if not os.path.exists('pbvs_dataset.zip'):
        print(f"📡 Downloading pbvs_dataset.zip from Google Drive...")
        url = f'https://drive.google.com/uc?id={DATASET_ZIP_ID}'
        gdown_lib.download(url, 'pbvs_dataset.zip', quiet=False)
    
    # Download Reference CSV if not exists (in root or mapped folder)
    if not os.path.exists('pbvs_mavic_dataset/Validation_reference.csv') and not os.path.exists('Validation_reference.csv'):
         print(f"📡 Downloading Validation_reference.csv from Google Drive...")
         url_csv = f'https://drive.google.com/uc?id={VALIDATION_CSV_ID}'
         gdown_lib.download(url_csv, 'Validation_reference.csv', quiet=False)

def extract_dataset(zip_path='pbvs_dataset.zip', extract_to='./pbvs_mavic_dataset'):
    """Extract the dataset zip and organize it."""
    if not os.path.exists(zip_path):
        print(f"⚠️ Warning: Zip file not found at {zip_path}. Attempting download...")
        download_from_drive()
    
    if not os.path.exists(extract_to):
        print(f"🛠 Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction complete.")
    else:
        print(f"♻️  Dataset already extracted at {extract_to}. Skipping extraction.")
    
    # If the CSV was downloaded to root, move it into the extracted folder
    target_csv = os.path.join(extract_to, 'Validation_reference.csv')
    if os.path.exists('Validation_reference.csv') and not os.path.exists(target_csv):
        print(f"📦 Moving Validation_reference.csv into {extract_to}")
        shutil.move('Validation_reference.csv', target_csv)

def check_structure(base_path='./pbvs_mavic_dataset'):
    """Check if the dataset structure matches expectations."""
    expected = [
        os.path.join(base_path, 'train/EO_Train'),
        os.path.join(base_path, 'train/SAR_Train'),
        os.path.join(base_path, 'val'),
        os.path.join(base_path, 'Validation_reference.csv')
    ]
    
    missing = []
    for path in expected:
        if not os.path.exists(path):
            missing.append(path)
            
    if missing:
        print("❌ Missing components:")
        for m in missing:
            print(f" - {m}")
        return False
    else:
        print("🚀 Dataset structure is PERFECT. Ready for training.")
        return True

if __name__ == "__main__":
    download_from_drive()
    extract_dataset('pbvs_dataset.zip')
    check_structure('./pbvs_mavic_dataset')
