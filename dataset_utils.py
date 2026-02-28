import zipfile
import os
import shutil
import subprocess
import sys
from src.logger import logger

# Provided Google Drive File IDs (from user request)
DATASET_ZIP_ID = "1loB3fs_iPukKH6RdRTmeIHwRI51pijAu"
VALIDATION_CSV_ID = "1DX-Y3HSCIb-LHASPSx5F0kg-n6FcmzcJ"

# Store large files on D drive to avoid filling up C drive
DATA_DIR = "D:\\pbvs_data"
DATASET_ZIP = os.path.join(DATA_DIR, "pbvs_dataset.zip")
DATASET_DIR = os.path.join(DATA_DIR, "pbvs_mavic_dataset")
VALIDATION_CSV = os.path.join(DATA_DIR, "Validation_reference.csv")

os.makedirs(DATA_DIR, exist_ok=True)

def install_gdown():
    """Ensure gdown is installed for automated downloads."""
    try:
        import gdown
    except ImportError:
        logger.info("📦 Installing gdown for automated data retrieval...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    return gdown

def download_from_drive():
    """Download dataset and metadata from Google Drive automatically."""
    gdown_lib = install_gdown()
    
    # Download Zip if not exists
    if not os.path.exists(DATASET_ZIP):
        logger.info(f"📡 Downloading pbvs_dataset.zip to {DATA_DIR} ...")
        url = f'https://drive.google.com/uc?id={DATASET_ZIP_ID}'
        gdown_lib.download(url, DATASET_ZIP, quiet=False)
    
    # Download Reference CSV if not exists
    target_csv = os.path.join(DATASET_DIR, 'Validation_reference.csv')
    if not os.path.exists(target_csv) and not os.path.exists(VALIDATION_CSV):
         logger.info(f"📡 Downloading Validation_reference.csv to {DATA_DIR} ...")
         url_csv = f'https://drive.google.com/uc?id={VALIDATION_CSV_ID}'
         gdown_lib.download(url_csv, VALIDATION_CSV, quiet=False)

def extract_dataset(zip_path=DATASET_ZIP, extract_to=DATASET_DIR):
    """Extract the dataset zip and organize it."""
    if not os.path.exists(zip_path):
        logger.warning(f"⚠️ Warning: Zip file not found at {zip_path}. Attempting download...")
        download_from_drive()
    
    if not os.path.exists(extract_to):
        logger.info(f"🛠 Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("✅ Extraction complete.")
    else:
        logger.info(f"♻️  Dataset already extracted at {extract_to}. Skipping extraction.")
    
    # If the CSV was downloaded to DATA_DIR, move it into the extracted folder
    target_csv = os.path.join(extract_to, 'Validation_reference.csv')
    if os.path.exists(VALIDATION_CSV) and not os.path.exists(target_csv):
        logger.info(f"📦 Moving Validation_reference.csv into {extract_to}")
        shutil.move(VALIDATION_CSV, target_csv)

def check_structure(base_path=DATASET_DIR):
    """Check if the dataset structure matches expectations."""
    required = [
        os.path.join(base_path, 'train', 'EO_Train'),
        os.path.join(base_path, 'test'),
        os.path.join(base_path, 'Validation_reference.csv')
    ]
    optional = [
        os.path.join(base_path, 'train', 'SAR_Train'),  # Not always included
    ]

    missing = [p for p in required if not os.path.exists(p)]

    if missing:
        logger.error("❌ Missing required components:")
        for m in missing:
            logger.error(f" - {m}")
        return False

    for p in optional:
        if not os.path.exists(p):
            logger.warning(f"⚠️ Optional path not found (EO-only training mode): {p}")

    logger.info("✅ Dataset structure verified. Ready for training.")
    return True
if __name__ == "__main__":
    download_from_drive()
    extract_dataset('pbvs_dataset.zip')
    check_structure('./pbvs_mavic_dataset')
