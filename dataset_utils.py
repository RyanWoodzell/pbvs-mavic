import zipfile
import os
import shutil

def extract_dataset(zip_path, extract_to='./pbvs_mavic_dataset'):
    """Extract the dataset zip and organize it."""
    if not os.path.exists(zip_path):
        print(f"Zip file not found at {zip_path}")
        return
    
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def check_structure(base_path):
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
        print("Missing components:")
        for m in missing:
            print(f" - {m}")
    else:
        print("Dataset structure looks good!")

if __name__ == "__main__":
    # Assuming the user downloads pbvs_dataset.zip to the current directory
    extract_dataset('pbvs_dataset.zip')
    check_structure('./pbvs_mavic_dataset')
