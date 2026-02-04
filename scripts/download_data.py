"""
Script to download NASA C-MAPSS Turbofan Engine Degradation Dataset.
Dataset source: NASA Prognostics Data Repository
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR


# C-MAPSS dataset URLs (multiple mirrors for reliability)
# Primary: GitHub mirror of the dataset
DOWNLOAD_URLS = [
    # Direct file links from popular GitHub repositories with C-MAPSS data
    "https://github.com/hankroark/Turbofan-Engine-Degradation/raw/master/CMAPSSData.zip",
    "https://github.com/biswajitsahoo1111/rul_codes_open/raw/master/CMAPSSData/CMAPSSData.zip",
]


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress indicator."""
    try:
        print(f"{desc} from {url}...")
        
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r{desc}: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n{desc} complete: {dest_path}")
        return True
    except Exception as e:
        print(f"\nError downloading from {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file."""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print(f"Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return False


def verify_dataset(data_dir: Path) -> bool:
    """Verify that all required files are present."""
    required_files = [
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
        "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
        "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
        "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    ]
    
    all_present = True
    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"Missing: {fname}")
            all_present = False
        else:
            print(f"Found: {fname}")
    
    return all_present


def main():
    """Download and extract C-MAPSS dataset."""
    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if verify_dataset(RAW_DATA_DIR):
        print("\nDataset already downloaded and verified!")
        return True
    
    # Download location
    zip_path = RAW_DATA_DIR / "CMAPSSData.zip"
    
    # Try each URL until one works
    downloaded = False
    for url in DOWNLOAD_URLS:
        if download_file(url, zip_path, "Downloading C-MAPSS dataset"):
            downloaded = True
            break
    
    if not downloaded:
        print("\nFailed to download from all sources.")
        print("Please manually download the C-MAPSS dataset from:")
        print("https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
        print(f"And extract it to: {RAW_DATA_DIR}")
        return False
    
    # Extract
    if not extract_zip(zip_path, RAW_DATA_DIR):
        return False
    
    # Some zip files have data in a subdirectory - check and move if needed
    subdirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    for subdir in subdirs:
        # Move files from subdirectory to raw data dir
        for f in subdir.iterdir():
            dest = RAW_DATA_DIR / f.name
            if not dest.exists():
                f.rename(dest)
    
    # Verify
    if verify_dataset(RAW_DATA_DIR):
        print("\nDataset downloaded and verified successfully!")
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
        return True
    else:
        print("\nDataset verification failed. Some files may be missing.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
