"""
Download Models Script

Downloads required models for the face recognition system:
1. dlib's shape_predictor_68_face_landmarks.dat for facial landmark detection
2. (Optional) Pre-trained face recognition models
"""

import os
import sys
import requests
import bz2
import shutil
import hashlib
import argparse
from tqdm import tqdm
from pathlib import Path

# URLs for models
SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
SHAPE_PREDICTOR_HASH = "677a91476056de0507f1915adc7ef86a"  # MD5 hash for verification

def download_file(url, output_path, desc=None):
    """
    Download a file with progress bar
    
    Parameters:
    -----------
    url : str
        URL to download
    output_path : str
        Path to save the downloaded file
    desc : str, optional
        Description for the progress bar
        
    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8KB blocks
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
                
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_bz2(input_path, output_path):
    """
    Extract a bz2 file
    
    Parameters:
    -----------
    input_path : str
        Path to bz2 file
    output_path : str
        Path to extract to
        
    Returns:
    --------
    bool
        True if extraction successful, False otherwise
    """
    try:
        with bz2.BZ2File(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
            shutil.copyfileobj(input_file, output_file)
        return True
    except Exception as e:
        print(f"Error extracting {input_path}: {e}")
        return False

def verify_file(file_path, expected_hash):
    """
    Verify a file using MD5 hash
    
    Parameters:
    -----------
    file_path : str
        Path to file to verify
    expected_hash : str
        Expected MD5 hash
        
    Returns:
    --------
    bool
        True if hash matches, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        print(f"Error verifying {file_path}: {e}")
        return False

def download_shape_predictor(output_dir):
    """
    Download and extract dlib's shape predictor model
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the model
        
    Returns:
    --------
    bool
        True if download and extraction successful, False otherwise
    """
    output_path = os.path.join(output_dir, "shape_predictor_68_face_landmarks.dat")
    
    # Check if model already exists and is valid
    if os.path.exists(output_path) and verify_file(output_path, SHAPE_PREDICTOR_HASH):
        print(f"Shape predictor model already exists at {output_path}")
        return True
        
    # Download compressed model
    compressed_path = output_path + ".bz2"
    print(f"Downloading shape predictor model to {compressed_path}")
    
    if not download_file(SHAPE_PREDICTOR_URL, compressed_path, desc="Downloading shape predictor"):
        return False
        
    # Extract compressed model
    print(f"Extracting shape predictor model to {output_path}")
    if not extract_bz2(compressed_path, output_path):
        return False
        
    # Verify extracted model
    print("Verifying shape predictor model...")
    if not verify_file(output_path, SHAPE_PREDICTOR_HASH):
        print("Error: Shape predictor model verification failed")
        return False
        
    # Clean up compressed file
    os.remove(compressed_path)
    
    print(f"Shape predictor model downloaded and extracted to {output_path}")
    return True

def main():
    """Main function to download models"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download required models for face recognition")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Directory to save the models (default: current directory)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download shape predictor model
    if download_shape_predictor(args.output_dir):
        print("Shape predictor model download completed successfully")
    else:
        print("Error: Failed to download shape predictor model")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 