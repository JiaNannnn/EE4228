import os
import sys
import urllib.request
import bz2
import shutil
import tempfile

def download_dlib_model():
    """Download the dlib face landmark predictor model if not already present"""
    
    print("Checking for dlib face landmark predictor file...")
    model_path = "shape_predictor_68_face_landmarks.dat"
    
    # Check if file already exists
    if os.path.exists(model_path):
        print(f"Model file already exists at {model_path}")
        return True
    
    # URL for the compressed model file
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    
    print(f"Downloading model from {url}...")
    
    try:
        # Create a temporary directory for downloading
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, compressed_file)
            
            # Download the compressed file
            urllib.request.urlretrieve(url, temp_file)
            
            print("Download complete. Decompressing...")
            
            # Decompress the file
            with open(model_path, 'wb') as outfile:
                with bz2.BZ2File(temp_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
            
            print(f"Model file saved to {model_path}")
            return True
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    try:
        import dlib
        print("dlib is installed, downloading model...")
        download_dlib_model()
    except ImportError:
        print("dlib is not installed. Please install it with: pip install dlib")
        print("Note: dlib might require additional dependencies (C++ compiler, etc.)")
        sys.exit(1) 