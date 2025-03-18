import os
import cv2
import numpy as np
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from face_preprocessor import FacePreprocessor
from pca_face_detector import PCAFaceDetector
import json

def download_file(url, destination):
    """Download a file from URL to destination with progress bar"""
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Download complete: {destination}")
    return destination

def extract_archive(archive_path, extract_to):
    """Extract a zip or tar.gz archive to the specified directory"""
    print(f"Extracting {archive_path} to {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"Extraction complete: {extract_to}")
    return extract_to

def download_att_database():
    """Download AT&T/ORL face database"""
    print("Downloading AT&T/ORL Database of Faces...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    att_dir = os.path.join(data_dir, "att_faces")
    if os.path.exists(att_dir):
        print(f"AT&T dataset already exists at {att_dir}")
        return att_dir
    
    # AT&T/ORL database URL
    att_url = "https://www.kaggle.com/datasets/kasikrit/att-database-of-faces/download?datasetVersionNumber=2"
    
    print("The AT&T dataset requires manual download from Kaggle.")
    print(f"Please download from: {att_url}")
    print("After downloading, place the file in the 'data' directory and name it 'att_faces.zip'")
    
    att_zip = os.path.join(data_dir, "att_faces.zip")
    if os.path.exists(att_zip):
        print(f"Found AT&T dataset at {att_zip}, extracting...")
        extract_archive(att_zip, data_dir)
        return att_dir
    else:
        print("Alternative URL for AT&T dataset (more direct):")
        print("https://cam-orl.co.uk/facedatabase.html")
        return None

def download_yale_database():
    """Download Yale face database"""
    print("Downloading Yale Face Database...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    yale_dir = os.path.join(data_dir, "yale_faces")
    if os.path.exists(yale_dir):
        print(f"Yale dataset already exists at {yale_dir}")
        return yale_dir
    
    # Yale B Extended database URL - smaller version for easier processing
    yale_url = "http://vision.ucsd.edu/~leekc/ExtYaleDatabase/Yale%20Face%20Database.zip"
    
    # Try to download the Yale database
    try:
        yale_zip = os.path.join(data_dir, "yale_faces.zip")
        download_file(yale_url, yale_zip)
        extract_archive(yale_zip, yale_dir)
        return yale_dir
    except Exception as e:
        print(f"Error downloading Yale dataset: {str(e)}")
        print("Alternative URL for Yale dataset:")
        print("http://vision.ucsd.edu/content/yale-face-database")
        return None

def load_att_faces(att_dir, target_size=(100, 100)):
    """Load AT&T/ORL face database"""
    if not os.path.exists(att_dir):
        print(f"AT&T dataset directory {att_dir} not found")
        return None, None
    
    print("Loading AT&T/ORL faces...")
    faces = []
    labels = []
    
    # The AT&T dataset structure: s1/1.pgm, s1/2.pgm, ... s40/10.pgm
    for person_id in range(1, 41):
        person_dir = os.path.join(att_dir, f"s{person_id}")
        
        if not os.path.exists(person_dir):
            print(f"Person directory {person_dir} not found, trying alternate structure...")
            # Try alternate structure
            person_dir = os.path.join(att_dir, "orl_faces", f"s{person_id}")
            if not os.path.exists(person_dir):
                print(f"Alternate person directory not found either")
                continue
        
        for image_id in range(1, 11):
            image_path = os.path.join(person_dir, f"{image_id}.pgm")
            
            if not os.path.exists(image_path):
                print(f"Image {image_path} not found")
                continue
            
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to read image {image_path}")
                    continue
                
                # Resize to target size
                img = cv2.resize(img, target_size)
                faces.append(img)
                labels.append(f"att_s{person_id}")
                
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
    
    print(f"Loaded {len(faces)} images from AT&T dataset")
    return np.array(faces), np.array(labels)

def load_yale_faces(yale_dir, target_size=(100, 100)):
    """Load Yale face database"""
    if not os.path.exists(yale_dir):
        print(f"Yale dataset directory {yale_dir} not found")
        return None, None
    
    print("Loading Yale faces...")
    faces = []
    labels = []
    
    # Try different possible structures for Yale dataset
    # First try standard Yale structure
    subject_dirs = [d for d in os.listdir(yale_dir) if os.path.isdir(os.path.join(yale_dir, d)) and 'subject' in d.lower()]
    
    if not subject_dirs:
        # Try another common structure
        for d in os.listdir(yale_dir):
            potential_dir = os.path.join(yale_dir, d)
            if os.path.isdir(potential_dir):
                subject_dirs = [os.path.join(d, sd) for sd in os.listdir(potential_dir) 
                               if os.path.isdir(os.path.join(potential_dir, sd)) and 'subject' in sd.lower()]
                if subject_dirs:
                    break
    
    if not subject_dirs:
        print("Could not find subject directories in Yale dataset")
        # Try a flat structure - look for any image files
        all_files = []
        for root, dirs, files in os.walk(yale_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.pgm', '.png', '.gif')):
                    all_files.append(os.path.join(root, f))
        
        for image_path in all_files:
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Extract subject from filename (typical Yale naming might have subject info)
                fname = os.path.basename(image_path)
                parts = fname.split('.')
                if len(parts) > 1:
                    # Try to extract subject
                    subject = f"yale_{parts[0]}"
                else:
                    subject = f"yale_unknown"
                
                # Resize to target size
                img = cv2.resize(img, target_size)
                faces.append(img)
                labels.append(subject)
                
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
    else:
        # Process structured directories
        for subject_dir in subject_dirs:
            full_subject_dir = os.path.join(yale_dir, subject_dir)
            if not os.path.isdir(full_subject_dir):
                full_subject_dir = subject_dir  # In case we already have the full path
                subject_dir = os.path.basename(subject_dir)
            
            # Extract subject ID
            subject_id = subject_dir.lower().replace('subject', '').strip()
            if not subject_id:
                subject_id = os.path.basename(full_subject_dir)
            
            all_files = [f for f in os.listdir(full_subject_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.pgm', '.png', '.gif'))]
            
            for image_file in all_files:
                image_path = os.path.join(full_subject_dir, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Failed to read image {image_path}")
                        continue
                    
                    # Resize to target size
                    img = cv2.resize(img, target_size)
                    faces.append(img)
                    labels.append(f"yale_{subject_id}")
                    
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
    
    print(f"Loaded {len(faces)} images from Yale dataset")
    return np.array(faces), np.array(labels)

def train_pca_model(faces, labels, n_components=0.95, target_size=(100, 100)):
    """Train a PCA model on the face data"""
    print(f"Training PCA model with {len(faces)} images...")
    
    # Flatten the images
    n_samples, height, width = faces.shape
    X = faces.reshape(n_samples, height * width)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train PCA
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Number of components: {pca.n_components_}")
    print(f"Explained variance by first component: {explained_variance[0]:.4f}")
    print(f"Cumulative explained variance (10 components): {cumulative_variance[min(9, len(cumulative_variance)-1)]:.4f}")
    
    # Transform data to PCA space
    X_pca = pca.transform(X_scaled)
    
    # Create a mean face
    mean_face = scaler.mean_.reshape(height, width)
    
    # Create eigenfaces visualization
    n_eigenfaces = min(16, pca.components_.shape[0])
    eigenfaces = pca.components_[:n_eigenfaces]
    
    # Compute reconstruction error
    X_rec = pca.inverse_transform(X_pca)
    reconstruction_errors = np.mean(np.square(X_scaled - X_rec), axis=1)
    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)
    threshold = mean_error + 2 * std_error
    
    # Prepare model for saving
    model_data = {
        'pca': pca,
        'scaler': scaler,
        'threshold': threshold,
        'mean_face': mean_face,
        'target_size': target_size,
        'labels': labels
    }
    
    # Create a PCA face detector and train it
    pca_detector = PCAFaceDetector(n_components=n_components, target_size=target_size)
    
    # We need non-face samples for the detector
    # Use random noise as non-face samples (for demonstration)
    non_faces = np.random.randint(0, 255, (n_samples, height, width), dtype=np.uint8)
    
    # Train the detector
    pca_detector.train(faces, non_faces)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, 'models/pca_face_model.joblib')
    pca_detector.save_model('models/pca_detector.joblib')
    
    # Update model state
    model_state = {
        'model_trained': True,
        'pca_trained': True,
        'face_samples': n_samples,
        'explained_variance': explained_variance[:10].tolist(),  # Save first 10 components
        'training_time': "2025-03-11 19:48:39",  # Just a placeholder time
        'datasets_used': ['att', 'yale']
    }
    
    with open('model_state.json', 'w') as f:
        json.dump(model_state, f)
    
    print("PCA model training complete and saved to models/pca_face_model.joblib")
    print("PCA detector saved to models/pca_detector.joblib")
    print("Model state updated in model_state.json")
    
    return model_data, eigenfaces, mean_face

def visualize_eigenfaces(eigenfaces, mean_face, save_path='eigenfaces.png'):
    """Visualize eigenfaces and mean face"""
    n_eigenfaces = len(eigenfaces)
    n_cols = 4
    n_rows = (n_eigenfaces + 1) // n_cols + 1  # +1 for mean face
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2*n_rows))
    axes = axes.flatten()
    
    # Plot mean face
    axes[0].imshow(mean_face, cmap='gray')
    axes[0].set_title('Mean Face')
    axes[0].axis('off')
    
    # Plot eigenfaces
    for i, eigenface in enumerate(eigenfaces):
        h, w = mean_face.shape
        ef = eigenface.reshape(h, w)
        
        # Normalize eigenface for visualization
        ef = (ef - ef.min()) / (ef.max() - ef.min())
        
        axes[i+1].imshow(ef, cmap='gray')
        axes[i+1].set_title(f'Eigenface {i+1}')
        axes[i+1].axis('off')
    
    # Hide empty subplots
    for i in range(n_eigenfaces + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Eigenfaces visualization saved to {save_path}")

def main():
    """Main function to download and train datasets"""
    # Download datasets
    att_dir = download_att_database()
    yale_dir = download_yale_database()
    
    # Check if at least one dataset is available
    if att_dir is None and yale_dir is None:
        print("No datasets available. Please download datasets manually.")
        return
    
    # Load faces from available datasets
    all_faces = []
    all_labels = []
    
    # Try to load AT&T faces
    if att_dir is not None:
        att_faces, att_labels = load_att_faces(att_dir)
        if att_faces is not None and len(att_faces) > 0:
            all_faces.append(att_faces)
            all_labels.append(att_labels)
    
    # Try to load Yale faces
    if yale_dir is not None:
        yale_faces, yale_labels = load_yale_faces(yale_dir)
        if yale_faces is not None and len(yale_faces) > 0:
            all_faces.append(yale_faces)
            all_labels.append(yale_labels)
    
    # Combine datasets if both were loaded
    if all_faces:
        if len(all_faces) > 1:
            faces = np.vstack(all_faces)
            labels = np.concatenate(all_labels)
        else:
            faces = all_faces[0]
            labels = all_labels[0]
        
        # Train PCA model
        model_data, eigenfaces, mean_face = train_pca_model(faces, labels)
        
        # Visualize eigenfaces
        visualize_eigenfaces(eigenfaces, mean_face)
    else:
        print("No face images could be loaded. Please check the dataset directories.")

if __name__ == "__main__":
    main() 