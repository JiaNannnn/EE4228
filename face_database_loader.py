import os
import requests
import zipfile
import tarfile
import numpy as np
import cv2
from tqdm import tqdm
import streamlit as st
import time
from sklearn.datasets import fetch_lfw_people

class FaceDatabaseLoader:
    """Utility class to download and prepare external face databases"""
    
    DATABASES = {
        'att': {
            'url': 'https://cs.nyu.edu/~roweis/data/olivettifaces.mat',
            'description': 'Olivetti/AT&T Database: 400 images of 40 subjects',
            'type': 'mat'
        },
        'synthetic': {
            'url': None,  # Generated internally
            'description': 'Synthetic Face Database: Programmatically generated faces',
            'type': 'internal'
        },
        'lfw': {
            'url': None,  # Using scikit-learn's built-in fetcher
            'description': 'Labeled Faces in the Wild: Sample of face images',
            'type': 'internal'
        }
    }
    
    def __init__(self, data_dir='external_databases'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.background_dir = os.path.join(self.data_dir, 'background')
        os.makedirs(self.background_dir, exist_ok=True)
        
    def list_available_databases(self):
        """List all available databases with descriptions"""
        st.write("### Available Face Databases:")
        for name, info in self.DATABASES.items():
            st.write(f"\n**{name.upper()}**:")
            st.write(info['description'])
            if name == 'synthetic' or name == 'lfw':
                st.write("✅ Always available (generated on-demand)")
            elif os.path.exists(os.path.join(self.data_dir, name)):
                st.write("✅ Downloaded")
            else:
                st.write("❌ Not downloaded")
        
        # Check background images
        if not os.listdir(self.background_dir):
            st.write("\n### Background Images (for non-face samples):")
            st.warning("No background images found. Will use synthetic non-face images.")
            st.write("Auto-generating background images...")
            self._generate_synthetic_backgrounds(10)
    
    def _generate_synthetic_backgrounds(self, count=10):
        """Generate synthetic background images for training"""
        for i in range(count):
            # Create random noise image
            noise = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            
            # Apply some filtering to make it look more natural
            noise = cv2.GaussianBlur(noise, (15, 15), 0)
            
            # Create gradient image
            gradient = np.zeros((512, 512), dtype=np.uint8)
            for y in range(512):
                gradient[y, :] = y // 2
            
            # Mix noise and gradient
            bg_image = cv2.addWeighted(noise, 0.7, gradient, 0.3, 0)
            
            # Add some random shapes
            for _ in range(np.random.randint(5, 15)):
                # Random rectangle
                pt1 = (np.random.randint(0, 400), np.random.randint(0, 400))
                pt2 = (pt1[0] + np.random.randint(50, 150), pt1[1] + np.random.randint(50, 150))
                color = np.random.randint(0, 255)
                thickness = np.random.randint(1, 5)
                cv2.rectangle(bg_image, pt1, pt2, color, thickness)
            
            # Save the image
            save_path = os.path.join(self.background_dir, f"synthetic_bg_{i}.jpg")
            cv2.imwrite(save_path, bg_image)
        
        st.success(f"Generated {count} synthetic background images")
        
    def _generate_synthetic_faces(self, num_samples=100, target_size=(100, 100)):
        """Generate synthetic face-like patterns for training"""
        st.info("Generating synthetic face patterns...")
        faces = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        for i in range(num_samples):
            # Create base oval face shape
            face = np.zeros(target_size, dtype=np.uint8)
            center = (target_size[0] // 2, target_size[1] // 2)
            axes = (target_size[0] // 3, target_size[1] // 2)
            cv2.ellipse(face, center, axes, 0, 0, 360, 200, -1)
            
            # Add eyes (two small circles)
            eye_y = target_size[1] // 3
            left_eye_x = target_size[0] // 3
            right_eye_x = target_size[0] * 2 // 3
            eye_radius = target_size[0] // 10
            
            cv2.circle(face, (left_eye_x, eye_y), eye_radius, 100, -1)
            cv2.circle(face, (right_eye_x, eye_y), eye_radius, 100, -1)
            
            # Add mouth (curved line)
            mouth_y = target_size[1] * 7 // 10
            mouth_width = target_size[0] // 2
            
            mouth_start = (target_size[0] // 4, mouth_y)
            mouth_end = (target_size[0] * 3 // 4, mouth_y)
            mouth_ctrl = (target_size[0] // 2, mouth_y + 10)
            
            for t in np.linspace(0, 1, 20):
                # Quadratic Bezier curve
                x = int((1-t)**2 * mouth_start[0] + 2*(1-t)*t*mouth_ctrl[0] + t**2*mouth_end[0])
                y = int((1-t)**2 * mouth_start[1] + 2*(1-t)*t*mouth_ctrl[1] + t**2*mouth_end[1])
                cv2.circle(face, (x, y), 1, 150, 2)
            
            # Add some noise and blur to make it realistic
            noise = np.random.randint(0, 30, target_size, dtype=np.uint8)
            face = cv2.add(face, noise)
            face = cv2.GaussianBlur(face, (5, 5), 0)
            
            faces.append(face)
            progress_bar.progress(min(1.0, (i + 1) / num_samples))
        
        return np.array(faces)
        
    def download_database(self, database_name):
        """Download or generate a specific face database"""
        if database_name not in self.DATABASES:
            raise ValueError(f"Unknown database: {database_name}")
            
        info = self.DATABASES[database_name]
        target_dir = os.path.join(self.data_dir, database_name)
        
        # Handle synthetic database
        if database_name == 'synthetic':
            st.info("Generating synthetic face database...")
            return True
            
        # Handle LFW through scikit-learn
        if database_name == 'lfw':
            st.info("Fetching LFW dataset through scikit-learn...")
            try:
                # This will download LFW if not already downloaded
                fetch_lfw_people(min_faces_per_person=10, resize=0.5)
                return True
            except Exception as e:
                st.error(f"Failed to fetch LFW dataset: {str(e)}")
                return False
                
        # Handle standard downloads
        archive_path = os.path.join(self.data_dir, f"{database_name}.{info['type']}")
        
        try:
            if os.path.exists(target_dir):
                st.info(f"Database {database_name} already exists in {target_dir}")
                return True
                
            st.info(f"Downloading {database_name} database...")
            
            # Download with progress bar
            response = requests.get(info['url'], stream=True, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded_size = 0
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size:
                            progress = min(1.0, downloaded_size / total_size)
                            progress_bar.progress(progress)
            
            # Handle different file types
            os.makedirs(target_dir, exist_ok=True)
            
            if info['type'] == 'zip':
                # Verify archive
                if not zipfile.is_zipfile(archive_path):
                    raise ValueError("Downloaded file is not a valid ZIP archive")
                    
                st.info("Extracting files...")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                    
            elif info['type'] == 'tar':
                st.info("Extracting files...")
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir)
            
            elif info['type'] == 'mat':
                # Just copy the file to the target directory
                import shutil
                shutil.copy(archive_path, os.path.join(target_dir, os.path.basename(archive_path)))
                    
            # Clean up
            if os.path.exists(archive_path):
                os.remove(archive_path)
                
            st.success(f"Successfully downloaded {database_name} database!")
            return True
            
        except requests.exceptions.RequestException as e:
            st.error(f"Download failed: {str(e)}")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            if os.path.exists(target_dir) and not os.listdir(target_dir):
                import shutil
                shutil.rmtree(target_dir)
            return False
            
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            return False
        
    def load_database(self, database_name, target_size=(100, 100)):
        """Load and preprocess images from a database"""
        # Handle synthetic database
        if database_name == 'synthetic':
            faces = self._generate_synthetic_faces(num_samples=100, target_size=target_size)
            labels = np.array(['synthetic_face'] * len(faces))
            return faces, labels
            
        # Handle LFW through scikit-learn
        if database_name == 'lfw':
            try:
                st.info("Loading LFW dataset...")
                lfw = fetch_lfw_people(min_faces_per_person=10, resize=0.5)
                faces = []
                labels = []
                
                # Create progress bar
                progress_bar = st.progress(0)
                
                for i in range(len(lfw.images)):
                    face = cv2.resize(lfw.images[i], target_size)
                    faces.append(face)
                    labels.append(lfw.target_names[lfw.target[i]])
                    progress_bar.progress(min(1.0, (i + 1) / len(lfw.images)))
                
                return np.array(faces), np.array(labels)
            except Exception as e:
                st.error(f"Failed to load LFW dataset: {str(e)}")
                return np.array([]), np.array([])
        
        # Handle standard databases
        target_dir = os.path.join(self.data_dir, database_name)
        if not os.path.exists(target_dir):
            if not self.download_database(database_name):
                return np.array([]), np.array([])
            
        images = []
        labels = []
        
        st.info(f"Loading {database_name} database...")
        progress_bar = st.progress(0)
        
        # Special handling for .mat files (Olivetti/AT&T)
        if database_name == 'att':
            try:
                from scipy.io import loadmat
                mat_file = os.path.join(target_dir, 'olivettifaces.mat')
                if os.path.exists(mat_file):
                    mat_data = loadmat(mat_file)
                    faces = mat_data['faces'].T
                    for i in range(faces.shape[0]):
                        face = faces[i].reshape(64, 64)
                        face = cv2.resize(face, target_size)
                        images.append(face)
                        # Use subject ID as label (10 images per subject, 40 subjects)
                        label = f"subject_{i // 10 + 1}"
                        labels.append(label)
                        progress_bar.progress(min(1.0, (i + 1) / faces.shape[0]))
                    return np.array(images), np.array(labels)
            except Exception as e:
                st.warning(f"Error processing .mat file: {str(e)}")
        
        # Standard image file handling
        file_list = []
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm')):
                    file_list.append((root, file))
        
        if not file_list:
            st.warning(f"No valid images found in {database_name} database")
            return np.array([]), np.array([])
        
        # Process files with progress bar
        for i, (root, file) in enumerate(file_list):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                try:
                    # Resize image
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    
                    # Extract label from directory name
                    label = os.path.basename(root)
                    labels.append(label)
                except Exception as e:
                    st.warning(f"Error processing {img_path}: {str(e)}")
            
            progress_bar.progress(min(1.0, (i + 1) / len(file_list)))
        
        if not images:
            st.warning(f"No valid images could be processed from {database_name} database")
            return np.array([]), np.array([])
            
        return np.array(images), np.array(labels)
        
    def get_face_samples(self, num_samples=1000, target_size=(100, 100)):
        """Get face samples from available databases"""
        all_images = []
        
        # Start with synthetic faces
        synthetic_faces, _ = self.load_database('synthetic', target_size)
        if len(synthetic_faces) > 0:
            all_images.append(synthetic_faces)
        
        # Try loading LFW 
        try:
            lfw_faces, _ = self.load_database('lfw', target_size)
            if len(lfw_faces) > 0:
                all_images.append(lfw_faces)
        except:
            pass
            
        # Try other databases
        for db_name in self.DATABASES.keys():
            if db_name not in ['synthetic', 'lfw']:
                try:
                    images, _ = self.load_database(db_name, target_size)
                    if len(images) > 0:
                        all_images.append(images)
                except Exception as e:
                    st.warning(f"Error loading {db_name}: {str(e)}")
                
        if not all_images:
            # Generate more synthetic samples as fallback
            st.warning("No external face samples could be loaded, generating more synthetic faces")
            synthetic_faces = self._generate_synthetic_faces(num_samples=num_samples, target_size=target_size)
            return synthetic_faces
            
        # Combine and shuffle
        all_images = np.concatenate(all_images)
        np.random.shuffle(all_images)
        
        # Return requested number of samples
        return all_images[:min(num_samples, len(all_images))]
        
    def get_non_face_samples(self, num_samples=1000, target_size=(100, 100)):
        """Generate non-face samples from random patches or synthetic data"""
        non_faces = []
        
        # Check if there are background images
        if not os.listdir(self.background_dir):
            st.warning(f"No background images found in: {self.background_dir}")
            st.info("Generating synthetic non-face patterns...")
            return self._generate_synthetic_non_faces(num_samples, target_size)
            
        # Use existing background images
        for img_file in os.listdir(self.background_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.background_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    h, w = img.shape
                    if h >= target_size[1] and w >= target_size[0]:
                        samples_per_image = num_samples // len(os.listdir(self.background_dir))
                        for _ in range(samples_per_image):
                            # Random crop
                            x = np.random.randint(0, w - target_size[0])
                            y = np.random.randint(0, h - target_size[1])
                            patch = img[y:y+target_size[1], x:x+target_size[0]]
                            non_faces.append(patch)
        
        if not non_faces:
            st.warning("Could not extract non-face samples from background images")
            return self._generate_synthetic_non_faces(num_samples, target_size)
            
        st.success(f"Created {len(non_faces)} non-face samples from background images")
        return np.array(non_faces)
        
    def _generate_synthetic_non_faces(self, num_samples=1000, target_size=(100, 100)):
        """Generate synthetic non-face patterns for training"""
        st.info("Generating synthetic non-face patterns...")
        non_faces = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        for i in range(num_samples):
            # Method 1: Random noise
            if i % 3 == 0:
                non_face = np.random.randint(0, 255, target_size, dtype=np.uint8)
                non_face = cv2.GaussianBlur(non_face, (5, 5), 0)
                
            # Method 2: Geometric shapes
            elif i % 3 == 1:
                non_face = np.zeros(target_size, dtype=np.uint8)
                # Add random shapes
                for _ in range(np.random.randint(1, 5)):
                    shape_type = np.random.randint(0, 3)
                    color = np.random.randint(100, 255)
                    
                    if shape_type == 0:  # Rectangle
                        pt1 = (np.random.randint(0, target_size[0]//2), np.random.randint(0, target_size[1]//2))
                        pt2 = (np.random.randint(pt1[0]+10, target_size[0]), np.random.randint(pt1[1]+10, target_size[1]))
                        cv2.rectangle(non_face, pt1, pt2, color, -1)
                        
                    elif shape_type == 1:  # Circle
                        center = (np.random.randint(0, target_size[0]), np.random.randint(0, target_size[1]))
                        radius = np.random.randint(5, min(target_size) // 2)
                        cv2.circle(non_face, center, radius, color, -1)
                        
                    else:  # Triangle
                        pts = np.array([
                            [np.random.randint(0, target_size[0]), np.random.randint(0, target_size[1])],
                            [np.random.randint(0, target_size[0]), np.random.randint(0, target_size[1])],
                            [np.random.randint(0, target_size[0]), np.random.randint(0, target_size[1])]
                        ])
                        cv2.fillPoly(non_face, [pts], color)
            
            # Method 3: Gradients with lines
            else:
                non_face = np.zeros(target_size, dtype=np.uint8)
                # Create gradient
                for y in range(target_size[1]):
                    value = int((y / target_size[1]) * 255)
                    non_face[y, :] = value
                
                # Add random lines
                for _ in range(np.random.randint(2, 6)):
                    pt1 = (np.random.randint(0, target_size[0]), np.random.randint(0, target_size[1]))
                    pt2 = (np.random.randint(0, target_size[0]), np.random.randint(0, target_size[1]))
                    color = np.random.randint(0, 255)
                    thickness = np.random.randint(1, 3)
                    cv2.line(non_face, pt1, pt2, color, thickness)
            
            non_faces.append(non_face)
            progress_bar.progress(min(1.0, (i + 1) / num_samples))
        
        st.success(f"Generated {len(non_faces)} synthetic non-face samples")
        return np.array(non_faces) 