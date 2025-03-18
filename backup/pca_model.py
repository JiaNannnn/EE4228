import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime

def train_and_save_pca_model(faces, labels=None, target_size=(100, 100), variance_to_retain=0.95, 
                            output_path='models/pca_face_model.joblib', visualize=False):
    """
    Train a PCA model on face images and save it to disk
    
    Parameters:
    -----------
    faces : numpy.ndarray
        Array of face images
    labels : numpy.ndarray, optional
        Array of face labels (needed for eigenfaces visualization)
    target_size : tuple
        Target size of face images (width, height)
    variance_to_retain : float
        Fraction of variance to retain (0.0 to 1.0)
    output_path : str
        Path to save the PCA model
    visualize : bool
        Whether to visualize eigenfaces and explained variance
        
    Returns:
    --------
    model_data : dict
        Dictionary containing PCA model data
    """
    try:
        if len(faces) == 0:
            raise ValueError("No faces provided for training")
        
        # Validate input faces
        valid_faces = []
        valid_labels = []
        
        for i, face in enumerate(faces):
            try:
                # Check if face is valid
                if face is None:
                    print(f"Skipping None face at index {i}")
                    continue
                    
                if np.isnan(face).any():
                    print(f"Skipping face with NaN values at index {i}")
                    continue
                    
                if face.shape[:2] != target_size:
                    # Try to resize the face
                    try:
                        face = cv2.resize(face, target_size)
                    except Exception as e:
                        print(f"Error resizing face at index {i}: {str(e)}")
                        continue
                
                valid_faces.append(face)
                if labels is not None:
                    valid_labels.append(labels[i])
                    
            except Exception as e:
                print(f"Error validating face at index {i}: {str(e)}")
                continue
        
        if len(valid_faces) == 0:
            raise ValueError("No valid faces found after validation")
        
        # Convert to numpy arrays
        faces = np.array(valid_faces)
        if labels is not None:
            labels = np.array(valid_labels)
        
        print(f"Training with {len(faces)} valid faces")
        
        # Flatten faces
        n_samples, h, w = faces.shape
        X = faces.reshape(n_samples, h * w)
        
        # Check for invalid values
        if np.isnan(X).any():
            raise ValueError("NaN values detected in flattened faces")
        if np.isinf(X).any():
            raise ValueError("Infinite values detected in flattened faces")
        
        # Standardize features with error checking
        try:
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            
            # Check for NaN values after standardization
            if np.isnan(X_std).any():
                raise ValueError("NaN values detected after standardization")
            
        except Exception as e:
            raise ValueError(f"Error during standardization: {str(e)}")
        
        # Apply PCA with error handling
        try:
            # If we have very few samples, adjust the number of components
            max_components = min(n_samples - 1, X_std.shape[1])
            if isinstance(variance_to_retain, float):
                n_components = min(max_components, int(X_std.shape[1] * variance_to_retain))
            else:
                n_components = min(max_components, variance_to_retain)
                
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(X_std)
            
            # Verify the results
            if np.isnan(pca.components_).any():
                raise ValueError("NaN values detected in PCA components")
            
        except Exception as e:
            raise ValueError(f"Error during PCA fitting: {str(e)}")
        
        # Create model data
        model_data = {
            'pca': pca,
            'scaler': scaler,
            'mean_face': scaler.mean_,
            'variance_explained': pca.explained_variance_ratio_,
            'n_components': pca.n_components_,
            'target_size': target_size,
            'faces_shape': faces.shape,
            'training_samples': n_samples,
            'training_date': datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model with error handling
        try:
            joblib.dump(model_data, output_path)
            print(f"PCA model saved to {output_path}")
            
            # Verify the saved file
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # Less than 1KB is suspicious
                raise ValueError(f"Saved model file is suspiciously small ({file_size} bytes)")
                
        except Exception as e:
            raise ValueError(f"Error saving model: {str(e)}")
        
        print(f"Number of components: {pca.n_components_}")
        print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2f}")
        
        # Visualize eigenfaces and explained variance if requested
        if visualize:
            try:
                visualize_eigenfaces(model_data, labels, output_dir=os.path.dirname(output_path))
                plot_explained_variance(model_data, output_path=os.path.join(os.path.dirname(output_path), 'variance_explained.png'))
            except Exception as e:
                print(f"Warning: Error during visualization: {str(e)}")
        
        return model_data
        
    except Exception as e:
        print(f"Error during PCA model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def load_pca_model(model_path='models/pca_face_model.joblib'):
    """
    Load a PCA model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the PCA model file
        
    Returns:
    --------
    model_data : dict
        Dictionary containing PCA model data
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PCA model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    
    print(f"PCA model loaded from {model_path}")
    print(f"Number of components: {model_data['n_components']}")
    print(f"Variance explained: {sum(model_data['variance_explained']):.2f}")
    
    return model_data

def visualize_eigenfaces(model_data, labels=None, n_eigenfaces=16, output_dir='models'):
    """
    Visualize eigenfaces from a PCA model
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing PCA model data
    labels : numpy.ndarray, optional
        Array of face labels
    n_eigenfaces : int
        Number of eigenfaces to visualize
    output_dir : str
        Directory to save the visualizations
    """
    pca = model_data['pca']
    components = pca.components_
    target_size = model_data['target_size']
    
    # Limit to the actual number of components
    n_eigenfaces = min(n_eigenfaces, len(components))
    
    # Calculate rows and columns for the grid
    n_cols = min(4, n_eigenfaces)
    n_rows = int(np.ceil(n_eigenfaces / n_cols))
    
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    
    for i in range(n_eigenfaces):
        # Reshape eigenface
        eigenface = components[i].reshape(target_size)
        
        # Normalize to [0, 1] for visualization
        eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'eigenfaces.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Eigenfaces visualization saved to {output_path}")

def plot_explained_variance(model_data, output_path='models/variance_explained.png'):
    """
    Plot the explained variance from a PCA model
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing PCA model data
    output_path : str
        Path to save the plot
    """
    variance_ratio = model_data['variance_explained']
    cumulative_variance = np.cumsum(variance_ratio)
    
    plt.figure(figsize=(10, 5))
    
    # Plot individual and cumulative explained variance
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.5, label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% explained variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% explained variance')
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Components')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Find how many components needed for 95% and 99% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    plt.annotate(f'95% variance: {n_components_95} components', 
                xy=(n_components_95, 0.95), 
                xytext=(n_components_95 + 5, 0.9),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.annotate(f'99% variance: {n_components_99} components', 
                xy=(n_components_99, 0.99), 
                xytext=(n_components_99 + 5, 0.94),
                arrowprops=dict(facecolor='green', shrink=0.05))
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Explained variance plot saved to {output_path}")

def create_hybrid_pca_lda_model(faces, labels, target_size=(100, 100), pca_variance=0.95, 
                               output_path='models/hybrid_pca_lda_model.joblib'):
    """
    Create a hybrid model combining PCA and LDA
    
    Parameters:
    -----------
    faces : numpy.ndarray
        Array of face images
    labels : numpy.ndarray
        Array of face labels
    target_size : tuple
        Target size of face images (width, height)
    pca_variance : float
        Fraction of variance to retain in PCA (0.0 to 1.0)
    output_path : str
        Path to save the hybrid model
        
    Returns:
    --------
    model_data : dict
        Dictionary containing hybrid model data
    """
    if len(faces) == 0 or len(labels) == 0:
        raise ValueError("No faces or labels provided for training")
    
    if len(faces) != len(labels):
        raise ValueError("Number of faces and labels must match")
    
    # Check if all faces have the same size
    if any(face.shape[:2] != target_size for face in faces):
        # Resize faces to the target size
        faces = np.array([cv2.resize(face, target_size) for face in faces])
    
    # Flatten faces
    n_samples, h, w = faces.shape
    X = faces.reshape(n_samples, h * w)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=pca_variance, svd_solver='full')
    X_pca = pca.fit_transform(X_std)
    
    # Apply LDA for class separation
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y)
    
    # Create model data
    model_data = {
        'pca': pca,
        'lda': lda,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_,
        'pca_variance': pca_variance,
        'pca_components': pca.n_components_,
        'lda_components': lda.n_components_,
        'target_size': target_size,
        'training_samples': n_samples
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    joblib.dump(model_data, output_path)
    print(f"Hybrid PCA+LDA model saved to {output_path}")
    print(f"PCA components: {pca.n_components_}, LDA components: {lda.n_components_}")
    print(f"Classes: {model_data['classes']}")
    
    return model_data

def load_hybrid_model(model_path='models/hybrid_pca_lda_model.joblib'):
    """
    Load a hybrid PCA+LDA model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the hybrid model file
        
    Returns:
    --------
    model_data : dict
        Dictionary containing hybrid model data
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hybrid model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    
    print(f"Hybrid PCA+LDA model loaded from {model_path}")
    print(f"PCA components: {model_data['pca_components']}, LDA components: {model_data['lda_components']}")
    print(f"Classes: {model_data['classes']}")
    
    return model_data

def predict_with_pca_model(face, model_data):
    """
    Transform a face using the PCA model
    
    Parameters:
    -----------
    face : numpy.ndarray
        Face image to transform
    model_data : dict
        Dictionary containing PCA model data
        
    Returns:
    --------
    embedding : numpy.ndarray
        PCA embedding of the face
    """
    target_size = model_data['target_size']
    
    # Resize face if needed
    if face.shape[:2] != target_size:
        face = cv2.resize(face, target_size)
    
    # Flatten face
    face_flat = face.flatten().reshape(1, -1)
    
    # Standardize face
    face_std = model_data['scaler'].transform(face_flat)
    
    # Transform face using PCA
    embedding = model_data['pca'].transform(face_std)[0]
    
    return embedding

def predict_with_hybrid_model(face, model_data):
    """
    Predict the class of a face using the hybrid PCA+LDA model
    
    Parameters:
    -----------
    face : numpy.ndarray
        Face image to classify
    model_data : dict
        Dictionary containing hybrid model data
        
    Returns:
    --------
    prediction : tuple
        Tuple containing (predicted_class, probability)
    """
    target_size = model_data['target_size']
    
    # Resize face if needed
    if face.shape[:2] != target_size:
        face = cv2.resize(face, target_size)
    
    # Flatten face
    face_flat = face.flatten().reshape(1, -1)
    
    # Standardize face
    face_std = model_data['scaler'].transform(face_flat)
    
    # Transform face using PCA
    face_pca = model_data['pca'].transform(face_std)
    
    # Predict class using LDA
    predicted_class_idx = model_data['lda'].predict(face_pca)[0]
    predicted_class = model_data['classes'][predicted_class_idx]
    
    # Get probability if available
    if hasattr(model_data['lda'], 'predict_proba'):
        probability = model_data['lda'].predict_proba(face_pca)[0][predicted_class_idx]
    else:
        # Use decision function as a proxy for confidence
        decision_values = model_data['lda'].decision_function(face_pca)[0]
        probability = 1.0 / (1.0 + np.exp(-decision_values[predicted_class_idx]))
    
    return (predicted_class, probability)

if __name__ == "__main__":
    print("This module provides functions for training and loading PCA face recognition models.")
    print("Import and use these functions in your application.") 