import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from tqdm import tqdm

class FaceRecognitionModel:
    def __init__(self, method='hybrid', n_components_pca=0.95, n_components_lda=None):
        """
        Initialize a face recognition model.
        
        Parameters:
        -----------
        method : str
            Method to use for face recognition:
            - 'pca': Principal Component Analysis (Eigenfaces)
            - 'lda': Linear Discriminant Analysis (Fisherfaces)
            - 'hybrid': PCA followed by LDA (default)
        n_components_pca : int or float
            Number of components for PCA:
            - If int: Exact number of components
            - If float between 0 and 1: Proportion of variance to retain
        n_components_lda : int or None
            Number of components for LDA (if using 'lda' or 'hybrid' method)
            If None, will use min(n_classes-1, n_features)
        """
        self.method = method
        self.n_components_pca = n_components_pca
        self.n_components_lda = n_components_lda
        self.pca = None
        self.lda = None
        self.scaler = StandardScaler()
        self.classifier = None
        self.trained = False
        self.target_size = (100, 100)  # Match preprocessed images
        self.class_names = None
        
    def load_images(self, gallery_dir, use_processed=True, use_basic=False, limit_per_person=None):
        """
        Load face images from gallery directories.
        
        Parameters:
        -----------
        gallery_dir : str
            Path to the gallery directory
        use_processed : bool
            Whether to use images from the processed directories
        use_basic : bool
            Whether to use images from the basic directories
        limit_per_person : int or None
            Maximum number of images to load per person
            
        Returns:
        --------
        X : numpy.ndarray
            Face images as a 2D array (n_samples, n_features)
        y : numpy.ndarray
            Labels (person names)
        """
        images = []
        labels = []
        
        # Get member directories (exclude _processed and _basic)
        member_dirs = [d for d in os.listdir(gallery_dir) 
                      if os.path.isdir(os.path.join(gallery_dir, d)) 
                      and not d.endswith('_processed') 
                      and not d.endswith('_basic')]
        
        print(f"Found {len(member_dirs)} members: {', '.join(member_dirs)}")
        
        for member in member_dirs:
            # Determine image directories to use based on parameters
            image_dirs = []
            
            if use_processed:
                processed_dir = os.path.join(gallery_dir, f"{member}_processed")
                if os.path.exists(processed_dir):
                    image_dirs.append(processed_dir)
            
            if use_basic:
                basic_dir = os.path.join(gallery_dir, f"{member}_basic")
                if os.path.exists(basic_dir):
                    image_dirs.append(basic_dir)
            
            # Load images from directories
            member_images = []
            for img_dir in image_dirs:
                image_files = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                             glob.glob(os.path.join(img_dir, "*.jpeg")) + \
                             glob.glob(os.path.join(img_dir, "*.png"))
                
                print(f"Found {len(image_files)} images for {member} in {img_dir}")
                
                for img_path in image_files:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Ensure image has the right size
                        if img.shape != self.target_size:
                            img = cv2.resize(img, self.target_size)
                        member_images.append(img)
            
            # Apply limit if specified
            if limit_per_person is not None and len(member_images) > limit_per_person:
                member_images = member_images[:limit_per_person]
            
            # Add images and labels
            if member_images:
                images.extend(member_images)
                labels.extend([member] * len(member_images))
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Reshape to 2D array (n_samples, n_pixels)
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1)
        
        print(f"Loaded {n_samples} images with shape {X.shape}")
        
        return X, y
    
    def train(self, X, y, classifier='svm', test_size=0.2, random_state=42):
        """
        Train the face recognition model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Face images as a 2D array (n_samples, n_features)
        y : numpy.ndarray
            Labels (person names)
        classifier : str
            Classifier to use after feature extraction:
            - 'svm': Support Vector Machine (default)
            - 'knn': K-Nearest Neighbors
        test_size : float
            Proportion of the dataset to use for testing
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing training results:
            - 'accuracy': Test accuracy
            - 'report': Classification report
            - 'confusion_matrix': Confusion matrix
        """
        # Store class names
        self.class_names = np.unique(y)
        print(f"Training with {len(self.class_names)} classes: {self.class_names}")
        
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Standardize features
        X_train_std = self.scaler.fit_transform(X_train)
        X_test_std = self.scaler.transform(X_test)
        
        # Feature extraction based on method
        if self.method == 'pca':
            print("Using PCA (Eigenfaces) method")
            # Apply PCA
            self.pca = PCA(n_components=self.n_components_pca)
            X_train_features = self.pca.fit_transform(X_train_std)
            X_test_features = self.pca.transform(X_test_std)
            
            # Calculate explained variance
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA: Using {self.pca.n_components_} components, explained variance: {explained_variance:.2%}")
            
        elif self.method == 'lda':
            print("Using LDA (Fisherfaces) method")
            # Apply LDA directly
            if self.n_components_lda is None:
                self.n_components_lda = min(len(self.class_names) - 1, X_train_std.shape[1])
            
            self.lda = LinearDiscriminantAnalysis(n_components=self.n_components_lda)
            X_train_features = self.lda.fit_transform(X_train_std, y_train)
            X_test_features = self.lda.transform(X_test_std)
            
            # Print using the parameter value instead of looking for n_components_ attribute
            print(f"LDA: Using {self.n_components_lda} components")
            
        elif self.method == 'hybrid':
            print("Using Hybrid PCA+LDA method")
            # Apply PCA first for dimensionality reduction
            self.pca = PCA(n_components=self.n_components_pca)
            X_train_pca = self.pca.fit_transform(X_train_std)
            X_test_pca = self.pca.transform(X_test_std)
            
            # Then apply LDA for class discrimination
            if self.n_components_lda is None:
                self.n_components_lda = min(len(self.class_names) - 1, X_train_pca.shape[1])
            
            self.lda = LinearDiscriminantAnalysis(n_components=self.n_components_lda)
            X_train_features = self.lda.fit_transform(X_train_pca, y_train)
            X_test_features = self.lda.transform(X_test_pca)
            
            # Calculate explained variance from PCA
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA: Using {self.pca.n_components_} components, explained variance: {explained_variance:.2%}")
            # Print using the parameter value instead of looking for n_components_ attribute
            print(f"LDA: Using {self.n_components_lda} components")
        
        # Train classifier
        if classifier == 'knn':
            print("Training KNN classifier")
            self.classifier = KNeighborsClassifier(n_neighbors=min(5, len(y_train)))
        else:  # Default to SVM
            print("Training SVM classifier")
            self.classifier = SVC(kernel='rbf', probability=True)
        
        self.classifier.fit(X_train_features, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(class_report)
        
        # Set trained flag
        self.trained = True
        
        return {
            'accuracy': accuracy,
            'report': class_report,
            'confusion_matrix': conf_matrix,
            'test_predictions': y_pred,
            'test_labels': y_test,
            'test_features': X_test_features
        }
    
    def predict(self, image):
        """
        Predict the identity of a face image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Face image to predict
            
        Returns:
        --------
        tuple
            (predicted_label, confidence)
        """
        if not self.trained:
            raise ValueError("Model is not trained yet")
        
        # Ensure image has the right size
        if image.shape != self.target_size:
            image = cv2.resize(image, self.target_size)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Reshape and standardize
        X = image.reshape(1, -1)
        X_std = self.scaler.transform(X)
        
        # Transform based on method
        if self.method == 'pca':
            features = self.pca.transform(X_std)
        elif self.method == 'lda':
            features = self.lda.transform(X_std)
        elif self.method == 'hybrid':
            X_pca = self.pca.transform(X_std)
            features = self.lda.transform(X_pca)
        
        # Predict label and confidence
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            label_idx = np.argmax(probabilities)
            confidence = probabilities[label_idx]
            predicted_label = self.classifier.classes_[label_idx]
        else:
            predicted_label = self.classifier.predict(features)[0]
            # Calculate distance-based confidence for SVM
            if hasattr(self.classifier, 'decision_function'):
                decision = self.classifier.decision_function(features)
                confidence = 1 / (1 + np.exp(-np.abs(decision)))
            else:
                confidence = 0.5  # Default confidence if not available
        
        return predicted_label, confidence
    
    def save_model(self, model_path):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        """
        if not self.trained:
            raise ValueError("Model is not trained yet")
        
        # Create model data dictionary
        model_data = {
            'method': self.method,
            'n_components_pca': self.n_components_pca,
            'n_components_lda': self.n_components_lda,
            'pca': self.pca,
            'lda': self.lda,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'target_size': self.target_size,
            'class_names': self.class_names
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model_data = joblib.load(model_path)
        
        # Set model parameters
        self.method = model_data['method']
        self.n_components_pca = model_data['n_components_pca']
        self.n_components_lda = model_data['n_components_lda']
        self.pca = model_data['pca']
        self.lda = model_data['lda']
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.target_size = model_data['target_size']
        self.class_names = model_data['class_names']
        self.trained = True
        
        print(f"Model loaded from {model_path}")
        print(f"Method: {self.method}")
        print(f"Classes: {self.class_names}")
    
    def visualize_eigenfaces(self, n_components=16, figsize=(12, 8)):
        """
        Visualize eigenfaces from PCA components.
        
        Parameters:
        -----------
        n_components : int
            Number of eigenfaces to visualize
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing eigenfaces visualization
        """
        if self.pca is None:
            raise ValueError("PCA model not available")
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=int(np.ceil(n_components / 4)), 
            ncols=4, 
            figsize=figsize,
            subplot_kw={'xticks': [], 'yticks': []}
        )
        
        # Get components
        components = self.pca.components_
        n_display = min(n_components, len(components))
        
        # Plot eigenfaces
        for i, ax in enumerate(axes.flat):
            if i < n_display:
                # Reshape component to image
                eigenface = components[i].reshape(self.target_size)
                # Normalize to [0, 1] for display
                eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
                # Plot
                ax.imshow(eigenface, cmap='gray')
                ax.set_title(f"Eigenface {i+1}")
            else:
                ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing confusion matrix visualization
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = np.unique(y_true)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Set title and labels
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.tight_layout()
        
        return fig

def optimize_model_parameters(gallery_dir, methods=None, classifiers=None, 
                             pca_components=None, test_size=0.2, random_state=42):
    """
    Optimize face recognition model parameters.
    
    Parameters:
    -----------
    gallery_dir : str
        Path to the gallery directory
    methods : list or None
        List of methods to try ('pca', 'lda', 'hybrid')
    classifiers : list or None
        List of classifiers to try ('svm', 'knn')
    pca_components : list or None
        List of PCA components to try (int or float values)
    test_size : float
        Proportion of the dataset to use for testing
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing optimization results
    """
    # Set default values if not provided
    if methods is None:
        methods = ['hybrid', 'pca', 'lda']
    if classifiers is None:
        classifiers = ['svm', 'knn']
    if pca_components is None:
        pca_components = [0.9, 0.95, 0.99]
    
    # Initialize results
    results = []
    
    # Load images once for all tests
    print("Loading images...")
    model = FaceRecognitionModel()
    X, y = model.load_images(gallery_dir, use_processed=True, use_basic=False)
    
    # Try different combinations
    for method in methods:
        for classifier in classifiers:
            for n_components in pca_components:
                if method == 'lda' and isinstance(n_components, float):
                    # Skip PCA components for LDA-only method
                    continue
                
                print(f"\nTrying: method={method}, classifier={classifier}, components={n_components}")
                
                # Create model
                model = FaceRecognitionModel(method=method, n_components_pca=n_components)
                
                # Train and evaluate
                try:
                    train_results = model.train(X, y, classifier=classifier, 
                                               test_size=test_size, random_state=random_state)
                    
                    # Store results
                    results.append({
                        'method': method,
                        'classifier': classifier,
                        'n_components': n_components,
                        'accuracy': train_results['accuracy'],
                        'model': model
                    })
                except Exception as e:
                    print(f"Error: {str(e)}")
    
    # Check if we have any successful results
    if not results:
        print("\nNo successful models were trained. Trying with simpler PCA-only method.")
        try:
            # Fallback to basic PCA with fixed number of components
            model = FaceRecognitionModel(method='pca', n_components_pca=min(30, X.shape[0] // 2))
            train_results = model.train(X, y, classifier='svm', test_size=test_size, random_state=random_state)
            results.append({
                'method': 'pca',
                'classifier': 'svm',
                'n_components': min(30, X.shape[0] // 2),
                'accuracy': train_results['accuracy'],
                'model': model
            })
        except Exception as e:
            print(f"Error with fallback method: {str(e)}")
            # If we still have no results, we can't proceed
            if not results:
                raise ValueError("Could not train any models, even with fallback method. Please check your data.")
    
    # Sort results by accuracy
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Return best result
    best_result = results[0]
    print(f"\nBest result: {best_result['method']}, {best_result['classifier']}, " + 
          f"components={best_result['n_components']}, accuracy={best_result['accuracy']:.2%}")
    
    return {
        'best_model': best_result['model'],
        'best_params': {
            'method': best_result['method'],
            'classifier': best_result['classifier'],
            'n_components': best_result['n_components']
        },
        'best_accuracy': best_result['accuracy'],
        'all_results': results
    }

if __name__ == "__main__":
    print("Face Recognition Model Module")
    print("This module provides classes and functions for training and evaluating face recognition models.")
    print("Import and use these functions in your application.") 