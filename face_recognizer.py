import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib # Use joblib for saving/loading sklearn models
import os
import cv2  # For visualization if needed

class FaceRecognizer:
    """
    Recognizes faces using the Eigenfaces method (PCA for dimensionality reduction
    followed by a KNN classifier). Includes a check for unknown faces based on
    PCA reconstruction error.
    """
    def __init__(self, n_components=0.95, n_neighbors=5, reconstruction_error_threshold=0.01):
        """
        Initializes the FaceRecognizer.

        Args:
            n_components (float or int): Number of PCA components to keep.
                                        If float (0<n<1), it's the variance ratio.
                                        If int, it's the absolute number of components.
            n_neighbors (int): Number of neighbors for the KNN classifier.
            reconstruction_error_threshold (float): Threshold for PCA reconstruction error.
                                                    Faces with error above this are 'Unknown'.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reconstruction_threshold = reconstruction_error_threshold # Renamed from reconstruction_threshold for clarity

        # Core components: Scaler, PCA, KNN
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, svd_solver='auto', whiten=True) # Whiten can sometimes help KNN
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance', metric='euclidean')

        # State attributes
        self.trained = False
        self.user_names = []  # List of unique user names (labels) in order
        self.labels = []      # Numerical labels corresponding to user_names
        self.mean_face = None # Store the mean face calculated during training

        # Removed attributes related to LDA, SVM, RF, Ensemble, single-user mode, complex similarity, etc.

        
    def _prepare_data(self, X):
        """Flatten 2D images into 1D vectors"""
        if X is None or X.size == 0:
             raise ValueError("Input data X cannot be empty")
        if len(X.shape) == 3:
            n_samples, height, width = X.shape
            if n_samples == 0:
                raise ValueError("Input data X has 0 samples")
            return X.reshape(n_samples, height * width)
        elif len(X.shape) == 2: # Handle single image case
            height, width = X.shape
            return X.reshape(1, height * width)
        elif len(X.shape) == 1: # Already flat
            return X.reshape(1, -1)
        else:
            raise ValueError(f"Invalid input shape: {X.shape}")

    def _compute_reconstruction_error(self, X_flat):
        """
        Compute PCA reconstruction error for flattened input data X_flat.
        Assumes X_flat is already scaled.
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Cannot compute reconstruction error.")
        try:
            X_pca = self.pca.transform(X_flat)
            X_rec = self.pca.inverse_transform(X_pca)
            # Calculate Mean Squared Error between original scaled and reconstructed scaled
            error = np.mean(np.square(X_flat - X_rec), axis=1)
            return error
        except Exception as e:
            print(f"Error during reconstruction computation: {e}")
            # Return a high error value if computation fails
            return np.full(X_flat.shape[0], np.inf)
        
    def train(self, X, y):
        """
        Train the face recognition model using PCA and KNN.

        Args:
            X (np.ndarray): Training images (n_samples, height, width).
            y (np.ndarray): Training labels (n_samples,).
        """
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must match.")

        # 1. Prepare Data
        X_flat = self._prepare_data(np.array(X))

        # Handle potential NaNs before scaling
        if np.isnan(X_flat).any():
            print("Warning: NaN values detected in training data. Replacing with column means.")
            col_mean = np.nanmean(X_flat, axis=0)
             # Find indices where NaN value is present
            inds = np.where(np.isnan(X_flat))
            # Place column means in the indices. Align the arrays using take
            X_flat[inds] = np.take(col_mean, inds[1])
            # If any column mean is still NaN (all values were NaN), replace with 0
            X_flat = np.nan_to_num(X_flat, nan=0.0)
            

        # 2. Scale Data
        X_scaled = self.scaler.fit_transform(X_flat)
        self.mean_face = self.scaler.mean_ # Store the mean face (average pixel values)
        
        # Handle potential NaNs after scaling (if scaler fails on zero-variance columns)
        if np.isnan(X_scaled).any():
            print("Warning: NaN values detected after scaling. Replacing with zeros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        

        # 3. Apply PCA
        # Ensure n_components is valid
        n_samples, n_features = X_scaled.shape
        max_components = min(n_samples, n_features)

        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # Determine components based on variance, but cap at max_components
             # Temporarily fit PCA to determine components needed for variance
            temp_pca = PCA(n_components=min(0.999, self.n_components), svd_solver='full') # Use 'full' for variance check
            temp_pca.fit(X_scaled)
            self.pca.n_components_ = temp_pca.n_components_
            print(f"PCA determined {self.pca.n_components_} components for {self.n_components*100:.1f}% variance.")
        elif isinstance(self.n_components, int):
            self.pca.n_components_ = min(self.n_components, max_components)
            print(f"PCA using fixed {self.pca.n_components_} components.")
        else:
            # Default case or invalid n_components, use max possible
            self.pca.n_components_ = max_components
            print(f"PCA using maximum possible {self.pca.n_components_} components.")
        
        
        # Ensure at least 1 component
        self.pca.n_components_ = max(1, self.pca.n_components_)
        print(f"Final PCA components to use: {self.pca.n_components_}")
        
        
        try:
            # Fit PCA and transform data
            X_pca = self.pca.fit_transform(X_scaled)
        except ValueError as e:
             print(f"Error during PCA fitting: {e}. Trying with fewer components.")
             # Reduce components and retry if possible
             if self.pca.n_components_ > 1:
                 self.pca.n_components_ = self.pca.n_components_ // 2
                 print(f"Retrying PCA with {self.pca.n_components_} components.")
            X_pca = self.pca.fit_transform(X_scaled)
        else:
                 raise RuntimeError("PCA failed even with 1 component.") from e

        if X_pca.shape[1] == 0:
             raise RuntimeError("PCA resulted in 0 components. Check input data variance.")


        # 4. Train KNN Classifier
        # Store unique names and create numerical labels for KNN
        self.user_names = sorted(list(np.unique(y)))
        name_to_label = {name: i for i, name in enumerate(self.user_names)}
        self.labels = np.array([name_to_label[name] for name in y])

        self.knn.fit(X_pca, self.labels)
            
            self.trained = True
        print(f"Training complete. Model trained on {n_samples} images for {len(self.user_names)} users: {self.user_names}")
        print(f"PCA Components: {self.pca.n_components_}, KNN Neighbors: {self.n_neighbors}")

        
    def predict(self, face_image):
        """
        Predict the identity of a single face image.

        Args:
            face_image (np.ndarray): A single preprocessed face image (height, width).
                                     Should be grayscale and aligned/normalized appropriately
                                     before calling this method.

        Returns:
            tuple: (predicted_name, confidence, is_known)
                   predicted_name (str): The name of the recognized person or "Unknown".
                   confidence (float): Confidence score (higher is better).
                                       For KNN, this is the probability of the predicted class.
                                       For 'Unknown', it's related to reconstruction error.
                   is_known (bool): True if the face is recognized as someone in the gallery,
                                    False if deemed "Unknown".
        """
        if not self.trained:
            # raise RuntimeError("Model not trained yet.")
            print("Warning: Predict called before training. Returning Unknown.")
            return "Unknown", 0.0, False

        if face_image is None or face_image.size == 0:
            print("Warning: Empty face image received for prediction.")
            return "Unknown", 0.0, False

        # 1. Prepare Input Image
        try:
            X_flat = self._prepare_data(face_image)
            # Handle potential NaNs in input
            if np.isnan(X_flat).any():
                print("Warning: NaN values detected in prediction input. Replacing with mean face values.")
                 # Use the mean stored from training scaler
                inds = np.where(np.isnan(X_flat))
                if self.mean_face is not None and self.mean_face.shape == X_flat.shape[1:]:
                     X_flat[inds] = np.take(self.mean_face, inds[1])
                else: # Fallback if mean_face isn't available/compatible
                    X_flat = np.nan_to_num(X_flat, nan=0.0)

            X_scaled = self.scaler.transform(X_flat)
            
             # Handle NaNs after scaling
            if np.isnan(X_scaled).any():
                print("Warning: NaN values detected in prediction input after scaling. Replacing with zeros.")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
        except ValueError as e:
             print(f"Error preparing prediction data: {e}")
             return "Unknown", 0.0, False
        except Exception as e: # Catch sklearn scaler errors if not fitted
            print(f"Error scaling prediction data (model trained?): {e}")
            return "Unknown", 0.0, False


        # 2. Check Reconstruction Error (Unknown Face Detection)
        try:
            reconstruction_error = self._compute_reconstruction_error(X_scaled)[0]
            # Normalize error? Maybe not necessary, threshold depends on data scale
            # Lower error means the face fits the learned PCA space well.
            # Confidence related to error: 1.0 is perfect reconstruction, 0.0 is at threshold
            unknown_confidence = max(0.0, 1.0 - (reconstruction_error / self.reconstruction_threshold))

        except RuntimeError as e: # Model not trained
            print(f"Error checking reconstruction: {e}")
            return "Unknown", 0.0, False
        except Exception as e:
            print(f"Unexpected error during reconstruction check: {e}")
            reconstruction_error = float('inf') # Assume unknown if error calculation fails
            unknown_confidence = 0.0

        is_known = reconstruction_error < self.reconstruction_threshold

        if not is_known:
             #print(f"Unknown face detected. Reconstruction Error: {reconstruction_error:.4f} > Threshold: {self.reconstruction_threshold:.4f}")
            return "Unknown", unknown_confidence, False


        # 3. If likely known, proceed with KNN prediction
        try:
             # Check if PCA is fitted
             if not hasattr(self.pca, 'mean_'):
                 print("Error: PCA seems not fitted during prediction.")
                 return "Unknown", 0.0, False

             X_pca = self.pca.transform(X_scaled)

             # Check if KNN is fitted
             if not hasattr(self.knn, 'classes_'):
                 print("Error: KNN seems not fitted during prediction.")
                 return "Unknown", 0.0, False


             probabilities = self.knn.predict_proba(X_pca)[0]
             best_class_index = np.argmax(probabilities)
             confidence = probabilities[best_class_index]
             predicted_label = self.knn.classes_[best_class_index]

             # Map numerical label back to name
             if predicted_label < len(self.user_names):
                 predicted_name = self.user_names[predicted_label]
                else:
                  print(f"Warning: Predicted label {predicted_label} out of bounds for user names.")
                  predicted_name = "Error" # Or handle appropriately


             # Optional: Add a confidence threshold for KNN prediction itself
             knn_confidence_threshold = 0.5 # Example threshold
             if confidence < knn_confidence_threshold:
                  #print(f"KNN confidence {confidence:.2f} below threshold {knn_confidence_threshold}. Classifying as Unknown.")
                  return "Unknown", confidence, False # Return KNN confidence even if classified Unknown here

             # Combine confidences? Maybe just use KNN confidence if deemed known by PCA error.
             final_confidence = confidence

             return predicted_name, final_confidence, True

        except Exception as e:
             print(f"Error during KNN prediction: {e}")
             # print traceback for debugging
             import traceback
             traceback.print_exc()
             return "Unknown", 0.0, False


    def save_model(self, directory):
        """
        Saves the trained PCA, KNN, Scaler models, and user names list.

        Args:
            directory (str): The directory path to save the model files.
        """
        if not self.trained:
            print("Warning: Model is not trained. Nothing to save.")
            return

        os.makedirs(directory, exist_ok=True) # Ensure directory exists

        try:
            joblib.dump(self.scaler, os.path.join(directory, 'scaler.joblib'))
            joblib.dump(self.pca, os.path.join(directory, 'pca.joblib'))
            joblib.dump(self.knn, os.path.join(directory, 'knn.joblib'))
            joblib.dump(self.user_names, os.path.join(directory, 'user_names.joblib'))
            # Save the threshold used during training
            joblib.dump(self.reconstruction_threshold, os.path.join(directory, 'reconstruction_threshold.joblib'))
            print(f"Model components saved to directory: {directory}")
        except Exception as e:
            print(f"Error saving model components: {e}")

    def load_model(self, directory):
        """
        Loads the trained PCA, KNN, Scaler models, and user names list.

        Args:
            directory (str): The directory path from where to load the model files.
        """
        scaler_path = os.path.join(directory, 'scaler.joblib')
        pca_path = os.path.join(directory, 'pca.joblib')
        knn_path = os.path.join(directory, 'knn.joblib')
        names_path = os.path.join(directory, 'user_names.joblib')
        threshold_path = os.path.join(directory, 'reconstruction_threshold.joblib')


        # Check if all necessary files exist
        required_files = [scaler_path, pca_path, knn_path, names_path, threshold_path]
        if not all(os.path.exists(p) for p in required_files):
             print(f"Error: Cannot load model. Missing one or more files in {directory}")
             print(f"Expected: scaler.joblib, pca.joblib, knn.joblib, user_names.joblib, reconstruction_threshold.joblib")
             self.trained = False
             return False


        try:
            self.scaler = joblib.load(scaler_path)
            self.pca = joblib.load(pca_path)
            self.knn = joblib.load(knn_path)
            self.user_names = joblib.load(names_path)
            self.reconstruction_threshold = joblib.load(threshold_path)


            # Basic validation after loading
            if not all(hasattr(self, attr) for attr in ['scaler', 'pca', 'knn', 'user_names', 'reconstruction_threshold']):
                 raise ValueError("Loaded model is missing essential components.")
            if not hasattr(self.pca, 'mean_') or not hasattr(self.knn, 'classes_'):
                raise ValueError("Loaded PCA or KNN model appears incomplete or not fitted.")


            self.trained = True
             # Restore n_components and n_neighbors if needed for info, they are intrinsic to loaded models
            self.n_components = self.pca.n_components_ if hasattr(self.pca, 'n_components_') else self.pca.n_components
            self.n_neighbors = self.knn.n_neighbors

            print(f"Model components loaded successfully from: {directory}")
            print(f"  - Users: {self.user_names}")
            print(f"  - PCA Components: {self.n_components}")
            print(f"  - KNN Neighbors: {self.n_neighbors}")
            print(f"  - Reconstruction Threshold: {self.reconstruction_threshold}")
        return True 
        except Exception as e:
            print(f"Error loading model components: {e}")
            self.trained = False
            # Reset components to avoid partial loading issues
            self.__init__() # Re-initialize to default state
            return False


# Example usage placeholder (usually this would be in a separate script)
if __name__ == '__main__':
    # This block is for testing purposes if you run the script directly
    print("FaceRecognizer class defined. Instantiate and use train/predict methods.")

    # Example:
    # recognizer = FaceRecognizer(n_components=50, n_neighbors=3, reconstruction_error_threshold=0.02)
    # Load data (X_train, y_train)
    # recognizer.train(X_train, y_train)
    # recognizer.save_model('my_pca_knn_model')

    # Later...
    # recognizer = FaceRecognizer()
    # recognizer.load_model('my_pca_knn_model')
    # Load image (img)
    # name, confidence, is_known = recognizer.predict(img)
    # print(f"Predicted: {name}, Confidence: {confidence:.2f}, Known: {is_known}") 