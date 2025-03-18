import os
import json
import numpy as np
import joblib
from datetime import datetime
import cv2

class UserManager:
    """Class for managing user registration and face embeddings"""
    
    def __init__(self, gallery_dir='gallery', embeddings_path='models/user_embeddings.joblib',
                 user_info_path='data/registered_users.json'):
        """
        Initialize user manager
        
        Parameters:
        -----------
        gallery_dir : str
            Directory containing user face images
        embeddings_path : str
            Path to save/load user embeddings
        user_info_path : str
            Path to save/load user information
        """
        self.gallery_dir = gallery_dir
        self.embeddings_path = embeddings_path
        self.user_info_path = user_info_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        os.makedirs(os.path.dirname(user_info_path), exist_ok=True)
        os.makedirs(gallery_dir, exist_ok=True)
        
        # Load existing data
        self.user_embeddings = self._load_embeddings()
        self.user_info = self._load_user_info()
    
    def _load_embeddings(self):
        """
        Load user embeddings from file
        
        Returns:
        --------
        user_embeddings : dict
            Dictionary mapping user names to face embeddings
        """
        if os.path.exists(self.embeddings_path):
            try:
                return joblib.load(self.embeddings_path)
            except Exception as e:
                print(f"Error loading user embeddings: {str(e)}")
                return {}
        else:
            return {}
    
    def save_embeddings(self):
        """Save user embeddings to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            
            joblib.dump(self.user_embeddings, self.embeddings_path)
            print(f"User embeddings saved to {self.embeddings_path}")
        except Exception as e:
            print(f"Error saving user embeddings: {str(e)}")
    
    def _load_user_info(self):
        """
        Load user information from JSON file
        
        Returns:
        --------
        user_info : dict
            Dictionary containing user information
        """
        if os.path.exists(self.user_info_path):
            try:
                with open(self.user_info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user information: {str(e)}")
                return {"users": {}}
        else:
            return {"users": {}}
    
    def save_user_info(self):
        """Save user information to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.user_info_path), exist_ok=True)
            
            with open(self.user_info_path, 'w') as f:
                json.dump(self.user_info, f, indent=4)
            print(f"User information saved to {self.user_info_path}")
        except Exception as e:
            print(f"Error saving user information: {str(e)}")
    
    def register_user(self, username, full_name=None, metadata=None):
        """
        Register a new user
        
        Parameters:
        -----------
        username : str
            Username (must be unique)
        full_name : str, optional
            Full name of the user
        metadata : dict, optional
            Additional user metadata
            
        Returns:
        --------
        success : bool
            Whether the registration was successful
        """
        # Check if user already exists
        if username in self.user_info["users"]:
            print(f"User '{username}' already exists")
            return False
        
        # Create user entry
        if full_name is None:
            full_name = username
            
        if metadata is None:
            metadata = {}
        
        # Add user to user_info
        self.user_info["users"][username] = {
            "full_name": full_name,
            "registered_at": datetime.now().isoformat(),
            "trained": False,  # Will be set to True after face is trained
            "metadata": metadata
        }
        
        # Create user gallery directory
        user_gallery_dir = os.path.join(self.gallery_dir, username)
        os.makedirs(user_gallery_dir, exist_ok=True)
        
        # Save user information
        self.save_user_info()
        
        print(f"User '{username}' registered successfully")
        return True
    
    def delete_user(self, username):
        """
        Delete a user
        
        Parameters:
        -----------
        username : str
            Username to delete
            
        Returns:
        --------
        success : bool
            Whether the deletion was successful
        """
        # Check if user exists
        if username not in self.user_info["users"]:
            print(f"User '{username}' does not exist")
            return False
        
        # Remove user from embeddings
        if username in self.user_embeddings:
            del self.user_embeddings[username]
            self.save_embeddings()
        
        # Remove user from user_info
        del self.user_info["users"][username]
        self.save_user_info()
        
        print(f"User '{username}' deleted successfully")
        return True
    
    def is_user_registered(self, username):
        """
        Check if user is registered
        
        Parameters:
        -----------
        username : str
            Username to check
            
        Returns:
        --------
        is_registered : bool
            Whether the user is registered
        """
        return username in self.user_info["users"]
    
    def is_user_trained(self, username):
        """
        Check if user's face is trained
        
        Parameters:
        -----------
        username : str
            Username to check
            
        Returns:
        --------
        is_trained : bool
            Whether the user's face is trained
        """
        if not self.is_user_registered(username):
            return False
        
        return self.user_info["users"][username].get("trained", False)
    
    def set_user_trained(self, username, trained=True):
        """
        Set user's trained status
        
        Parameters:
        -----------
        username : str
            Username to update
        trained : bool
            Whether the user's face is trained
            
        Returns:
        --------
        success : bool
            Whether the update was successful
        """
        if not self.is_user_registered(username):
            print(f"User '{username}' is not registered")
            return False
        
        self.user_info["users"][username]["trained"] = trained
        self.save_user_info()
        
        print(f"User '{username}' trained status set to {trained}")
        return True
    
    def get_user_data(self, username):
        """
        Get user data
        
        Parameters:
        -----------
        username : str
            Username to retrieve
            
        Returns:
        --------
        user_data : dict or None
            User data dictionary, or None if user does not exist
        """
        if not self.is_user_registered(username):
            return None
        
        return self.user_info["users"][username]
    
    def get_all_users(self):
        """
        Get all registered users
        
        Returns:
        --------
        users : dict
            Dictionary of all registered users
        """
        return self.user_info["users"]
    
    def get_trained_users(self):
        """
        Get all trained users
        
        Returns:
        --------
        trained_users : dict
            Dictionary of trained users
        """
        return {username: data for username, data in self.user_info["users"].items() 
                if data.get("trained", False)}
    
    def get_untrained_users(self):
        """
        Get all untrained users
        
        Returns:
        --------
        untrained_users : dict
            Dictionary of untrained users
        """
        return {username: data for username, data in self.user_info["users"].items() 
                if not data.get("trained", False)}
    
    def store_user_embedding(self, username, embedding):
        """
        Store user's face embedding
        
        Parameters:
        -----------
        username : str
            Username to store embedding for
        embedding : numpy.ndarray
            Face embedding vector
            
        Returns:
        --------
        success : bool
            Whether the storage was successful
        """
        if not self.is_user_registered(username):
            print(f"User '{username}' is not registered")
            return False
        
        # Store embedding
        if isinstance(embedding, list):
            embedding = np.array(embedding)
            
        self.user_embeddings[username] = embedding
        self.save_embeddings()
        
        # Mark user as trained
        self.set_user_trained(username, True)
        
        print(f"Embedding stored for user '{username}'")
        return True
    
    def get_user_embedding(self, username):
        """
        Get user's face embedding
        
        Parameters:
        -----------
        username : str
            Username to retrieve embedding for
            
        Returns:
        --------
        embedding : numpy.ndarray or None
            Face embedding vector, or None if not available
        """
        if username not in self.user_embeddings:
            return None
        
        return self.user_embeddings[username]
    
    def get_all_embeddings(self):
        """
        Get all user embeddings
        
        Returns:
        --------
        embeddings : dict
            Dictionary mapping usernames to face embeddings
        """
        return self.user_embeddings
    
    def get_gallery_path(self, username):
        """
        Get path to user's gallery directory
        
        Parameters:
        -----------
        username : str
            Username to get gallery for
            
        Returns:
        --------
        gallery_path : str
            Path to user's gallery directory
        """
        return os.path.join(self.gallery_dir, username)
    
    def count_gallery_images(self, username):
        """
        Count number of face images in user's gallery
        
        Parameters:
        -----------
        username : str
            Username to count images for
            
        Returns:
        --------
        image_count : int
            Number of face images in gallery
        """
        gallery_path = self.get_gallery_path(username)
        
        if not os.path.exists(gallery_path):
            return 0
        
        image_files = [f for f in os.listdir(gallery_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        return len(image_files)
    
    def load_gallery_images(self, username, preprocessor=None):
        """
        Load face images from user's gallery
        
        Parameters:
        -----------
        username : str
            Username to load images for
        preprocessor : FacePreprocessor, optional
            Preprocessor to apply to images
            
        Returns:
        --------
        face_images : numpy.ndarray
            Array of face images
        """
        gallery_path = self.get_gallery_path(username)
        
        if not os.path.exists(gallery_path):
            return np.array([])
        
        image_files = [f for f in os.listdir(gallery_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        face_images = []
        
        for image_file in image_files:
            image_path = os.path.join(gallery_path, image_file)
            
            try:
                img = cv2.imread(image_path)
                
                if img is None:
                    continue
                
                # Apply preprocessing if provided
                if preprocessor is not None:
                    img = preprocessor.preprocess(img)
                
                face_images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
        
        return np.array(face_images)

def get_user_summary():
    """
    Get a summary of registered and trained users
    
    Returns:
    --------
    summary : dict
        Dictionary containing user summary
    """
    user_manager = UserManager()
    
    all_users = user_manager.get_all_users()
    trained_users = user_manager.get_trained_users()
    untrained_users = user_manager.get_untrained_users()
    
    gallery_counts = {}
    for username in all_users:
        gallery_counts[username] = user_manager.count_gallery_images(username)
    
    summary = {
        "registered_users": len(all_users),
        "trained_users": len(trained_users),
        "untrained_users": len(untrained_users),
        "gallery_counts": gallery_counts,
        "users": all_users
    }
    
    return summary

if __name__ == "__main__":
    print("This module provides user registration and management functions.")
    print("Import this module and use its classes and functions in your application.") 