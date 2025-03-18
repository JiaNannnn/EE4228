"""
Visualize Preprocessing Steps

Script to visualize the different preprocessing steps on a sample image.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from face_preprocessor import FacePreprocessor
from face_detector import FaceDetector
from config import (
    ATT_DATASET_PATH, USE_CLAHE, USE_HIST_EQUALIZATION,
    USE_CONTRAST_STRETCHING, USE_GAMMA_CORRECTION, USE_FACE_ALIGNMENT
)

def draw_eye_positions(image, left_eye=None, right_eye=None):
    """
    Draw eye positions on an image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    left_eye : tuple
        Left eye position (x, y)
    right_eye : tuple
        Right eye position (x, y)
        
    Returns:
    --------
    numpy.ndarray
        Image with eye positions marked
    """
    result = image.copy()
    
    # If the image is grayscale, convert to color
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    # Draw left eye
    if left_eye is not None:
        cv2.circle(result, left_eye, 5, (0, 0, 255), -1)  # Red circle
        
    # Draw right eye
    if right_eye is not None:
        cv2.circle(result, right_eye, 5, (0, 255, 0), -1)  # Green circle
        
    # Draw line between eyes if both are detected
    if left_eye is not None and right_eye is not None:
        cv2.line(result, left_eye, right_eye, (255, 0, 0), 1)  # Blue line
        
    return result

def visualize_eye_detection(image_path):
    """
    Visualize eye detection on a face image
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image '{image_path}' not found")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image '{image_path}'")
        return
    
    # Detect faces
    detector = FaceDetector()
    faces = detector.detect_faces(image)
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return
    
    # Get the first face
    face_rect = faces[0]
    face_img = detector.get_face_roi(image, face_rect)
    
    # Detect eyes
    left_eye, right_eye = detector.detect_eyes(face_img)
    
    # Mark eye positions
    marked_face = draw_eye_positions(face_img, left_eye, right_eye)
    
    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert images to RGB (matplotlib uses RGB)
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    marked_face_rgb = cv2.cvtColor(marked_face, cv2.COLOR_BGR2RGB)
    
    # Plot original face
    axes[0].imshow(face_img_rgb)
    axes[0].set_title("Original Face")
    axes[0].axis("off")
    
    # Plot face with eye positions
    axes[1].imshow(marked_face_rgb)
    axes[1].set_title("Detected Eye Positions")
    axes[1].axis("off")
    
    # Save and show figure
    plt.tight_layout()
    plt.savefig("eye_detection.png", dpi=150)
    print("Eye detection visualization saved to 'eye_detection.png'")
    plt.show()

def visualize_preprocessing(image_path):
    """
    Visualize preprocessing steps on an image
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image '{image_path}' not found")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image '{image_path}'")
        return
    
    # Create preprocessor with all enhancements enabled
    preprocessor = FacePreprocessor(
        use_clahe=True,
        use_hist_equalization=True,
        use_contrast_stretching=True,
        use_gamma_correction=True,
        use_face_alignment=True  # Enable face alignment
    )
    
    # Get preprocessing steps
    steps = preprocessor.visualize_preprocessing_steps(image)
    
    # Prepare plot
    n_steps = len(steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 4, 4))
    
    # Plot each step
    for i, (step_name, step_image) in enumerate(steps.items()):
        ax = axes[i]
        
        # Convert to RGB for plotting if needed (matplotlib uses RGB)
        if len(step_image.shape) == 2:
            # Convert grayscale to RGB for display
            step_image_rgb = cv2.cvtColor(step_image, cv2.COLOR_GRAY2RGB)
        elif step_image.shape[2] == 3:
            # Convert BGR to RGB for display
            step_image_rgb = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
        else:
            step_image_rgb = step_image
            
        # Plot the image
        ax.imshow(step_image_rgb)
        ax.set_title(step_name.replace('_', ' ').title())
        ax.axis('off')
        
        # Add histogram as inset if grayscale
        if len(step_image.shape) == 2:
            # Create inset axes for histogram
            hist_ax = ax.inset_axes([0.1, 0.1, 0.8, 0.2])
            
            # Calculate histogram
            hist = cv2.calcHist([step_image], [0], None, [256], [0, 256])
            
            # Plot histogram
            hist_ax.plot(hist)
            hist_ax.set_xlim([0, 256])
            hist_ax.set_xticks([])
            hist_ax.set_yticks([])
            hist_ax.set_title('Histogram', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('preprocessing_steps.png', dpi=150)
    print(f"Preprocessing visualization saved to 'preprocessing_steps.png'")
    
    # Show figure
    plt.show()

def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize preprocessing steps')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to input image')
    parser.add_argument('--show-eyes', action='store_true',
                        help='Visualize eye detection')
    args = parser.parse_args()
    
    # If image path is provided, use it
    if args.image:
        image_path = args.image
    else:
        # Try to find a sample image from the dataset
        if os.path.exists(ATT_DATASET_PATH):
            # Look for the first image in the dataset
            for subject_id in range(1, 41):
                subject_dir = os.path.join(ATT_DATASET_PATH, f"s{subject_id}")
                if os.path.exists(subject_dir):
                    for image_id in range(1, 11):
                        image_path = os.path.join(subject_dir, f"{image_id}.pgm")
                        if os.path.exists(image_path):
                            print(f"Using sample image: {image_path}")
                            break
                    if 'image_path' in locals():
                        break
        
        # If no image found, use the default
        if 'image_path' not in locals():
            print("No sample image found. Please specify an image path with --image")
            return
    
    # If show eyes flag is set, visualize eye detection
    if args.show_eyes:
        visualize_eye_detection(image_path)
    else:
        # Visualize preprocessing steps
        visualize_preprocessing(image_path)

if __name__ == '__main__':
    main() 