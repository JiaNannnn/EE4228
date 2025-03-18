"""
Visualization Utilities for Face Preprocessing

This module provides utilities for:
1. Visualizing intermediate preprocessing steps
2. Creating success rate visualizations
3. Generating reports of preprocessing results
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure module logger
logger = logging.getLogger(__name__)

def create_preprocessing_visualization(subject_name, intermediates, output_path=None):
    """
    Create visualization of intermediate preprocessing steps
    
    Args:
        subject_name: Name of the subject (for title)
        intermediates: Dictionary of intermediate results from preprocessor
        output_path: Path to save the visualization image (optional)
    
    Returns:
        Matplotlib figure object (if output_path is None)
    """
    # Create visualization image
    fig = plt.figure(figsize=(12, 8))
    
    # Determine grid size
    n_plots = len(intermediates)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    # Plot each intermediate result
    for i, (title, img) in enumerate(intermediates.items()):
        plt.subplot(rows, cols, i+1)
        
        # Handle different image types
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        
        plt.title(title)
        plt.axis('off')
    
    plt.suptitle(f"Face Preprocessing Steps - Subject: {subject_name}", fontsize=16)
    plt.tight_layout()
    
    # Save visualization if requested
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved visualization to {output_path}")
        return None
    
    return fig

def create_success_rate_visualization(subject_stats, output_path=None):
    """
    Create visualization of face detection success rate by subject
    
    Args:
        subject_stats: Dictionary with subject statistics
        output_path: Path to save the visualization image (optional)
    
    Returns:
        Matplotlib figure object (if output_path is None)
    """
    subjects = list(subject_stats.keys())
    success_rates = [subject_stats[subject]['success_rate'] for subject in subjects]
    
    fig = plt.figure(figsize=(12, 6))
    bars = plt.bar(subjects, success_rates, color='steelblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{height:.1f}%", ha='center', va='bottom')
    
    plt.title('Face Detection Success Rate by Subject', fontsize=16)
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 105)  # Leave room for text above bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save visualization if requested
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Saved success rate visualization to {output_path}")
        return None
    
    return fig

def generate_preprocessing_report(subject_stats, total_images, total_success, 
                                 total_time, output_path=None):
    """
    Generate a report of preprocessing results
    
    Args:
        subject_stats: Dictionary with subject statistics
        total_images: Total number of images processed
        total_success: Total number of successful face detections
        total_time: Total processing time in seconds
        output_path: Path to save the report (optional)
    
    Returns:
        Report text (if output_path is None)
    """
    # Calculate overall statistics
    success_rate = (total_success / total_images) * 100 if total_images > 0 else 0
    avg_time = total_time / total_images if total_images > 0 else 0
    
    # Create report
    report_lines = [
        "Face Preprocessing Report",
        "======================",
        f"Total images processed: {total_images}",
        f"Successful face detections: {total_success} ({success_rate:.2f}%)",
        f"Failed face detections: {total_images - total_success} ({100 - success_rate:.2f}%)",
        f"Total processing time: {total_time:.2f} seconds",
        f"Average time per image: {avg_time:.2f} seconds",
        "",
        "Subject-wise Statistics:",
        "----------------------"
    ]
    
    # Add subject-wise statistics
    for subject, stats in subject_stats.items():
        report_lines.append(f"Subject: {subject}")
        report_lines.append(f"  Total images: {stats['total']}")
        report_lines.append(f"  Successful detections: {stats['success']} ({stats['success_rate']:.2f}%)")
        report_lines.append(f"  Failed detections: {stats['failed']} ({100 - stats['success_rate']:.2f}%)")
        report_lines.append("")
    
    report_text = '\n'.join(report_lines)
    
    # Print summary to log
    logger.info(f"Summary: Processed {total_images} images with {success_rate:.2f}% success rate")
    logger.info(f"Average processing time: {avg_time:.2f} seconds per image")
    
    # Write report to file if requested
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved preprocessing report to {output_path}")
        return None
    
    return report_text 