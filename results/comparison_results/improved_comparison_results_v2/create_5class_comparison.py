#!/usr/bin/env python3
"""
Create a compact 5-class comparison image from the best performing categories
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

def create_5class_comparison():
    """Create a compact 5-class comparison image"""
    
    # Select the 5 best performing categories based on quantum advantage
    selected_categories = ['frog', 'truck', 'cat', 'deer', 'horse']
    
    # Set up paths
    base_dir = Path('.')
    real_images_dir = base_dir / 'real_images_png'
    quantum_images_dir = base_dir / 'generated_images_png' / 'quantum'
    classical_images_dir = base_dir / 'generated_images_png' / 'classical'
    
    # Image parameters
    img_size = 64  # Smaller size for compact display
    num_images_per_class = 4  # Number of images per class per row
    
    # Create figure
    fig, axes = plt.subplots(3, len(selected_categories) * num_images_per_class, 
                            figsize=(len(selected_categories) * num_images_per_class * 1.2, 6))
    
    if len(selected_categories) == 1:
        axes = axes.reshape(3, -1)
    
    # Set random seed for consistent selection
    random.seed(42)
    
    for col_idx, category in enumerate(selected_categories):
        print(f"Processing category: {category}")
        
        # Get image paths
        real_category_dir = real_images_dir / category
        quantum_category_dir = quantum_images_dir / category
        classical_category_dir = classical_images_dir / category
        
        # Get available images
        real_images = sorted(list(real_category_dir.glob('*.png')))
        quantum_images = sorted(list(quantum_category_dir.glob('*.png')))
        classical_images = sorted(list(classical_category_dir.glob('*.png')))
        
        # Select random images for display
        selected_real = random.sample(real_images, min(num_images_per_class, len(real_images)))
        selected_quantum = random.sample(quantum_images, min(num_images_per_class, len(quantum_images)))
        selected_classical = random.sample(classical_images, min(num_images_per_class, len(classical_images)))
        
        # Display images
        for img_idx in range(num_images_per_class):
            col = col_idx * num_images_per_class + img_idx
            
            # Real images (top row)
            if img_idx < len(selected_real):
                real_img = cv2.imread(str(selected_real[img_idx]))
                real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                real_img = cv2.resize(real_img, (img_size, img_size))
                axes[0, col].imshow(real_img)
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            if img_idx == 0:  # Only label the first image of each category
                axes[0, col].set_ylabel(f'Real\n{category.upper()}', fontsize=10, fontweight='bold')
            
            # Quantum images (middle row)
            if img_idx < len(selected_quantum):
                quantum_img = cv2.imread(str(selected_quantum[img_idx]))
                quantum_img = cv2.cvtColor(quantum_img, cv2.COLOR_BGR2RGB)
                quantum_img = cv2.resize(quantum_img, (img_size, img_size))
                axes[1, col].imshow(quantum_img)
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])
            if img_idx == 0:  # Only label the first image of each category
                axes[1, col].set_ylabel('Quantum', fontsize=10, fontweight='bold')
            
            # Classical images (bottom row)
            if img_idx < len(selected_classical):
                classical_img = cv2.imread(str(selected_classical[img_idx]))
                classical_img = cv2.cvtColor(classical_img, cv2.COLOR_BGR2RGB)
                classical_img = cv2.resize(classical_img, (img_size, img_size))
                axes[2, col].imshow(classical_img)
            axes[2, col].set_xticks([])
            axes[2, col].set_yticks([])
            if img_idx == 0:  # Only label the first image of each category
                axes[2, col].set_ylabel('Classical', fontsize=10, fontweight='bold')
    
    # Add title
    fig.suptitle('Compact Comparison: Real vs Quantum vs Classical Generated Images\n(Top 5 Categories by Quantum Advantage)', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the image
    output_path = 'compact_5class_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Compact 5-class comparison saved to: {output_path}")
    
    # Also create a version with category labels
    create_labeled_version(selected_categories, num_images_per_class, img_size)
    
    plt.show()

def create_labeled_version(categories, num_images_per_class, img_size):
    """Create a version with clear category labels"""
    
    base_dir = Path('.')
    real_images_dir = base_dir / 'real_images_png'
    quantum_images_dir = base_dir / 'generated_images_png' / 'quantum'
    classical_images_dir = base_dir / 'generated_images_png' / 'classical'
    
    # Create figure with more space for labels
    fig, axes = plt.subplots(3, len(categories), figsize=(len(categories) * 2, 6))
    
    if len(categories) == 1:
        axes = axes.reshape(3, -1)
    
    random.seed(42)
    
    for col_idx, category in enumerate(categories):
        print(f"Creating labeled version for: {category}")
        
        # Get image paths
        real_category_dir = real_images_dir / category
        quantum_category_dir = quantum_images_dir / category
        classical_category_dir = classical_images_dir / category
        
        # Get available images
        real_images = sorted(list(real_category_dir.glob('*.png')))
        quantum_images = sorted(list(quantum_category_dir.glob('*.png')))
        classical_images = sorted(list(classical_category_dir.glob('*.png')))
        
        # Select one representative image per category
        selected_real = random.choice(real_images) if real_images else None
        selected_quantum = random.choice(quantum_images) if quantum_images else None
        selected_classical = random.choice(classical_images) if classical_images else None
        
        # Display images
        # Real images (top row)
        if selected_real:
            real_img = cv2.imread(str(selected_real))
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            real_img = cv2.resize(real_img, (img_size*2, img_size*2))
            axes[0, col_idx].imshow(real_img)
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        axes[0, col_idx].set_title(f'Real\n{category.upper()}', fontsize=12, fontweight='bold')
        
        # Quantum images (middle row)
        if selected_quantum:
            quantum_img = cv2.imread(str(selected_quantum))
            quantum_img = cv2.cvtColor(quantum_img, cv2.COLOR_BGR2RGB)
            quantum_img = cv2.resize(quantum_img, (img_size*2, img_size*2))
            axes[1, col_idx].imshow(quantum_img)
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
        axes[1, col_idx].set_title('Quantum', fontsize=12, fontweight='bold')
        
        # Classical images (bottom row)
        if selected_classical:
            classical_img = cv2.imread(str(selected_classical))
            classical_img = cv2.cvtColor(classical_img, cv2.COLOR_BGR2RGB)
            classical_img = cv2.resize(classical_img, (img_size*2, img_size*2))
            axes[2, col_idx].imshow(classical_img)
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])
        axes[2, col_idx].set_title('Classical', fontsize=12, fontweight='bold')
    
    # Add main title
    fig.suptitle('Compact Comparison: Real vs Quantum vs Classical Generated Images\n(Top 5 Categories by Quantum Advantage)', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the labeled version
    output_path = 'compact_5class_labeled_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Compact 5-class labeled comparison saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating compact 5-class comparison image...")
    print("Selected categories (by quantum advantage): frog, truck, cat, deer, horse")
    create_5class_comparison()
    print("Done!")
