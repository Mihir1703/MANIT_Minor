import glob
import cv2
import numpy as np
import math

# Define the file paths of the original and processed image datasets
original_dataset_path = '/home/mihir/Minor/Minor_project/raw_files/Images_1/Ads'
processed_dataset_path = '/home/mihir/Minor/Minor_project/processed_files/compressed'

# Load the images from the original and processed datasets
original_images = [cv2.imread(f) for f in glob.glob(original_dataset_path + '/*.jpg')]
processed_images = [cv2.imread(f) for f in glob.glob(processed_dataset_path + '/*.jpg.webp')]

# Calculate the entropy of the original and processed datasets
def calculate_entropy(images):
    entropy = 0
    for img in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram of pixel intensities
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalize histogram to get probability density function
        pdf = hist / np.sum(hist)
        
        # Calculate entropy using Shannon's formula
        entropy += -np.sum(pdf * np.log2(pdf + 1e-10))
        
    return entropy / len(images)

original_entropy = calculate_entropy(original_images)
processed_entropy = calculate_entropy(processed_images)

# Calculate the information gain
information_gain = original_entropy - processed_entropy

# Print the information gain as a percentage
print(f'Information gain: {information_gain / original_entropy * 100:.2f}%')
