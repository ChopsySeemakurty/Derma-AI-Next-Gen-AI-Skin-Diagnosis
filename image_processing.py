import cv2
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# Function to preprocess image for model input
def preprocess_image(image):
    """
    Preprocess the image for model prediction.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Resize image to 224x224
    image_resized = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    return img_array

# Function to extract Region of Interest (ROI)
def extract_roi(image, roi_coords=None, method="manual"):
    """
    Extract Region of Interest from the image.
    
    Args:
        image (PIL.Image): Input image
        roi_coords (tuple): Coordinates of ROI (x1, y1, x2, y2)
        method (str): ROI extraction method ('manual' or 'auto')
        
    Returns:
        PIL.Image: Cropped ROI image
        tuple: ROI coordinates
    """
    # Convert PIL image to numpy array for OpenCV processing
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if method == "manual" and roi_coords is not None:
        # Manual ROI extraction
        x1, y1, x2, y2 = roi_coords
        roi = img_array[y1:y2, x1:x2]
        roi_pil = Image.fromarray(roi)
        return roi_pil
    
    elif method == "auto":
        # Automatic ROI detection using image processing techniques
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract ROI
            try:
                roi = img_array[y:y+h, x:x+w]
                if roi.size == 0:
                    return None, (0, 0, 0, 0)
                roi_pil = Image.fromarray(roi)
                # Return ROI and coordinates
                return roi_pil, (x, y, x+w, y+h)
            except Exception as e:
                print(f"Error extracting ROI: {str(e)}")
                return None, (0, 0, 0, 0)
        else:
            # If no contour is found, return the original image
            return image, (0, 0, image.width, image.height)
    
    else:
        # Return the original image if method is not recognized
        return image, (0, 0, image.width, image.height)

# Function to highlight areas of interest in the image
def highlight_areas(image, areas):
    """
    Highlight specific areas in the image.
    
    Args:
        image (PIL.Image): Input image
        areas (list): List of areas to highlight, each as (x1, y1, x2, y2)
        
    Returns:
        PIL.Image: Image with highlighted areas
    """
    # Create a copy of the image
    highlighted_img = image.copy()
    draw = ImageDraw.Draw(highlighted_img)
    
    # Draw rectangles around areas of interest
    for area in areas:
        x1, y1, x2, y2 = area
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
    
    return highlighted_img

# Function to resize image maintaining aspect ratio
def resize_image(image, max_size=800):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image (PIL.Image): Input image
        max_size (int): Maximum size for either dimension
        
    Returns:
        PIL.Image: Resized image
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize image
    resized_img = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

# Function to enhance image for better visibility
def enhance_image(image):
    """
    Enhance image for better visibility.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Enhanced image
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to BGR for OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convert back to RGB and then to PIL image
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    enhanced_pil = Image.fromarray(enhanced_rgb)
    
    return enhanced_pil
