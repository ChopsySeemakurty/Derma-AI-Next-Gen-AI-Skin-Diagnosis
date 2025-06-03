import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

def analyze_color_profile(image, n_clusters=5):
    """
    Analyze the color profile of the image using K-means clustering.
    
    Args:
        image (PIL.Image): Input image
        n_clusters (int): Number of color clusters to identify
        
    Returns:
        dict: Dictionary mapping color names to percentages
        PIL.Image: Visualization of the color segmentation
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Get labels for all points
    labels = kmeans.labels_
    
    # Count occurrences of each label
    label_counts = np.bincount(labels)
    
    # Calculate percentages
    percentages = label_counts / len(labels) * 100
    
    # Create a dictionary mapping color names to percentages
    color_dict = {}
    for i, color in enumerate(colors):
        # Convert RGB to hex
        hex_color = mcolors.to_hex(color / 255.0)
        
        # Determine color name (simplified)
        r, g, b = color
        if r > 200 and g > 200 and b > 200:
            color_name = "White"
        elif r < 50 and g < 50 and b < 50:
            color_name = "Black"
        elif r > max(g, b) + 50:
            color_name = "Red"
        elif g > max(r, b) + 50:
            color_name = "Green"
        elif b > max(r, g) + 50:
            color_name = "Blue"
        elif r > 200 and g > 150 and b < 100:
            color_name = "Yellow"
        elif r > 150 and g < 100 and b < 100:
            color_name = "Brown"
        elif r > 150 and g > 100 and b > 100 and abs(r - g) < 50 and abs(r - b) < 50:
            color_name = "Gray"
        else:
            color_name = f"Color {i+1}"
        
        color_dict[color_name] = percentages[i]
    
    # Create segmented image for visualization
    segmented_img = np.zeros_like(img_array)
    for i in range(n_clusters):
        segmented_img[labels.reshape(img_array.shape[0], img_array.shape[1]) == i] = colors[i]
    
    # Convert back to PIL image
    segmented_pil = Image.fromarray(segmented_img)
    
    return color_dict, segmented_pil

def get_color_distribution_chart(color_dict):
    """
    Create a pie chart of color distribution.
    
    Args:
        color_dict (dict): Dictionary mapping color names to percentages
        
    Returns:
        matplotlib.figure.Figure: Pie chart figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get colors and percentages
    colors = list(color_dict.keys())
    percentages = list(color_dict.values())
    
    # Create color map for the pie chart
    color_map = []
    for color in colors:
        if color == "Red":
            color_map.append("red")
        elif color == "Green":
            color_map.append("green")
        elif color == "Blue":
            color_map.append("blue")
        elif color == "Yellow":
            color_map.append("yellow")
        elif color == "Brown":
            color_map.append("brown")
        elif color == "White":
            color_map.append("white")
        elif color == "Black":
            color_map.append("black")
        elif color == "Gray":
            color_map.append("gray")
        else:
            color_map.append("purple")
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        percentages, 
        labels=colors, 
        colors=color_map,
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=90
    )
    
    # Customize text properties
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    ax.set_title('Color Distribution', fontsize=16)
    
    return fig

def analyze_skin_tone(image):
    """
    Analyze the skin tone of the image.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        dict: Dictionary with skin tone information
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to BGR for OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to YCrCb color space (better for skin detection)
    img_ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color bounds in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create binary mask for skin regions
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to get only skin pixels
    skin_pixels = cv2.bitwise_and(img_cv, img_cv, mask=skin_mask)
    
    # Convert back to RGB
    skin_pixels_rgb = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2RGB)
    
    # Extract non-zero RGB values (skin pixels)
    non_zero_indices = np.where(skin_mask > 0)
    skin_pixels_flat = skin_pixels_rgb[non_zero_indices]
    
    # If no skin pixels detected
    if len(skin_pixels_flat) == 0:
        return {
            "skin_detected": False,
            "skin_tone": "Unknown",
            "average_rgb": (0, 0, 0),
            "skin_percentage": 0.0
        }
    
    # Calculate average RGB values of skin pixels
    avg_r = np.mean(skin_pixels_flat[:, 0])
    avg_g = np.mean(skin_pixels_flat[:, 1])
    avg_b = np.mean(skin_pixels_flat[:, 2])
    
    # Determine skin tone category (simplified)
    r, g, b = avg_r, avg_g, avg_b
    
    # Simple skin tone categorization
    if r > 200 and g > 170 and b > 150:
        skin_tone = "Very Light"
    elif r > 180 and g > 140 and b > 120:
        skin_tone = "Light"
    elif r > 160 and g > 120 and b > 100:
        skin_tone = "Medium"
    elif r > 140 and g > 100 and b > 80:
        skin_tone = "Olive"
    elif r > 120 and g > 80 and b > 60:
        skin_tone = "Tan"
    elif r > 100 and g > 60 and b > 40:
        skin_tone = "Deep"
    else:
        skin_tone = "Very Deep"
    
    # Calculate percentage of skin in the image
    skin_percentage = (np.count_nonzero(skin_mask) / skin_mask.size) * 100
    
    return {
        "skin_detected": True,
        "skin_tone": skin_tone,
        "average_rgb": (int(avg_r), int(avg_g), int(avg_b)),
        "skin_percentage": skin_percentage
    }

def create_skin_tone_visualization(image):
    """
    Create a visualization of detected skin regions.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Visualization with skin regions highlighted
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to BGR for OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to YCrCb color space
    img_ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color bounds in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create binary mask for skin regions
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply GrabCut for better segmentation
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Convert mask to format required by grabCut
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    mask[skin_mask > 0] = cv2.GC_PR_FGD  # Probable foreground
    mask[skin_mask == 0] = cv2.GC_BGD  # Background
    
    # Apply GrabCut
    cv2.grabCut(img_cv, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    
    # Create mask where sure or probable foreground
    mask2 = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 255, 0).astype('uint8')
    
    # Create colored mask for visualization
    colored_mask = np.zeros_like(img_cv)
    colored_mask[mask2 > 0] = [0, 255, 0]  # Green for skin
    
    # Blend original image with mask
    alpha = 0.5
    blended = cv2.addWeighted(img_cv, 1, colored_mask, alpha, 0)
    
    # Convert back to RGB for PIL
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL image
    visualization = Image.fromarray(blended_rgb)
    
    return visualization
