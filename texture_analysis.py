import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis

def analyze_texture(image):
    """
    Perform texture analysis on the image.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        dict: Dictionary with texture metrics
        PIL.Image: Visualization of texture features
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Calculate texture features
    results = {}
    
    # 1. GLCM (Gray-Level Co-occurrence Matrix) features
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Normalize grayscale image to reduce quantization effects
    gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Quantize to 16 gray levels to make GLCM computation faster
    gray_quantized = (gray_normalized // 16).astype(np.uint8)
    
    # Calculate GLCM
    glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, 
                       levels=16, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    results['contrast'] = float(graycoprops(glcm, 'contrast').mean())
    results['dissimilarity'] = float(graycoprops(glcm, 'dissimilarity').mean())
    results['homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
    results['energy'] = float(graycoprops(glcm, 'energy').mean())
    results['correlation'] = float(graycoprops(glcm, 'correlation').mean())
    results['ASM'] = float(graycoprops(glcm, 'ASM').mean())  # Angular Second Moment
    
    # 2. First-order statistics
    results['mean'] = float(gray.mean())
    results['std'] = float(gray.std())
    results['entropy'] = float(shannon_entropy(gray))
    results['skewness'] = float(skew(gray.flatten()))
    results['kurtosis'] = float(kurtosis(gray.flatten()))
    
    # 3. Local Binary Pattern (LBP) - Using a simplified version to avoid overflow
    radius = 1  # Reduced radius to prevent overflow
    n_points = 8  # Fixed number of points to prevent too large values
    
    def local_binary_pattern(image, p, r):
        """Compute local binary pattern with overflow protection"""
        rows, cols = image.shape
        output = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(r, rows - r):
            for j in range(r, cols - r):
                # Get center pixel value
                center = image[i, j]
                binary_code = 0
                # Loop through neighbors
                for k in range(min(p, 8)):  # Limit to 8 bits to prevent overflow in uint8
                    angle = 2 * np.pi * k / p
                    x = i + r * np.cos(angle)
                    y = j + r * np.sin(angle)
                    # Interpolate pixel value if coordinates are not integers
                    x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
                    
                    # Boundary checks
                    if x_floor < 0 or x_floor >= rows - 1 or y_floor < 0 or y_floor >= cols - 1:
                        neighbor = 0
                    else:
                        x_ceil = min(x_floor + 1, rows - 1)
                        y_ceil = min(y_floor + 1, cols - 1)
                        
                        fx = x - x_floor
                        fy = y - y_floor
                        neighbor = (1 - fx) * (1 - fy) * image[x_floor, y_floor] + \
                                  fx * (1 - fy) * image[x_ceil, y_floor] + \
                                  (1 - fx) * fy * image[x_floor, y_ceil] + \
                                  fx * fy * image[x_ceil, y_ceil]
                    
                    # Add bit to binary code if neighbor >= center (using values that won't overflow uint8)
                    if neighbor >= center and k < 8:  # Ensure we don't exceed 8 bits for uint8
                        binary_code += (1 << k)  # Use bit shifting instead of power to be more explicit
                
                output[i, j] = binary_code
        return output
    
    # Use try-except to handle potential errors
    try:
        lbp = local_binary_pattern(gray, n_points, radius)
    except Exception as e:
        print(f"Error computing LBP: {e}")
        # Provide a fallback
        lbp = np.zeros_like(gray)
    
    # Calculate LBP histogram
    hist, _ = np.histogram(lbp.flatten(), bins=np.arange(0, 2**n_points + 1))
    hist = hist.astype(float)
    hist /= np.sum(hist)
    
    # Calculate basic statistics of LBP histogram
    results['lbp_mean'] = float(np.mean(hist))
    results['lbp_std'] = float(np.std(hist))
    results['lbp_entropy'] = float(shannon_entropy(hist))
    
    # 4. Edge features
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calculate edge metrics
    results['edge_mean'] = float(magnitude.mean())
    results['edge_std'] = float(magnitude.std())
    results['edge_density'] = float(np.sum(magnitude > 30) / magnitude.size)
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original grayscale image
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title('Grayscale Image')
    axs[0, 0].axis('off')
    
    # LBP visualization
    axs[0, 1].imshow(lbp, cmap='jet')
    axs[0, 1].set_title('Local Binary Pattern')
    axs[0, 1].axis('off')
    
    # Edge magnitude
    axs[1, 0].imshow(magnitude, cmap='hot')
    axs[1, 0].set_title('Edge Magnitude')
    axs[1, 0].axis('off')
    
    # GLCM correlation
    glcm_vis = graycoprops(glcm, 'correlation')
    axs[1, 1].bar(range(len(glcm_vis.flatten())), glcm_vis.flatten())
    axs[1, 1].set_title('GLCM Correlation for Different Directions')
    axs[1, 1].set_xlabel('Distance-Angle Combination')
    axs[1, 1].set_ylabel('Correlation')
    
    plt.tight_layout()
    
    # Save figure to BytesIO object
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    # Convert back to PIL image
    visualization = Image.open(buf)
    
    return results, visualization

def analyze_skin_irregularities(image):
    """
    Analyze skin irregularities in the image.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        dict: Dictionary with irregularity metrics
        PIL.Image: Visualization of detected irregularities
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to BGR for OpenCV processing
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Apply bilateral filter to reduce noise while preserving edges
    img_filtered = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to ignore noise
    min_area = img_array.shape[0] * img_array.shape[1] * 0.0001  # 0.01% of image size
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create visualization image
    visualization = img_cv.copy()
    cv2.drawContours(visualization, significant_contours, -1, (0, 255, 0), 2)
    
    # Calculate metrics
    irregularity_metrics = {}
    
    # Number of detected irregularities
    irregularity_metrics['irregularity_count'] = len(significant_contours)
    
    # Total area of irregularities
    total_area = sum(cv2.contourArea(cnt) for cnt in significant_contours)
    image_area = img_array.shape[0] * img_array.shape[1]
    irregularity_metrics['irregularity_area_percentage'] = (total_area / image_area) * 100
    
    # Average circularity (4π*area/perimeter²) - 1.0 for perfect circle
    if significant_contours:
        circularities = []
        for cnt in significant_contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circularities.append(circularity)
        
        irregularity_metrics['avg_circularity'] = np.mean(circularities) if circularities else 0
        irregularity_metrics['min_circularity'] = np.min(circularities) if circularities else 0
        irregularity_metrics['max_circularity'] = np.max(circularities) if circularities else 0
    else:
        irregularity_metrics['avg_circularity'] = 0
        irregularity_metrics['min_circularity'] = 0
        irregularity_metrics['max_circularity'] = 0
    
    # Convert visualization back to RGB for PIL
    visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL image
    visualization_pil = Image.fromarray(visualization_rgb)
    
    return irregularity_metrics, visualization_pil
