import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('image.jpg')  

# Get bounding box coordinates
x, y, w, h = 100, 200, 300, 400

# Create empty density map  
density_map = np.zeros(img.shape[:2])

# Calculate center of bounding box
cx = x + w//2
cy = y + h//2

# Create Gaussian kernel
kernel_size = (h, w)  
sigma = 50
kernel = cv2.getGaussianKernel(kernel_size[0], sigma) * cv2.getGaussianKernel(kernel_size[1], sigma).T 
kernel = kernel / kernel.max()

# Apply Gaussian kernel to density map 
density_map[cy-h//2:cy+h//2, cx-w//2:cx+w//2] = kernel

# Display density map
plt.imshow(density_map, cmap='hot') 
plt.show()