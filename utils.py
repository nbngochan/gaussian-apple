import numpy as np
import cv2


def gaussian(shape, sigma=10):
    """
    Gaussian kernel for density map generation
    """
    m = (shape[0] - 1) / 2.  # center point m, n
    n = (shape[1] - 1) / 2.  
    y, x = np.ogrid[-m:m+1, -n:n+1]  # mesh of coordinates
    k = np.exp(-(x*x + y*y) / (2*sigma*sigma))  # gaussian kernel at each point
    k = k / k.max()  # normalize
    k[k<0.0001] = 0  
    return k


def tricube(size, factor=3):
    """
    Tricube kernel for density map generation
    """
    s = (size - 1) / 2
    m, n = [(ss - 1.) / 2. for ss in [size, size]]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    k_x = (1 - (abs((x/s)**3)))**factor
    k_y = (1 - (abs((y/s)**3)))**factor
    
    k = k_y.dot(k_x)   
    
    k = k / k.max()  
    return k


def quartic(shape):
    m, n = [(s - 1) / 2. for s in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    k = 1 - 4*(x**2 + y**2) + 6*(x**3 + y**3) - 4*(x**4 + y**4)
    k = k / k.max()
    k[k<0.0001] = 0
    
    return k


def triweight(size, factor=3):
    s = (size - 1) / 2
    m, n = [(s - 1.) / 2. for s in [size, size]]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    k_x = ((1 - (abs(x/s)**2))**factor)(35 - 30*((abs(x/s))**2) + 3*abs((x/s))**4)
    k_y = ((1 - (abs(y/s)**2))**factor)(35 - 30*((abs(y/s))**2) + 3*abs((y/s))**4)
    
    k = k_y.dot(k_x)
    k = k / k.max()
    
    return k


def colormap(density_map, image):
    heatmap = cv2.applyColorMap((density_map * 255).astype('uint8'), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = cv2.addWeighted(image[:, :, ::-1], 0.5, heatmap, 0.5, 0)
    return overlay