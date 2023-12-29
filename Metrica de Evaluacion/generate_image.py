import cv2
import os
import numpy as np

def calculate_histogram(img): 
    hist = np.zeros(256, dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1

    return hist

def cumulative_distribution_function(hist):
    cdf = np.zeros(256, dtype=float)
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    return cdf

def histogram_equalization(img, intensity_factor=1.0):
    hist = calculate_histogram(img)
    cdf = cumulative_distribution_function(hist)

    cdf_normalized = cdf / cdf[-1]

    equalized_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized_img[i, j] = int(cdf_normalized[img[i, j]] * 255 * intensity_factor)

    return equalized_img.astype(np.uint8)

def gaussian_filter(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)

def median_filter(img, kernel_size):
    result = np.copy(img)
    rows, cols = img.shape
    radius = kernel_size // 2

    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            values = []
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    values.append(img[i + x, j + y])

            values.sort()
            median_position = len(values) // 2
            result[i, j] = (values[median_position - 1])

    return result

input_folder = 'data-test'
output_folder = 'data-result'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(1, 51):  
    input_image_path = os.path.join(input_folder, f'j{i}.jpg')
    output_image_path = os.path.join(output_folder, f'j{i}_result.jpg')

    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    equalized_img = histogram_equalization(img)
    sigma = 1.5
    gaussian_img = gaussian_filter(equalized_img, sigma)
    kernel_size = 3 
    final_img = median_filter(gaussian_img, kernel_size)
    cv2.imwrite(output_image_path, final_img)

print("Proceso completado.")
