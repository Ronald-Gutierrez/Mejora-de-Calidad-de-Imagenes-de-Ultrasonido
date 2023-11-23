import numpy as np
import cv2

def gaussian_filter(img, sigma):
    rows, cols = img.shape
    result = np.zeros((rows, cols), dtype=np.float32)

    #Se determina el tamaño del kernel basado en la desviación estándar (sigma) y se crea una matriz de ceros para almacenar el kernel.
    kernel_size = int(6 * sigma) + 1
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # Construir la matriz de convolución
    # Se llena el kernel con valores calculados a partir de la función gaussiana
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - kernel_size // 2
            y = j - kernel_size // 2
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)  # Normalizar el kernel para que sume 1

    # Rellenar bordes de la imagen con ceros
    img_padded = np.pad(img, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant')

    # Aplicar la convolución
    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(img_padded[i:i+kernel_size, j:j+kernel_size] * kernel)

    return result.astype(np.uint8)  # Ajustar la escala de valores

# Cargar tu propia imagen
img_path = "median_filtered_image.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Parámetros del filtro gaussiano
sigma = 1.5

# Aplicar el filtro gaussiano
result_img = gaussian_filter(img, sigma)

# Guardar la imagen filtrada
output_path = 'filtered_image.jpg'
cv2.imwrite(output_path, result_img)

# Visualizar las imágenes
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Filtro Gaussiano', cv2.WINDOW_NORMAL)

cv2.imshow('Original', img)
cv2.imshow('Filtro Gaussiano', result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
