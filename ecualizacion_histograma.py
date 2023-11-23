import cv2
import numpy as np

def calculate_histogram(img):
    hist = np.zeros(256, dtype=int)

    # Calcular el histograma
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1

    return hist

def cumulative_distribution_function(hist):
    cdf = np.zeros(256, dtype=float)
    cdf[0] = hist[0]

    # Calcular la función de distribución acumulativa (CDF)
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    return cdf

def histogram_equalization(img):
    # Obtener el histograma de la imagen original
    hist = calculate_histogram(img)

    # Calcular la función de distribución acumulativa (CDF)
    cdf = cumulative_distribution_function(hist)

    # Normalizar la CDF para estar en el rango [0, 1]
    cdf_normalized = cdf / cdf[-1]

    # Mapear los valores de intensidad originales a los valores ecualizados
    equalized_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized_img[i, j] = int(cdf_normalized[img[i, j]] * 255)

    return equalized_img.astype(np.uint8)

# Cargar tu propia imagen
img_path = "filtered_image.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Aplicar la ecualización de histogramas
equalized_img = histogram_equalization(img)

# Mostrar las imágenes original y ecualizada
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Ecualizacion de Histogramas', cv2.WINDOW_NORMAL)

cv2.imshow('Original', img)
cv2.imshow('Ecualizacion de Histogramas', equalized_img)

# Guardar la imagen ecualizada
output_path = 'equalized_image.jpg'
cv2.imwrite(output_path, equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
