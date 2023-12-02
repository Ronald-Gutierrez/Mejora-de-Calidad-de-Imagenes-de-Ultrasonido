import cv2
import numpy as np

def median_filter(img, kernel_size):
    # Copiar la imagen original para no modificarla directamente
    result = np.copy(img)
    rows, cols = img.shape
    radius = kernel_size // 2

    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            # Obtener el conjunto de valores dentro del kernel
            values = []
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    values.append(img[i + x, j + y])

            values.sort()
            median_position = len(values) // 2

            # Asignar el valor medio de la distribución ordenada al píxel en la posición actual
            result[i, j] = (values[median_position - 1])

    return result

img_path = 'image-test2.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Tamaño del kernel para el filtro de mediana
kernel_size = 9  

result_img = median_filter(img, kernel_size)

output_path = 'img_filtro_medio.jpg'
cv2.imwrite(output_path, result_img)

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Median Filtered', cv2.WINDOW_NORMAL)

cv2.imshow('Original', img)
cv2.imshow('Median Filtered', result_img)

cv2.resizeWindow('Original', 753, 603)
cv2.resizeWindow('Median Filtered', 753, 603)

cv2.waitKey(0)
cv2.destroyAllWindows()
