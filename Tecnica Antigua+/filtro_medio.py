import cv2
import numpy as np

def median_filter(img, kernel_size):
    # Copiar la imagen original para no modificarla directamente
    result = np.copy(img)
    rows, cols = img.shape

    # Calcular el radio del kernel
    radius = kernel_size // 2

    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            # Obtener el conjunto de valores dentro del kernel
            values = []
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    values.append(img[i + x, j + y])

            # Ordenar los valores
            values.sort()

            # Calcular la posición de la mediana
            median_position = len(values) // 2

            # Asignar el valor medio de la distribución ordenada al píxel en la posición actual
            result[i, j] = (values[median_position - 1])

    return result

# Cargar la imagen
img_path = 'image-test3.jpg'  # Reemplaza con la ruta de tu imagen
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Tamaño del kernel para el filtro de mediana
kernel_size = 3  # Puedes ajustar el tamaño del kernel según sea necesario

# Aplicar el filtro de mediana
result_img = median_filter(img, kernel_size)

# Guardar la imagen filtrada automáticamente
output_path = 'median_filtered_image.jpg'
cv2.imwrite(output_path, result_img)

# Crear ventanas con nombres
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Median Filtered', cv2.WINDOW_NORMAL)

# Mostrar las imágenes original y filtrada
cv2.imshow('Original', img)
cv2.imshow('Median Filtered', result_img)

# Ajustar el tamaño de las ventanas
cv2.resizeWindow('Original', 753, 603)
cv2.resizeWindow('Median Filtered', 753, 603)

cv2.waitKey(0)
cv2.destroyAllWindows()
