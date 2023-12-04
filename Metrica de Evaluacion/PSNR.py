import numpy as np
import cv2
from math import log10, sqrt

def cargar_imagen(ruta):
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

def calcular_mse(imagen_original, imagen_comprimida):
    if imagen_original.shape != imagen_comprimida.shape:
        raise ValueError("Las dimensiones de las imágenes no coinciden.")
    # Calcular el Error Cuadrático Medio (MSE)
    mse = np.sum((imagen_original - imagen_comprimida)**2) / (imagen_original.shape[0] * imagen_original.shape[1])
    return mse

def calcular_psnr(imagen_original, imagen_comprimida, max_pixel_value=255):
    # Calcular el MSE
    mse = calcular_mse(imagen_original, imagen_comprimida)
    # Calcular el PSNR
    if mse == 0:
        return float('inf')
    psnr = 20 * log10(max_pixel_value / sqrt(mse))
    return psnr

ruta_imagen_original = "image-test2-1106 x 670.jpg"
ruta_imagen_comprimida = "img_mejorada.jpg"
imagen_original = cargar_imagen(ruta_imagen_original)
imagen_comprimida = cargar_imagen(ruta_imagen_comprimida)
valor_mse = calcular_mse(imagen_original, imagen_comprimida)
valor_psnr = calcular_psnr(imagen_original, imagen_comprimida)

print(f"El valor de MSE es: {valor_mse}")
print(f"El valor de PSNR es: {valor_psnr} dB")
