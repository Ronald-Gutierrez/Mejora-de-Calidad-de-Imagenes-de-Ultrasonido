import numpy as np
import cv2
from math import log10, sqrt

def cargar_imagen(ruta, target_shape=None):
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if target_shape is not None:
        imagen = cv2.resize(imagen, target_shape[::-1])  # Invierte las dimensiones para (ancho, alto)
    return imagen

def calcular_mse(imagen_original, imagen_comprimida):
    if imagen_original.shape != imagen_comprimida.shape:
        raise ValueError("Las dimensiones de las imágenes no coinciden.")
    mse = np.sum((imagen_original - imagen_comprimida)**2) / (imagen_original.shape[0] * imagen_original.shape[1])
    return mse

def calcular_psnr(imagen_original, imagen_comprimida, max_pixel_value=255):
    mse = calcular_mse(imagen_original, imagen_comprimida)
    if mse == 0:
        return float('inf')
    psnr = 20 * log10(max_pixel_value / sqrt(mse))
    return psnr

ruta_imagen_original = "img_original.jpg"
ruta_imagen_comprimida = "img_mejorada.jpg"
imagen_original = cargar_imagen(ruta_imagen_original)
imagen_comprimida = cargar_imagen(ruta_imagen_comprimida, target_shape=imagen_original.shape)
valor_mse = calcular_mse(imagen_original, imagen_comprimida)
valor_psnr = calcular_psnr(imagen_original, imagen_comprimida)
print(f"El valor de MSE es: {valor_mse}")
print(f"El valor de PSNR es: {valor_psnr} dB")
