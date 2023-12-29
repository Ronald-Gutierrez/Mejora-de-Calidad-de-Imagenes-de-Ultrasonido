import os
import cv2
import numpy as np
from math import log10, sqrt

def cargar_imagen(ruta):
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

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

def evaluar_imagenes(data_test_folder, data_result_folder, output_file):
    psnr_list = []
    mse_list = []

    with open(output_file, 'w') as file:
        for i in range(1, 51):  
            ruta_imagen_original = os.path.join(data_test_folder, f'j{i}.jpg')
            ruta_imagen_comprimida = os.path.join(data_result_folder, f'j{i}_result.jpg')

            imagen_original = cargar_imagen(ruta_imagen_original)
            imagen_comprimida = cargar_imagen(ruta_imagen_comprimida)

            valor_mse = calcular_mse(imagen_original, imagen_comprimida)
            valor_psnr = calcular_psnr(imagen_original, imagen_comprimida)

            file.write(f"Para j{i}.jpg:\n")
            file.write(f"  - El valor de MSE es: {valor_mse}\n")
            file.write(f"  - El valor de PSNR es: {valor_psnr} dB\n\n")

            psnr_list.append(valor_psnr)
            mse_list.append(valor_mse)

        promedio_psnr = sum(psnr_list) / len(psnr_list)
        promedio_mse = sum(mse_list) / len(mse_list)

        file.write("Promedio General:\n")
        file.write(f"  - El valor de MSE promedio es: {promedio_mse}\n")
        file.write(f"  - El valor de PSNR promedio es: {promedio_psnr} dB\n")

data_test_folder = "data-test"
data_result_folder = "data-result"
output_file = "psnr_y_mse.txt"

evaluar_imagenes(data_test_folder, data_result_folder, output_file)

print("Los resultados estan en psnr_y_mse.txt")
