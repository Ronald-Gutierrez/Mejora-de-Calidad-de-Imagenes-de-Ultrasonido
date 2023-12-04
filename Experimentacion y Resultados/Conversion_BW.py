import cv2
imagen = cv2.imread('mejora_final_tecnicaModerna.jpg')

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

cv2.imwrite('imagen_blanco_y_negro.jpg', imagen_gris)
cv2.imshow('Imagen Blanco y Negro', imagen_gris)
cv2.waitKey(0)
cv2.destroyAllWindows()
