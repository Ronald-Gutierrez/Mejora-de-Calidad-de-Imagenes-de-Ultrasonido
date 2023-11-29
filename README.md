# Proyecto de Mejora de Calidad de Imágenes de Ultrasonido

## Introducción

En un contexto médico en constante evolución, las imágenes de ultrasonido desempeñan un papel crucial en el diagnóstico y tratamiento de diversas condiciones de salud. Sin embargo, las limitaciones en la calidad de estas imágenes, como la falta de contraste, nitidez insuficiente y presencia persistente de ruido, pueden obstaculizar la interpretación precisa por parte de los profesionales de la salud. Este proyecto se enfoca en abordar estos desafíos mediante técnicas avanzadas de procesamiento de imágenes y el uso de modelos de *Deep Learning*.

## Objetivo

El objetivo principal es mejorar la calidad y resolución de las imágenes de ultrasonido médico para facilitar diagnósticos más precisos. Se explorarán técnicas de procesamiento de imágenes, como el filtrado mediano, el filtrado gaussiano y la ecualización de histogramas. Además, se implementará el modelo Real-ESRGAN, una red neuronal avanzada diseñada para realizar super resolución en imágenes.

## Metodología

### Técnicas de Procesamiento de Imágenes

#### Filtro Mediano

El filtro mediano se utilizará para reducir el ruido en las imágenes de ultrasonido. Este filtro estadístico de orden tomará la mediana de un conjunto de valores en una ventana deslizante, lo que ayudará a mitigar valores atípicos o ruido impulsivo.

#### Filtro Gaussiano

El filtrado gaussiano se aplicará para suavizar la imagen, reduciendo el ruido y preservando características de baja frecuencia. La intensidad de difuminado estará controlada por la desviación estándar, ajustándola para lograr el nivel de suavizado deseado.

#### Ecualización de Histogramas

Esta técnica se empleará para mejorar el contraste de las imágenes. Al redistribuir los valores de intensidad, se maximizará el contraste sin perder información crucial. Esto es especialmente útil para mejorar la visualización de detalles en las imágenes de ultrasonido.

### Modelo Real-ESRGAN

El modelo Real-ESRGAN, una red neuronal profunda especializada en super resolución, se implementará para mejorar la resolución de las imágenes de ultrasonido. Su arquitectura avanzada, compuesta por capas convolucionales y bloques residuales, permite realizar super resolución con diferentes factores de escala.

---

*Este proyecto se basa en la integración de técnicas propuestas por diversos autores, para el procesamiento de imágenes con técnicas tradicionales de procesamiento de imagenes y el uso del modelo Real-ESRGAN.*
