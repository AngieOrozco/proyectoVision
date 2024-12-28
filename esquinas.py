import cv2
import numpy as np
import os

def shi_tomasi_corner_detector(image: np.array, max_corners: int, quality_level: float, min_distance: float, mask: np.array = None):
    '''
    Detecta esquinas utilizando el método de Shi-Tomasi y devuelve la cantidad de esquinas detectadas.

    Parameters:
    - image: Imagen de entrada.
    - max_corners: Número máximo de esquinas a detectar.
    - quality_level: Calidad mínima de las esquinas seleccionadas en relación a la mejor esquina.
    - min_distance: Distancia mínima entre las esquinas detectadas.
    - mask: Máscara binaria para limitar la detección de esquinas a una región específica de la imagen.

    Returns:
    - segmented_with_corners: Imagen segmentada con las esquinas detectadas resaltadas en color verde.
    - corner_count: Número de esquinas detectadas dentro de la máscara.
    '''
    # Crear una máscara completamente negra para la segmentación
    segmented_image = np.zeros_like(image)
    segmented_image[mask > 0] = image[mask > 0]

    # Aplicar filtro gaussiano para suavizar la imagen segmentada
    segmented_image = cv2.bilateralFilter(segmented_image, d=9, sigmaColor=75, sigmaSpace=75)
    segmented_image_smoothed = cv2.GaussianBlur(segmented_image, (9, 9), 2)

    # Convertir la imagen suavizada a escala de grises
    gray = cv2.cvtColor(segmented_image_smoothed, cv2.COLOR_BGR2GRAY)

    # Aplicar el detector de esquinas de Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(
        gray, 
        maxCorners=max_corners, 
        qualityLevel=quality_level, 
        minDistance=min_distance, 
        mask=mask,
        blockSize=3
    )

    # Verificar si se detectaron esquinas
    if corners is not None:
        corner_count = corners.shape[0]
        corners = np.int0(corners)

        # Resaltar las esquinas detectadas en verde sobre la imagen segmentada
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(segmented_image, (x, y), 3, (0, 255, 0), -1)
    else:
        corner_count = 0

    return segmented_image, corner_count

def create_red_mask(image):
    '''
    Crea una máscara para segmentar regiones rojas en una imagen.

    Parameters:
    - image: Imagen de entrada en formato BGR.

    Returns:
    - mask: Máscara binaria donde las regiones rojas tienen valor 255.
    '''
    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rangos de color rojo en el espacio HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Crear máscaras para ambos rangos de rojo
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combinar las máscaras
    mask = cv2.bitwise_or(mask1, mask2)

    # Devolver una máscara binaria en formato uint8
    return (mask > 0).astype(np.uint8) * 255

# Parámetros del detector de esquinas
max_corners = 20     # Número máximo de esquinas a detectar
quality_level = 0.2   # Nivel mínimo de calidad
min_distance = 100      # Distancia mínima entre esquinas

# Lista de rutas de imágenes
paths = ["fotos/frames/incorrect/frame_398.jpg", "fotos/frames/incorrect/frame_497.jpg", "fotos/frames/incorrect/frame_698.jpg", "fotos/frames/incorrect/frame_816.jpg"]
output_dir = "fotos/frames/incorrect"  # Directorio donde se guardarán las imágenes procesadas

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Procesar cada imagen en la lista de paths
for i, path in enumerate(paths):
    # Leer la imagen de entrada
    image = cv2.imread(path)

    if image is None:
        print(f"Error: No se pudo leer la imagen en {path}")
        continue

    # Crear una máscara para el color rojo
    red_mask = create_red_mask(image)

    # Detectar esquinas y contar el número de esquinas exteriores en el polígono rojo
    segmented_with_corners, corner_count = shi_tomasi_corner_detector(
        image, 
        max_corners=max_corners, 
        quality_level=quality_level, 
        min_distance=min_distance, 
        mask=red_mask
    )

    # Guardar la imagen segmentada con esquinas detectadas
    save_name = f"Segmented_with_corners_{i + 1}.jpg"
    cv2.imwrite(os.path.join(output_dir, save_name), segmented_with_corners)

    # Imprimir el número de esquinas detectadas
    print(f"Imagen {path}: {corner_count} esquinas exteriores detectadas.")