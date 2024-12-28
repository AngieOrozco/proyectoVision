# -- coding: utf-8 --

import cv2
import os
import numpy as np

def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """
    # Open the video file using VideoCapture
    cap = cv2.VideoCapture(videopath)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print('Error: Could not open the video file')
        return None, None, None, None

    # Get the size of frames (width and height) and the frame rate
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the video frames
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the video frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    # List to store the frames
    frames = []

    # Read the frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Release the video capture object
    cap.release()

    return frames, frame_width, frame_height, frame_rate

# Solicitar descripción del sospechoso
color_sospechoso = input("Introduce el color del sospechoso (por ejemplo, azul): ").lower()

# Definir rangos de color en HSV según el color ingresado
if color_sospechoso == "azul":
    lower_color = np.array([100, 150, 50])
    upper_color = np.array([140, 255, 255])
elif color_sospechoso == "amarillo":
    lower_color = np.array([30, 100, 140])
    upper_color = np.array([36, 150, 180])
elif color_sospechoso == "rojo":
    lower_color = np.array([153, 213, 116]) - np.array([20, 20, 20])
    upper_color = np.array([162, 222, 125]) + np.array([20, 20, 20])
else:
    print(f"El color '{color_sospechoso}' no está predefinido. No se va a poder detectar al sospechoso.")
    lower_color, upper_color = None, None

# Path to the video file (visiontraffic.avi)
videopath = 'colors/gris.mp4'

# Read the video and obtain the frames and properties
frames, frame_width, frame_height, frame_rate = read_video(videopath)

if not frames:
    print("Failed to load video.")
    exit()

# Crear carpeta para guardar el video de salida
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "gris.avi")

# Configurar el VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Inicializar el sustractor de fondo MOG2
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

for frame in frames:
    # Aplicar sustracción de fondo
    mask = mog2.apply(frame)

    # Detectar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movimiento_detectado = False
    sospechoso_detectado = False

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar objetos pequeños
            movimiento_detectado = True

            # Dibujar el rectángulo alrededor del objeto detectado
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]

            if lower_color is not None and upper_color is not None:
                # Convertir el ROI al espacio HSV y detectar el color
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_color = cv2.inRange(hsv_roi, lower_color, upper_color)

                if cv2.countNonZero(mask_color) > 0:
                    sospechoso_detectado = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Sospechoso localizado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Solo rectángulos verdes

    # Agregar texto según el estado
    if sospechoso_detectado:
        mensaje = "Sospechoso localizado"
    elif movimiento_detectado:
        mensaje = "Peatones caminando"
    else:
        mensaje = "Calle vacia"

    cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar el frame procesado
    cv2.imshow('Video Procesado', frame)

    # Escribir el frame en el video de salida
    out.write(frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

print(f"Video procesado guardado en: {output_video_path}")
