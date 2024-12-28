import cv2
import time
from picamera2 import Picamera2


def record_video():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Configuración para guardar el video
    video_filename = "/home/pi/colors/amarillo.mp4"
    fps = 20
    frame_size = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    print("Grabando video por 60 segundos...")
    start_time = time.time()

    while True:
        frame = picam.capture_array()
        out.write(frame)
        cv2.imshow("picam", frame)

        # Detener después de 15 segundos
        if time.time() - start_time > 60:
            print(f"Video guardado como {video_filename}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Grabación detenida manualmente")
            break

    # Liberar recursos
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_video()




