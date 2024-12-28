
# from picamera2 import Picamera2

# def stream_video():
#     picam = Picamera2()
#     picam.preview_configuration.main.size=(1280, 720)
#     picam.preview_configuration.main.format="RGB888"
#     picam.preview_configuration.align()
#     picam.configure("preview")
#     picam.start()

#     while True:
#         frame = picam.capture_array()
#         cv2.imshow("picam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     stream_video()

import cv2
import time
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    last_saved_time = time.time()  # Tiempo inicial
    photo_count = 0  # Contador para numerar las fotos

    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        current_time = time.time()
        if current_time - last_saved_time >= 5: 
            filename = f"/home/pi/colors/photo_ROJO_{photo_count}.jpg"  # Nombre de archivo
            cv2.imwrite(filename, frame) 
            print(f"Saved {filename}")
            photo_count += 1
            last_saved_time = current_time 
            


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()


