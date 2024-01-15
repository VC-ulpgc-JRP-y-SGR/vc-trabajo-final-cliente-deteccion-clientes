from .control import PersonCounterController
from .camera import CameraServer
import cv2

def main():
    control = PersonCounterController(width = 1020, height = 600, limit = 20, max_age = 10, model = "./yolo/yolov8n-seg.pt")
    camera = CameraServer(ip="172.20.10.3", port=5006)
    camera.start()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = control.read(frame)
        camera.send_frame(frame)

        if cv2.waitKey(20) == 27: break

    cv2.destroyAllWindows()