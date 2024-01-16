import cv2
from .control  import PersonCounterController

def main():
    control = PersonCounterController(width = 1020, height = 600, limit = 40, max_age = 10, model = "./yolo/yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = control.read(frame)
        cv2.imshow("detections", frame)
        if cv2.waitKey(20) == 27: break

    cap.release()
    cv2.destroyAllWindows()

main()