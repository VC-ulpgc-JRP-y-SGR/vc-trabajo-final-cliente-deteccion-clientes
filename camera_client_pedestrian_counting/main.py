from .control import PersonCounterController
from .camera import CameraServer
import threading
import cv2
import queue

def send_frames(frame_queue, exit_event, camera_server_info):
    camera = CameraServer(ip=camera_server_info[0], port=camera_server_info[1])
    camera.start()

    while not exit_event.is_set():
        while not frame_queue.empty():
            if frame_queue.qsize() > 10:
                frame_queue.get_nowait()
            else:
                break

        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                camera.send_frame(frame)
            except Exception as e:
                print(f"Failed to send frame: {e}")
                exit_event.set()

    camera.stop()

def main():
    # Initialize control
    control = PersonCounterController(width=1020, height=600, limit=20, max_age=10, model="./yolo/yolov8n-seg.pt")

    # Initialize frame queue and exit flag
    frame_queue = queue.Queue()
    exit_event = threading.Event()

    # Start the frame sending thread
    camera_server_info = ("127.0.0.1", 5006)
    sending_thread = threading.Thread(target=send_frames, args=(frame_queue, exit_event, camera_server_info))
    sending_thread.start()
    # Main loop to capture and process frames
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = control.read(frame)
            frame_queue.put(frame)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(20) == 27: break  # Escape key pressed
    finally:
        exit_event.set()
        sending_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == 'main':
    main()