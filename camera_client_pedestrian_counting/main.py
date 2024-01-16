from .control import PersonCounterController
from .camera import CameraServer
import multiprocessing
import cv2
import os

def send_frames(frame_queue, exit_flag, camera_server_info):
    camera = CameraServer(ip=camera_server_info[0], port=camera_server_info[1])
    camera.start()
    print(frame_queue)

    while not exit_flag.value:
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
                exit_flag.value = 1

    camera.stop()

def main():
    # Initialize control
    control = PersonCounterController(width=1020, height=600, limit=20, max_age=10, model="./yolo/yolov8n-seg.pt")

    # Initialize frame queue and exit flag
    frame_queue = multiprocessing.Queue()
    exit_flag = multiprocessing.Value('i', 0)

    # Start the frame sending process
    camera_server_info = ("127.0.0.1", 5006)
    sending_process = multiprocessing.Process(target=send_frames, args=(frame_queue, exit_flag, camera_server_info))
    sending_process.start()

    # Main loop to capture and process frames
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = control.read(frame)
            frame_queue.put(frame)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(20) == 27: break
    finally:
        exit_flag.value = 1
        sending_process.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
