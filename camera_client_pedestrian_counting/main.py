import cv2
import multiprocessing as mp
import numpy as np
import ctypes
from .control  import PersonCounterController

from camera_client_pedestrian_counting.camera import CameraServer

camera_server_info = ("127.0.0.1", 5006)
frame_shape = (480, 640, 3)  # Example frame shape
frame_size = frame_shape[0] * frame_shape[1] * frame_shape[2]
queue_size = 60  # Number of frames in the pool

def create_shared_frame():
    # Create a shared array for each frame
    return mp.Array(ctypes.c_uint8, frame_size, lock=False)

def frame_to_shared(shared_frame, frame):
    # Copy the frame data to the shared frame
    np.copyto(np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape), frame)

def shared_to_frame(shared_frame):
    # Create a NumPy array view for the shared frame
    return np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)

def cam_capturer(queue):
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: failed to capture image")
            break
        frame = PersonCounterController(frame.shape[1], frame.shape[0], 10, 10, "./yolo/yolov8n-seg.pt").read(frame)
        queue.put(frame)

def cam_server(queue):
    # Initialize camera server...
    camera = CameraServer(ip=camera_server_info[0], port=camera_server_info[1])
    camera.start()

    while True:
        if queue.empty():
            continue
        frame = queue.get()
        camera.send_frame(frame)

        print(frame)

def main():
    # Create a pool of shared frames
    queue = mp.Queue(maxsize=queue_size)
    
    # Create a queue for managing frame indices
    frame_queue = mp.Queue(maxsize=queue_size)
    for i in range(queue_size):
        frame_queue.put(i)

    p1 = mp.Process(target=cam_server, args=(queue, ))
    # p2 = mp.Process(target=cam_server, args=(queue, ))
    # p3 = mp.Process(target=cam_server, args=(queue, ))
    p4 = mp.Process(target=cam_capturer, args=(queue, ))

    p1.start()
    # p2.start()
    # p3.start()
    p4.start()

    p1.join()
    #  p2.join()
    # p3.join()
    p4.join()

if __name__ == "__main__":
    main()
