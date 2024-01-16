import cv2
import multiprocessing as mp
import numpy as np
import ctypes
from .control  import PersonCounterController
from camera_client_pedestrian_counting.camera import CameraServer

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame = PersonCounterController(frame.size[0], frame.size[1], 10, "./yolo/yolov8n-seg.pt").read(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



