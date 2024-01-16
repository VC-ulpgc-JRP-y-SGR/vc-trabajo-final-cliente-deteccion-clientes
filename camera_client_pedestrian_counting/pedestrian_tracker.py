from ultralytics import YOLO
from numpy import array

class Person():
    def __init__(self, bbox: tuple):
        self.bbox = bbox
        self.center = (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])/2))
        self.tracks = []
        self.dir = 0
        self.state = 0

    def update_coords(self, bbox: tuple) -> None:
        self.tracks.append(int(self.center[0]))
        self.bbox = bbox
        self.center = (int((bbox[0] + bbox[2])//2), int((bbox [1] + bbox[3])/2))

    def calculate_dir(self, x_left: int, x_right: int) -> None:
        if len(self.tracks) >= 2:
            if self.tracks[-1] > x_right and self.tracks[-2] <= x_right:
                if self.dir != 1:
                    self.dir = 1
                    self.state = 1
            elif self.tracks[-1] < x_left and self.tracks[-2] >= x_left:
                if self.dir != -1:
                    self.dir = -1
                    self.state = 1

    def get_dir(self) -> int:
        if self.state == 1: 
            self.state = 0
            return self.dir
        else: return 0

class YOLOPersonDetector():
    def __init__(self, model):
        self.model = YOLO(model)

    def predict(self, img: array) -> list:
        predictions = self.model.predict(img)
        detected_persons = []
        
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                category = int(box.cls[0])
                if category == 0:
                    detected_persons.append((x1, y1, x2, y2))

        return detected_persons