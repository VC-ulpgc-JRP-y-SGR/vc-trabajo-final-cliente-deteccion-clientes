from ultralytics import YOLO
from numpy import array

class Person():
    def __init__(self, bbox: tuple, max_age: int):
        self.bbox = bbox
        self.center = (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])/2))
        self.tracks = []
        self.dir = 0
        self.age = 0
        self.max_age = max_age
        self.done = False
    
    def age_one(self) -> None:
        if self.age < self.max_age: self.age += 1
        else: self.done = True

    def is_time_out(self) -> bool:
        return self.done

    def update_coords(self, bbox: tuple) -> None:
        self.tracks.append([bbox[0], bbox[2]])
        self.bbox = bbox
        self.center = (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])/2))

    def calculate_dir(self, x_left: int, x_right: int) -> None:
        if len(self.tracks) >= 2 and self.dir == 0:
            if self.tracks[-1][0] > x_right and self.tracks[-1][1] > x_right:
                    if self.tracks[-2][0] <= x_right:
                        self.dir = 1
            elif self.tracks[-1][0] < x_left and self.tracks[-1][1] < x_left:
                    if self.tracks[-2][1] >= x_left:
                        self.dir = -1
        else: self.dir = 0

    def is_going_right(self) -> bool:
        return self.dir == 1
    
    def is_going_left(self) -> bool:
        return self.dir == -1

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