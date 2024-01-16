from camera_client_pedestrian_counting.notificator import notify_client_entered, notify_client_leave
import threading
from .pedestrian_tracker import YOLOPersonDetector, Person
from numpy import array
import numpy as np
import cv2

class Painter():
    def __init__(self, font: int = cv2.FONT_HERSHEY_SIMPLEX, font_size: int = 2):
        self.font = font
        self.font_size = font_size

    def paint_lines(self, img: array, line_left: list, line_right: list) -> array:
        x1, y1, x2, y2 = line_left[0][0], line_left[0][1], line_right[1][0], line_right[1][0]
        sub_img = img[y1:y2, x1:x2]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
        img[y1:y2, x1:x2] = res

        cv2.line(img, line_left[0], line_left[1], (255, 0, 0), 2)
        cv2.putText(img, "OUT", (line_left[1][0] - 50, line_left[1][1] - 20), self.font, 0.5, (255, 0, 0), self.font_size)
        
        w = x2 - x1
        n = w/15
        x = n

        while x < w:
            cv2.line(img, (x1+int(x), 0), (x1+int(x), y2), (255, 255, 255), 1)
            x += n

        cv2.line(img, line_right[0], line_right[1], (0, 0, 255), 2)
        cv2.putText(img, "IN", (line_right[1][0] + 20, line_right[1][1] - 20), self.font, 0.5, (0, 0, 255), self.font_size)
        return img
    
    def paint_counter(self, img: array, counter: int) -> array:
        x, y = 30, 30
        x1, y1, x2, y2 = x-5, y+5, x+150, y-15
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(img, "Contador: " + str(counter), (x, y), self.font, 0.5, (255, 255, 255), self.font_size)
        return img
    
    def paint_HUB(self, img: array, left: list, right: list, counter: int) -> array:
        img = self.paint_lines(img, left, right)
        img = self.paint_counter(img, counter)
        return img

    def paint_person(self, img: array, person: Person) -> array:
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, person.center, 5, (255, 0, 255), -1)
        cv2.putText(img, "Person", (x1, y1 - 10), self.font, 0.5, (0, 255, 0), self.font_size)
        return img

class PersonCounterController():
    def __init__(self, width: int, height: int, limit: int, max_age: int, model: str):
        self.detector = YOLOPersonDetector(model)
        self.painter = Painter()
        
        self.width = width
        self.height = height
        self.limit = int(limit*width/100)
        
        self.max_age = max_age
        self.counter = 0
        self.person = None

    def count(self):
        n = self.person.get_dir()
        self.counter += n
        if n == 1: 
            thread = threading.Thread(target = notify_client_entered)
            thread.start()
        elif n == -1: 
            thread = threading.Thread(target = notify_client_leave)
            thread.start()

    def track(self, detections: list) -> None:
        if len(detections) == 0: self.person = None
        for detection in detections:
            if self.person == None: self.person = Person(detection, self.max_age)
            else:
                self.person.update_coords(detection)
                self.person.calculate_dir(self.limit, self.width - self.limit)
    
    def read(self, img: array) -> array:
        img = cv2.resize(img, (self.width, self.height))
        detections = self.detector.predict(img)

        img = self.painter.paint_HUB(img, [(self.limit, 0), (self.limit, self.height)], [(self.width - self.limit, 0), (self.width - self.limit, self.height)], self.counter)
        self.track(detections)
        if self.person != None: 
            self.count()
            img = self.painter.paint_person(img, self.person)
            img = self.painter.paint_counter(img, self.counter)

        return img
