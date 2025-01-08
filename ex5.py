import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

image_path = "ex2.jpg"

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = YOLO("yolov8x.pt")
results = model(image_path)

boxes = results[0].boxes

lower_blue = np.array([100, 50, 100])  # 青色の下限値
upper_blue = np.array([160, 255, 255])  # 青色の上限値

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    person_mask = blue_mask[y1:y2, x1:x2]
    blue_pixel_count = cv2.countNonZero(person_mask)


    if blue_pixel_count > 0:
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
