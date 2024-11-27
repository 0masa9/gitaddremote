import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

image_path = "ex2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = YOLO("yolov8x.pt")
results = model(image_path)

boxes = results[0].boxes

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(image)
plt.axis("off")
plt.show()
