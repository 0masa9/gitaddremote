import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

image_path = "ex2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = YOLO("yolov8x.pt")
results = model(image_path)

boxes = results[0].boxes

max=0


for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    s = abs(x2-x1)*abs(y2-y1)
    if(max<s):
        max = s
        max_x1=x1
        max_x2=x2
        max_y1=y1
        max_y2=y2

image = cv2.rectangle(image, (max_x1, max_y1), (max_x2, max_y2), (255, 0, 0), 2)

plt.imshow(image)
plt.axis("off")
plt.show()
