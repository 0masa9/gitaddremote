import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

image_path = "ex1.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = YOLO("yolov8n-pose.pt")

results = model(image_path)
keypoints = results[0].keypoints

cnt = 0
exp_x=0
exp_y=0

# キーポイントをプロットする既存のコード
for point in keypoints[0].data:
    for x, y, z in point[5:]:
        if(cnt==0 or cnt==1 or cnt==6 or cnt==7):
            exp_x = exp_x+x
            exp_y = exp_y+y
        cnt=cnt+1
        cv2.circle(image, (int(x), int(y)), 5, (255, 255, 0), -1)

exp_x = exp_x/4
exp_y = exp_y/4


cv2.circle(image, (int(exp_x), int(exp_y)), 5, (255, 0, 0), -1)

plt.imshow(image)
plt.axis("off")
plt.show()
