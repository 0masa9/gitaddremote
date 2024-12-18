import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

video_path = "ex3b.mp4"
model = YOLO("yolov8n-pose.pt")  
cap = cv2.VideoCapture(video_path)

image_path = "ex1.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image_results = model(image_path)
image_keypoints = image_results[0].keypoints
image_data=image_keypoints[0].data

# 動画を開く
cnt = 0
# フレーム番号を0にする

skeleton = [
    (2,4),(3,5),(0,2),(1,3),
    (0,1),(6,7),
    (0,6),(1,7),
    (6,8),(7,9),(8,10),(9,11)
]
min = 10000000
min_flame=0
while cap.isOpened():
    count=0
    success, frame = cap.read()
    # フレームを読み出す

    results = model(frame)
    keypoints = results[0].keypoints
    video_data=keypoints[0].data


    sum = 0
    for i in range(5,17):
        for j in range (2):
            test = video_data[0][i][j]-image_data[0][i][j]
            test=abs(test)
            sum=sum+test

    if(min>sum):
        min_flame= cnt
        min=sum

    print(cnt)
    print(sum)
    print(min)
    print(min_flame)
    if success:
        # 読み出しに成功すれば以下を実行する
        cv2.putText(
            frame, str(cnt), (100, 300), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4
        )
        # フレームにフレーム番号を描画する
        cnt += 1
        # フレーム番号をインクリメントする
        cv2.imshow("", frame)
        # フレームを表示する

        if cv2.waitKey(20) == 27:
            break
        # ESCが押されれば終了
    else:
        break

print("↓一番近いフレーム↓")
print(min_flame)
cap.release()
cv2.destroyAllWindows()

