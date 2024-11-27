import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

image_path = "ex1.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

model = YOLO("yolov8n-pose.pt")  

results = model(image_path)
keypoints = results[0].keypoints

# キーポイントをプロットする既存のコード
for point in keypoints[0].data:
    for x, y, z in point[5:]:
        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

# ここからボーンを描画するコードを追加
# COCOフォーマットのキーポイント接続を定義
skeleton = [
    (2,4),(3,5),(0,2),(1,3),
    (0,1),(6,7),
    (0,6),(1,7),
    (6,8),(7,9),(8,10),(9,11)
]

# キーポイント座標を取得してボーンを描画
for point in keypoints[0].data:
    keypoints_coords = [(int(x), int(y)) for x, y, z in point[5:]]


    # 各キーポイントペアに対して処理
    for start, end in skeleton:
        if start < len(keypoints_coords) and end < len(keypoints_coords):  # インデックスが範囲内か確認
            # キーポイントが有効かチェック（例: 座標が0以上か）
            if keypoints_coords[start][0] > 0 and keypoints_coords[end][0] > 0:
                cv2.line(image, keypoints_coords[start], keypoints_coords[end], (0, 255, 0), 2)

plt.imshow(image)
plt.axis("off")
plt.show()
