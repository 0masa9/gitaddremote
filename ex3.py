import cv2
import mediapipe as mp
import numpy as np
import math

# 動画パス
video_path = "ex3b.mp4"

# Mediapipe初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 角度計算関数
def calculate_angle(a, b, c):
    """
    3点 a, b, c を使って角度を計算する
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

# 動画を開く
cap = cv2.VideoCapture(video_path)

# フレーム番号を初期化
cnt = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # フレームにフレーム番号を描画
        cv2.putText(
            frame, str(cnt), (100, 300), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4
        )

        # Mediapipeで骨格を抽出
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 必要なポイントを取得
            height, width, _ = frame.shape
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]

            # 骨格全体を青色で描画
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

            # 右腕の角度を計算
            angle = calculate_angle(r_shoulder, r_elbow, r_hip)

            # 角度が80°〜100°の範囲なら右腕を赤色で描画
            if 80 <= angle <= 100:
                cv2.line(frame, tuple(map(int, r_shoulder)), tuple(map(int, r_elbow)), (0, 0, 255), 4)
                cv2.line(frame, tuple(map(int, r_elbow)), tuple(map(int, r_hip)), (0, 0, 255), 4)

        # フレーム番号をインクリメント
        cnt += 1

        # フレームを表示
        cv2.imshow("Pose Detection", frame)

        # ESCが押されれば終了
        if cv2.waitKey(20) == 27:
            break
    else:
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()
pose.close()
