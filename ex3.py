import cv2
import numpy as np
import mediapipe as mp

# Mediapipeを使用して骨格検出
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 動画ファイルの読み込み
input_video = "ex3b.mp4"
cap = cv2.VideoCapture(input_video)

# 動画のプロパティを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

def calculate_angle(a, b, c):
    """3点 (a, b, c) を受け取り、bを頂点とした角度を計算する"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        right_shoulder = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_height
        ])
        right_elbow = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame_height
        ])
        right_hip = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_height
        ])

        angle = calculate_angle(right_shoulder, right_elbow, right_hip)

        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            start_coords = (int(start_point.x * frame_width), int(start_point.y * frame_height))
            end_coords = (int(end_point.x * frame_width), int(end_point.y * frame_height))

            if angle >= 80 and angle <= 100 and (
                start_idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value] and
                end_idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                            mp_pose.PoseLandmark.RIGHT_WRIST.value]
            ):
                color = (0, 0, 255)
                cv2.line(frame, start_coords, end_coords, color, 2)
            else:
                color = (255, 0, 0)
                cv2.line(frame, start_coords, end_coords, color, 2)

            

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
