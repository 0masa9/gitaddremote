import cv2

video_path = "ex3b.mp4"

cap = cv2.VideoCapture(video_path)
# 動画を開く
cnt = 0
# フレーム番号を0にする

while cap.isOpened():
    success, frame = cap.read()
    # フレームを読み出す

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

cap.release()
cv2.destroyAllWindows()
