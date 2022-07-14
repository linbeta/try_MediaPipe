import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

webcam = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    if not webcam.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = webcam.read()
        # 0: vertical flip / 1: mirror
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Error: Cannot load frames!!!")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_style.get_default_hand_landmarks_style(),
                    mp_drawing_style.get_default_hand_connections_style()
                )

        cv2.imshow("MediaPipe tutorial", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            print("Exit")
            break

webcam.release()
cv2.destroyAllWindows()
