import cv2
import joblib
import numpy as np
import mediapipe as mp
import time

# 모델 및 도구 로드
model = joblib.load("gesture_model.joblib")
scaler = joblib.load("gesture_scaler.joblib")
label_encoder = joblib.load("gesture_label_encoder.joblib")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# 전역 변수
last_prediction = None
last_prediction_time = 0
last_landmarks = None

def recognize_gesture(frame):
    global last_prediction, last_prediction_time, last_landmarks

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 추출
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # 디버깅: 원본 랜드마크 출력
            print("원본 랜드마크:", landmarks)

            # 스케일링
            try:
                scaled_landmarks = scaler.transform([landmarks])
            except Exception as e:
                print(f"[ERROR] 스케일링 실패: {e}")
                return None, None

            # 디버깅: 스케일링된 랜드마크 출력
            print("스케일링된 랜드마크:", scaled_landmarks)

            # 예측
            probabilities = model.predict_proba(scaled_landmarks)
            confidence = np.max(probabilities)
            prediction = label_encoder.inverse_transform(model.predict(scaled_landmarks))[0]

            # 디버깅: 예측 결과 출력
            print(f"예측: {prediction}, 신뢰도: {confidence}")

            # 현재 시간
            current_time = time.time()

            # 손 이동 거리 계산
            if last_landmarks is not None:
                distance = np.linalg.norm(landmarks - last_landmarks)
            else:
                distance = float('inf')

            # 쌍자음 조건
            if (
                last_prediction == prediction and
                current_time - last_prediction_time <= 1 and
                distance <= 0.05 and
                confidence > 0.5  # 신뢰도 임계값 조정
            ):
                return f"쌍자음 ({prediction})", confidence
            else:
                # 현재 예측을 기록
                last_prediction = prediction
                last_prediction_time = current_time
                last_landmarks = landmarks

            return prediction, confidence
    return None, None

# 웹캠 실행
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 손 랜드마크 시각화
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # 제스처 인식
    prediction, confidence = recognize_gesture(frame)
    if prediction:
        if "쌍자음" in prediction:
            color = (0, 0, 255)  # 빨간색 (쌍자음)
        else:
            color = (0, 255, 0)  # 초록색 (단일 자음)

        text = f"{prediction} ({confidence * 100:.1f}%)"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 화면 출력
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
