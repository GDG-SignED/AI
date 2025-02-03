import cv2
import numpy as np
import mediapipe as mp
import os
import time

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 저장 경로
dataset_path = "./Sign_ED/test/testdata"
os.makedirs(dataset_path, exist_ok=True)

# 저장할 제스처 라벨 목록 입력받기
manual_labels = input("👉 저장할 라벨을 입력하세요 (쉼표로 구분, 예: ㄱ, ㄲ, ㄷ, ㅘ, ㅙ): ").split(',')
manual_labels = [label.strip() for label in manual_labels]  # 공백 제거

# 쌍자음/이중모음 처리 규칙
double_consonants = {'ㄲ': ['ㄱ', 'ㄱ'], 'ㄸ': ['ㄷ', 'ㄷ'], 'ㅃ': ['ㅂ', 'ㅂ'], 'ㅆ': ['ㅅ', 'ㅅ'], 'ㅉ': ['ㅈ', 'ㅈ']}
double_vowels = {'ㅘ': ['ㅗ', 'ㅏ'], 'ㅙ': ['ㅗ', 'ㅐ'], 'ㅝ': ['ㅜ', 'ㅓ'], 'ㅞ': ['ㅜ', 'ㅔ']}

# 웹캠 열기
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

for manual_label in manual_labels:
    # 쌍자음 & 이중모음 체크
    if manual_label in double_consonants:
        gesture_components = double_consonants[manual_label]
    elif manual_label in double_vowels:
        gesture_components = double_vowels[manual_label]
    else:
        gesture_components = [manual_label]

    sequence = []  # 라벨당 한 개의 npy 파일을 저장하기 위해 초기화

    for component in gesture_components:
        print(f"📸 현재 촬영 중: {component} (손을 올리고 기다려 주세요...)")

        component_sequence = []  # 개별 제스처 데이터 저장

        start_time = time.time()
        while time.time() - start_time < 3:  # 3초 동안 촬영
            ret, frame = cap.read()
            if not ret:
                break

            # BGR → RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            # 손 랜드마크 그리기
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 랜드마크 데이터 수집
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    component_sequence.append(landmarks)

            else:
                component_sequence.append([0] * 63)  # 손이 감지되지 않을 경우 0으로 패딩

            # 화면에 표시
            cv2.putText(frame, f"Label: {component} (종료: Q)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Webcam Feed", frame)

            # 종료키 (Q 누르면 종료)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("🚪 프로그램 종료!")
                exit()

        # 촬영한 데이터를 하나의 리스트에 추가
        sequence.extend(component_sequence)

    # 최종적으로 한 개의 npy 파일로 저장
    np.save(os.path.join(dataset_path, f"test_landmarks_{manual_label}.npy"), np.array(sequence))
    np.save(os.path.join(dataset_path, f"test_label_{manual_label}.npy"), np.array([manual_label]))
    print(f"✅ {manual_label} 데이터 저장 완료!")

    # 다음 촬영을 위한 3초 대기
    print("⏳ 다음 제스처로 이동 중... (손을 내렸다가 다시 올려주세요)")
    time.sleep(3)

cap.release()
cv2.destroyAllWindows()
print("🚀 모든 데이터 저장 완료!")
