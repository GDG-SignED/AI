import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ✅ MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ✅ 데이터 저장 경로 및 제스처 설정 (라벨 통합 적용)
dataset_path = "./dataset"
gesture_mapping = {
    'ㄱ': 'ㄱ', 'ㄲ': 'ㄲ', 'ㄴ': 'ㄴ', 'ㄷ': 'ㄷ', 'ㄸ': 'ㄸ', 'ㄹ': 'ㄹ', 'ㅁ': 'ㅁ',
    'ㅂ': 'ㅂ', 'ㅃ': 'ㅃ', 'ㅅ': 'ㅅ', 'ㅆ': 'ㅆ', 'ㅇ': 'ㅇ', 'ㅈ': 'ㅈ', 'ㅉ': 'ㅉ',
    'ㅊ': 'ㅊ', 'ㅋ': 'ㅋ', 'ㅌ': 'ㅌ', 'ㅍ': 'ㅍ', 'ㅎ': 'ㅎ', 'ㅏ': 'ㅏ', 'ㅑ': 'ㅑ',
    'ㅓ': 'ㅓ', 'ㅕ': 'ㅕ', 'ㅗ': 'ㅗ', 'ㅛ': 'ㅛ', 'ㅜ': 'ㅜ', 'ㅠ': 'ㅅ',  # ✅ 'ㅠ'를 'ㅅ'으로 저장 (라벨 통합)
    'ㅡ': 'ㅡ', 'ㅣ': 'ㅣ', 'ㅐ': 'ㅐ', 'ㅒ': 'ㅒ', 'ㅔ': 'ㅔ', 'ㅖ': 'ㅖ', 'ㅢ': 'ㅢ',
    'ㅚ': 'ㅚ', 'ㅟ': 'ㅟ', 'ㅘ': 'ㅘ', 'ㅙ': 'ㅙ', 'ㅞ': 'ㅞ', 'ㅝ': 'ㅝ'
}
gestures = list(set(gesture_mapping.values()))  # ✅ 중복 제거 후 리스트 변환

# ✅ 쌍자음, 이중모음 처리 규칙
double_consonants = {'ㄲ': ['ㄱ', 'ㄱ'], 'ㄸ': ['ㄷ', 'ㄷ'], 'ㅃ': ['ㅂ', 'ㅂ'], 'ㅆ': ['ㅅ', 'ㅅ'], 'ㅉ': ['ㅈ', 'ㅈ']}
double_vowels = {'ㅘ': ['ㅗ', 'ㅏ'], 'ㅙ': ['ㅗ', 'ㅐ'], 'ㅝ': ['ㅜ', 'ㅓ'], 'ㅞ': ['ㅜ', 'ㅔ']}

# ✅ 프레임 및 시퀀스 설정
frame_count = 30  # 프레임 개수 (1 시퀀스 당 30개 프레임)
sequences = 30    # 시퀀스 개수 (1 제스처 당 30개의 시퀀스)
files_per_gesture = 40  # 일반 자모음 파일 개수
files_per_double_gesture = 200  # 쌍자음/이중모음 파일 개수

# ✅ 헷갈리는 제스처 (150개 촬영)
special_gestures = {'ㅔ', 'ㅕ', 'ㅌ', 'ㅋ', 'ㄹ', 'ㅓ', 'ㅜ'}
special_files_per_gesture = 150

# ✅ 사용자 이름
user_name = "이수연"

# ✅ 데이터 저장 디렉토리 생성
os.makedirs(dataset_path, exist_ok=True)
for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

# ✅ 손 랜드마크 데이터 캡처 함수
def capture_hand_data(cap, hands):
    frame_list = []
    while len(frame_list) < frame_count:  # 30개 프레임 수집 완료될 때까지 반복
        ret, frame = cap.read()
        if not ret:
            print("카메라 읽기에 실패하였습니다.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                frame_list.append(landmarks)
        else:
            frame_list.append(np.zeros(21 * 3))  # 손이 감지되지 않으면 빈 데이터 추가
            print("손이 감지되지 않았습니다.")

        # 캡처 중 화면 출력
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            return None

    return frame_list if len(frame_list) == frame_count else None  # 30프레임이 모이면 반환

# ✅ 쌍자음 및 이중모음 처리 함수
def handle_double_gestures(cap, hands, base_gesture, components):
    frame_list = []
    for i, comp in enumerate(components):
        print(f"{base_gesture} 촬영 중... {i + 1}번째 구성 제스처 ({comp}) 촬영")
        result = capture_hand_data(cap, hands)
        if result is None:  # 사용자가 `q`를 누르면 종료
            return None
        frame_list.extend(result)
    return frame_list

# ✅ 데이터 수집 메인 루프
def collect_gesture_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기에 실패하였습니다.")
        return

    print("웹캠 연결에 성공하였습니다.")
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        for gesture in gesture_mapping.keys():  # 원래 입력값 기준으로 반복
            mapped_gesture = gesture_mapping[gesture]  # ✅ 매핑된 라벨 사용
            gesture_path = os.path.join(dataset_path, mapped_gesture)

            # ✅ 파일 개수 설정
            if gesture in double_consonants or gesture in double_vowels:
                target_files = files_per_double_gesture
            elif gesture in special_gestures:
                target_files = special_files_per_gesture
            else:
                target_files = files_per_gesture

            # ✅ "ㅜ" 데이터 수집량 증가 적용
            if gesture == "ㅜ":
                target_files = 250  

            for sequence in range(target_files):
                file_name = f"{mapped_gesture}_{user_name}_{sequence + 1}.npy"
                save_path = os.path.join(gesture_path, file_name)
                if os.path.exists(save_path):
                    print(f"{file_name} 이미 저장되어 있습니다. 건너뜁니다.")
                    continue

                print(f"데이터 수집 중: {gesture} (실제 저장: {mapped_gesture}), sequence {sequence + 1}/{target_files}")

                # ✅ 쌍자음 & 이중모음 처리
                if gesture in double_consonants:
                    frame_list = handle_double_gestures(cap, hands, gesture, double_consonants[gesture])
                elif gesture in double_vowels:
                    frame_list = handle_double_gestures(cap, hands, gesture, double_vowels[gesture])
                else:
                    frame_list = capture_hand_data(cap, hands)

                if frame_list:
                    np.save(save_path, np.array(frame_list))
                    print(f"{mapped_gesture}의 데이터 저장 완료!")

    cap.release()
    cv2.destroyAllWindows()

# ✅ 실행
if __name__ == "__main__":
    collect_gesture_data()
