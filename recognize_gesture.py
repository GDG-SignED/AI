import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import json
import os
from PIL import ImageFont, ImageDraw, Image
from collections import deque
from time import time

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
try:
    font = ImageFont.truetype(font_path, 30)
except Exception as e:
    print(f"폰트 로드 실패: {e}")
    font = ImageFont.load_default()

class GestureDetector:
    def __init__(self, time_threshold=0.5, confidence_threshold=0.8, stable_duration=0.1, max_history=8):
        self.last_gesture = None
        self.last_gesture_time = 0
        self.gesture_history = deque(maxlen=max_history)  # 음운 리스트
        self.time_threshold = time_threshold  # 쌍자음 인식을 위한 시간 임계값
        self.confidence_threshold = confidence_threshold  # 신뢰도 임계값
        self.stable_duration = stable_duration  # 제스처 안정화 시간
        self.current_stable_gesture = None
        self.stable_start_time = 0

    def is_gesture_stable(self, predicted_label, confidence, current_time):
        """제스처가 일정 시간 동안 안정적인지 확인"""
        if self.current_stable_gesture != predicted_label:
            self.current_stable_gesture = predicted_label
            self.stable_start_time = current_time
            return False
        
        return (current_time - self.stable_start_time >= self.stable_duration and 
                confidence >= self.confidence_threshold)

    def get_most_frequent_gesture(self):
        """리스트에서 가장 많이 등장한 음운 반환"""
        if not self.gesture_history:
            return None
        return max(set(self.gesture_history), key=self.gesture_history.count)

    def process_gesture(self, predicted_label, confidence):
        current_time = time()
        result_label = predicted_label
        is_double = False

        # 제스처가 안정적인지 확인
        if not self.is_gesture_stable(predicted_label, confidence, current_time):
            return self.last_gesture or predicted_label, False, confidence

        # 히스토리에 추가
        self.gesture_history.append(predicted_label)

        # 최빈값 계산
        most_frequent_gesture = self.get_most_frequent_gesture()

        # 쌍자음 패턴 체크
        if len(self.gesture_history) >= 2:
            last_two = list(self.gesture_history)[-2:]
            time_diff = current_time - self.last_gesture_time

            # 쌍자음 조건 검사
            if (last_two[0] == last_two[1] and  # 같은 제스처
                time_diff < self.time_threshold and    # 시간 임계값 이내
                time_diff > self.stable_duration):     # 최소 안정화 시간 이상
                
                result_label = f"쌍{predicted_label}"
                is_double = True
                # 쌍자음 인식 후 히스토리 초기화
                self.gesture_history.clear()

        self.last_gesture = predicted_label
        self.last_gesture_time = current_time

        # 최빈값을 반환
        return most_frequent_gesture, is_double, confidence

def load_and_preprocess_data(data_dir="dataset/output_json"):
    X = []  # 특성 데이터 (angles)
    y = []  # 라벨 데이터
    
    # 각 라벨 폴더 순회
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        
        if not os.path.isdir(label_path):
            continue
            
        # 각 라벨 폴더 안의 JSON 파일 찾기
        json_files = [f for f in os.listdir(label_path) if f.endswith('.json') and not f.startswith('test')]
        
        for json_file in json_files:
            json_path = os.path.join(label_path, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # 데이터 형식 확인 및 처리
                    if isinstance(data, list):  # 조예인 형식
                        for frame_data in data:
                            landmarks = frame_data.get('landmarks', [])
                            if len(landmarks) == 21:
                                joint = np.zeros((21, 3))
                                for i, landmark in enumerate(landmarks):
                                    joint[i] = [landmark['x'], landmark['y'], landmark['z']]
                                angles = calculate_angles_from_joint(joint)
                                X.append(angles)
                                y.append(label_folder)
                    elif isinstance(data, dict):  # 이수연 형식
                        for filename, landmarks_list in data.items():
                            for landmarks_frame in landmarks_list:
                                if len(landmarks_frame) == 21:  # 21개의 랜드마크 확인
                                    joint = np.zeros((21, 3))
                                    for i, landmark in enumerate(landmarks_frame):
                                        joint[i] = [landmark['x'], landmark['y'], landmark['z']]
                                    angles = calculate_angles_from_joint(joint)
                                    X.append(angles)
                                    y.append(label_folder)
                        
            except Exception as e:
                print(f"Error loading file {json_path}: {str(e)}")
                continue
    
    if not X:
        raise ValueError("No valid data found in the specified directory")
        
    return np.array(X), np.array(y)

def calculate_angles_from_joint(joint):
    """관절 각도 계산"""
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
    v = v2 - v1
    
    # 벡터 정규화
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    
    # 각도 계산을 위한 벡터 쌍 선택
    compareV1 = v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :]
    compareV2 = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
    
    # 각도 계산
    angle = np.arccos(np.clip(np.einsum('nt,nt->n', compareV1, compareV2), -1.0, 1.0))
    return np.degrees(angle)

def draw_text_with_pillow(image, text, position, font, color=(0, 255, 0)):
    """Pillow를 사용해 한글 텍스트를 OpenCV 이미지에 그리기"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    try:
        # 데이터 로드 및 전처리
        print("Loading and preprocessing data...")
        X, y = load_and_preprocess_data()
        
        print(f"데이터 로드 완료: {len(X)} 샘플, 라벨: {np.unique(y)}")
        
        # 훈련/테스트 세트 분할
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, 
                                                           stratify=y)
        
        # KNN 모델 학습
        print("Training KNN model...")
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        
        # 모델 평가
        train_accuracy = knn.score(X_train, y_train)
        test_accuracy = knn.score(X_test, y_test)
        print(f"Training accuracy: {train_accuracy:.2f}")
        print(f"Test accuracy: {test_accuracy:.2f}")
        
        # 제스처 감지기 초기화 (파라미터 조정 가능)
        gesture_detector = GestureDetector(
            time_threshold=0.5,        # 쌍자음 인식을 위한 최대 시간 간격
            confidence_threshold=0.8,  # 최소 신뢰도
            stable_duration=0.1        # 제스처 안정화를 위한 최소 시간
        )
        
        # 실시간 테스트
        print("Starting real-time testing... Press 'q' or ESC to exit.")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Cannot access the camera")
            
        # 윈도우 생성 및 설정
        cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Gesture Recognition', 800, 600)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # 좌우 반전
            frame = cv2.flip(frame, 1)
            
            # MediaPipe 처리
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # 랜드마크 그리기
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    try:
                        # 관절 좌표 추출
                        joint = np.zeros((21, 3))
                        for j, lm in enumerate(hand_landmarks.landmark):
                            joint[j] = [lm.x, lm.y, lm.z]
                        
                        # 각도 계산 및 예측
                        angles = calculate_angles_from_joint(joint)
                        predicted_label = knn.predict([angles])[0]
                        confidence = max(knn.predict_proba([angles])[0])
                        
                        # 제스처 처리
                        final_label, is_double, conf = gesture_detector.process_gesture(
                            predicted_label, confidence)
                        
                        # 결과 표시 (쌍자음일 경우 다른 색상으로 표시)
                        result_text = f"예측: {final_label} ({conf:.2f})"
                        color = (0, 0, 255) if is_double else (0, 255, 0)
                        frame = draw_text_with_pillow(frame, result_text, (10, 30), font, color=color)
                        
                        # 안정성 상태 표시
                        stability_text = "안정적" if gesture_detector.is_gesture_stable(
                            predicted_label, confidence, time()) else "불안정"
                        frame = draw_text_with_pillow(
                            frame, f"상태: {stability_text}", (10, 70), font,
                            (0, 255, 0) if stability_text == "안정적" else (0, 165, 255)
                        )
                        
                    except Exception as e:
                        print(f"예측 중 오류 발생: {e}")
            
            # 안내 메시지 표시
            frame = draw_text_with_pillow(frame, "종료: ESC 또는 q", (10, 550), font)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:  # ESC 또는 'q'
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        # 창이 완전히 닫히도록 대기
        for _ in range(5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
