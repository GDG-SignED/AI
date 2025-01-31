import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
from PIL import Image, ImageTk
from collections import deque
import json
import os

def calculate_angles_from_joint(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    compareV1 = v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :]
    compareV2 = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
    angle = np.arccos(np.clip(np.einsum('nt,nt->n', compareV1, compareV2), -1.0, 1.0))
    return np.degrees(angle)

def load_and_preprocess_data(data_dir="dataset/output_json"):
    X = []
    y = []

    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        json_files = [f for f in os.listdir(label_path) if f.endswith('.json') and not f.startswith('test')]

        for json_file in json_files:
            json_path = os.path.join(label_path, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    if isinstance(data, list):
                        for frame_data in data:
                            if isinstance(frame_data, dict) and 'landmarks' in frame_data:
                                landmarks = frame_data['landmarks']
                                if len(landmarks) == 21:
                                    joint = np.zeros((21, 3))
                                    for i, landmark in enumerate(landmarks):
                                        joint[i] = [landmark['x'], landmark['y'], landmark['z']]
                                    angles = calculate_angles_from_joint(joint)
                                    X.append(angles)
                                    y.append(label_folder)

                    elif isinstance(data, dict):
                        for key, frames in data.items():
                            if isinstance(frames, list):
                                for frame in frames:
                                    if isinstance(frame, list):
                                        joint = np.zeros((21, 3))
                                        for i, landmark in enumerate(frame):
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

class SignLanguageTest:
    def __init__(self, root):
        self.root = root
        self.hangul_labels = [
            'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ',
            'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
        ]
        self.double_consonants = {'ㄲ': 'ㄱ', 'ㄸ': 'ㄷ', 'ㅃ': 'ㅂ', 'ㅆ': 'ㅅ', 'ㅉ': 'ㅈ'}
        self.gesture_history = deque(maxlen=8)
        self.intensity_history = deque(maxlen=8)
        self.setup_model()
        self.setup_gui()
        self.setup_test_data()
        self.setup_camera()
        
    def setup_model(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        X, y = load_and_preprocess_data()
        self.model.fit(X, y)
        
    def setup_test_data(self):
        self.test_data = {
            "자음": [l for l in self.hangul_labels if l in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'],
            "쌍자음": [l for l in self.hangul_labels if l in 'ㄲㄸㅃㅆㅉ'],
            "모음": [l for l in self.hangul_labels if l not in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ']
        }
        self.current_question = None
        self.answers = []
        self.timer = 30
        self.test_in_progress = False

    def calculate_gesture_intensity(self, joint):
        velocity = np.diff(joint, axis=0)
        intensity = np.mean(np.abs(velocity))
        return intensity
        
    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.pack(fill='x', pady=5)
        
        self.question_label = ttk.Label(
            self.top_frame, 
            text="테스트를 시작하려면 '시작' 버튼을 누르세요", 
            font=('맑은 고딕', 14)
        )
        self.question_label.pack(side='left')
        
        self.timer_label = ttk.Label(
            self.top_frame, 
            text="30", 
            font=('맑은 고딕', 14)
        )
        self.timer_label.pack(side='right')
        
        self.camera_frame = ttk.Frame(self.main_frame)
        self.camera_frame.pack(expand=True, fill='both')
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(expand=True, fill='both')
        
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill='x', pady=5)
        
        self.feedback_label = ttk.Label(
            self.bottom_frame, 
            text="", 
            font=('맑은 고딕', 12)
        )
        self.feedback_label.pack(pady=5)
        
        self.start_button = ttk.Button(
            self.bottom_frame, 
            text="테스트 시작", 
            command=self.start_test
        )
        self.start_button.pack(pady=5)
        
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.update_camera()
        
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS)
                self.process_hand_landmarks(results.multi_hand_landmarks[0], frame)
            elif self.test_in_progress:
                self.feedback_label.config(text="손동작을 인식할 수 없습니다. 다시 시도해 주세요.")
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            self.camera_label.config(image=photo)
            self.camera_label.image = photo
            
        self.root.after(10, self.update_camera)
        
    def process_hand_landmarks(self, hand_landmarks, frame):
        if not self.test_in_progress:
            return
            
        joint = np.zeros((21, 3))
        for j, lm in enumerate(hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z]
            
        angles = calculate_angles_from_joint(joint)
        predicted_label = self.model.predict([angles])[0]
        confidence = max(self.model.predict_proba([angles])[0])
        
        intensity = self.calculate_gesture_intensity(joint)
        self.intensity_history.append(intensity)
        self.gesture_history.append(predicted_label)
        
        most_frequent_gesture = max(set(self.gesture_history), key=self.gesture_history.count)
        avg_intensity = np.mean(self.intensity_history)
        
        final_prediction = most_frequent_gesture

        # ㅅ과 ㅠ 관련 특별 처리
        if self.current_question == "ㅠ":
            if most_frequent_gesture != "ㅠ":
                final_prediction = "ㅅ" if random.random() < 0.5 else "ㅆ"
        elif most_frequent_gesture == "ㅠ":
            final_prediction = "ㅅ" if random.random() < 0.5 else "ㅆ"
        # 문제 유형에 따른 예측 보정
        elif self.current_question in self.test_data["쌍자음"]:
            base_consonant = self.double_consonants.get(self.current_question)
            if most_frequent_gesture == base_consonant and avg_intensity > 0.1:
                final_prediction = self.current_question
        else:
            # 쌍자음이 아닌 문제에서는 높은 신뢰도가 필요
            if confidence < 0.8:
                for double, single in self.double_consonants.items():
                    if final_prediction == double:
                        final_prediction = single

        if len(self.gesture_history) == self.gesture_history.maxlen:
            if final_prediction == self.current_question:
                self.handle_correct_answer()
            self.feedback_label.config(
                text=f"인식된 수어: {final_prediction} (신뢰도: {confidence:.2f}, 강도: {avg_intensity:.3f})"
            )
        
    def start_test(self):
        self.test_in_progress = True
        self.answers = []
        self.timer = 30
        self.next_question()
        self.update_timer()
        self.start_button.config(state='disabled')
        
    def next_question(self):
        if self.timer > 0:
            category = random.choice(list(self.test_data.keys()))
            self.current_question = random.choice(self.test_data[category])
            self.question_label.config(text=f"문제: {self.current_question}")
            
    def update_timer(self):
        if self.timer > 0:
            self.timer -= 1
            self.timer_label.config(text=str(self.timer))
            self.root.after(1000, self.update_timer)
        else:
            self.show_results()
            
    def handle_correct_answer(self):
        self.answers.append(True)
        self.gesture_history.clear()
        self.intensity_history.clear()
        self.next_question()
        
    def show_results(self):
        self.test_in_progress = False
        correct = sum(self.answers)
        total = len(self.answers)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        result_window = tk.Toplevel(self.root)
        result_window.title("테스트 결과")
        
        ttk.Label(
            result_window,
            text=f"정답률: {accuracy:.1f}%\n맞춘 문제: {correct}/{total}",
            font=('맑은 고딕', 14)
        ).pack(padx=20, pady=20)
        
        self.start_button.config(state='normal')
        self.question_label.config(text="테스트를 시작하려면 '시작' 버튼을 누르세요")

def main():
    root = tk.Tk()
    root.title("수어 테스트")
    app = SignLanguageTest(root)
    root.mainloop()

if __name__ == "__main__":
    main()