import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 제스처 정의
actions = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
           'ㅘ', 'ㅙ', 'ㅝ', 'ㅞ']

# 데이터 경로 및 설정
dataset_path = "./dataset"
max_frames = 30  # 모든 시퀀스의 프레임 길이를 30으로 고정
feature_size = 63  # 각 프레임의 특징 개수

# ✅ "ㅜ" 초반 프레임 강화
def augment_start_frames(sequence, gesture, max_frames=30):
    if gesture == 'ㅜ':  
        start_frames = sequence[:5]  
        sequence = np.concatenate([start_frames] * 3 + [sequence], axis=0)  

        # ✅ 길이가 max_frames보다 크다면 자르기
        if sequence.shape[0] > max_frames:
            sequence = sequence[:max_frames]

    return sequence

# ✅ 길이 정규화
def normalize_sequence_length(data, target_length, feature_size):
    """데이터의 길이를 target_length로 맞춤 (짧으면 패딩, 길면 자름)"""
    if len(data) < target_length:
        padding = np.zeros((target_length - len(data), feature_size), dtype='float32')
        return np.vstack((data, padding)).astype('float32')  
    elif len(data) > target_length:
        return data[:target_length].astype('float32')  
    return data.astype('float32')  

# ✅ 데이터 전처리 함수
def process_gesture_data(actions, dataset_path, max_frames, feature_size):
    data = []
    labels = []
    
    for gesture in actions:
        folder_path = os.path.join(dataset_path, gesture)
        if not os.path.exists(folder_path):
            print(f"문제 있음: {gesture} 폴더가 없습니다.")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(folder_path, file_name)
                sequence_data = np.load(file_path, allow_pickle=True)

                if not isinstance(sequence_data, np.ndarray):
                    print(f"오류: {file_name} 데이터 타입이 ndarray가 아님")
                    continue

                # ✅ "ㅜ" 초반 프레임 강화 적용
                sequence_data = augment_start_frames(sequence_data, gesture) 

                # ✅ 데이터 길이 정규화 (한 번만 실행)
                normalized_data = normalize_sequence_length(sequence_data, max_frames, feature_size)  
                normalized_data = normalized_data.astype('float32')  

                # ✅ 원본 데이터 추가
                data.append(normalized_data)
                labels.append(actions.index(gesture))

                # ✅ 좌우반전 데이터 추가
                flipped_data = normalized_data.copy()
                flipped_data[:, 0] = -flipped_data[:, 0]  
                data.append(flipped_data)
                labels.append(actions.index(gesture))

                # ✅ "ㅜ" 데이터를 더 많이 학습할 수 있도록 좌우반전 데이터 추가
                if gesture == 'ㅜ':
                    for _ in range(3):  
                        data.append(flipped_data)
                        labels.append(actions.index(gesture))

    return np.array(data, dtype='float32'), np.array(labels)

# ✅ 데이터 저장 함수
def save_preprocessed_data(x_data, y_data, save_dir="./dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    y_data = to_categorical(y_data, num_classes=len(actions))
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=2020)

    np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

# ✅ 실행
x_data, labels = process_gesture_data(actions, dataset_path, max_frames, feature_size)
save_preprocessed_data(x_data, labels)
