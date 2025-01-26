import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os

def load_json_data(data_dir="dataset/output_json"):  # 경로 수정
    X = []  # 특성 데이터
    y = []  # 라벨 데이터

    # 데이터 디렉토리 순회
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)

        # 디렉토리가 아닌 경우 스킵
        if not os.path.isdir(label_path):
            continue

        # 각 라벨 폴더 안의 JSON 파일 찾기
        json_files = [f for f in os.listdir(label_path) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(label_path, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    # JSON 데이터 처리
                    if isinstance(data, list):
                        # 데이터가 리스트인 경우
                        for entry in data:
                            if isinstance(entry, dict) and 'landmarks' in entry:
                                landmarks = entry['landmarks']
                                features = []
                                for landmark in landmarks:
                                    features.extend([landmark['x'], landmark['y'], landmark['z']])
                                if len(features) == 63:  # 21개의 랜드마크 * 3(x, y, z)
                                    X.append(features)
                                    y.append(label_folder)  # 폴더명이 라벨
                    elif isinstance(data, dict):
                        # 데이터가 딕셔너리인 경우
                        for key, value in data.items():
                            if isinstance(value, list):
                                for frame in value:
                                    features = []
                                    for landmark in frame:
                                        features.extend([landmark['x'], landmark['y'], landmark['z']])
                                    if len(features) == 63:  # 21개의 랜드마크 * 3(x, y, z)
                                        X.append(features)
                                        y.append(label_folder)  # 폴더명이 라벨
            
            except Exception as e:
                print(f"Error loading file {json_path}: {str(e)}")
                continue

    if not X:
        raise ValueError("No valid data found in the specified directory")

    return np.array(X), np.array(y)

def main():
    try:
        # JSON 데이터 로드
        print("Loading data from JSON files...")
        X, y = load_json_data()

        print(f"Data shape: X={X.shape}, y={y.shape}")

        # 라벨 인코딩 (문자열 라벨을 숫자로 변환)
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = np.array([label_map[label] for label in y])

        print(f"Loaded {len(X)} samples with {len(unique_labels)} different labels")
        print("Labels found:", unique_labels)
        print(f"Label mapping: {label_map}")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                           test_size=0.2, 
                                                           random_state=42,
                                                           stratify=y_encoded)  # 균형잡힌 분할을 위해 stratify 추가

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 학습
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, 
                                     random_state=42,
                                     n_jobs=-1)  # 모든 CPU 코어 사용
        model.fit(X_train_scaled, y_train)

        # 모델 평가
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")

        # 모델과 스케일러, 라벨 매핑 저장
        print("\nSaving model and related files...")
        model_path = 'gesture_model.joblib'
        scaler_path = 'gesture_scaler.joblib'
        label_map_path = 'label_map.json'

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=4)

        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Label map saved to: {label_map_path}")
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
