import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def load_json_data(data_dir="AI/dataset/output_json"):
    """JSON 데이터를 로드하고 학습에 사용할 데이터셋 준비"""
    X = []  # 특징 데이터
    y = []  # 라벨 데이터

    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        # _json 폴더 접근
        json_folder = os.path.join(label_path, f"{label_folder}_json")
        if not os.path.exists(json_folder):
            print(f"[WARNING] JSON 폴더가 없습니다: {json_folder}")
            continue

        # _merged.json 파일 접근
        merged_json_path = os.path.join(json_folder, f"{label_folder}_merged.json")
        if not os.path.exists(merged_json_path):
            print(f"[WARNING] _merged.json 파일이 없습니다: {merged_json_path}")
            continue

        try:
            with open(merged_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # 데이터 구조에 맞게 파싱
                for image_name, frames in data.items():
                    if not isinstance(frames, list):  # frames가 리스트인지 확인
                        print(f"[INFO] Invalid frames structure for image: {image_name}")
                        continue

                    for frame in frames:
                        features = []
                        for point in frame:
                            if isinstance(point, dict) and all(k in point for k in ["x", "y", "z"]):
                                features.extend([point["x"], point["y"], point["z"]])

                        if features:  # 좌표 데이터가 존재할 경우
                            X.append(features)
                            y.append(label_folder)
                        else:
                            print(f"[INFO] Empty features for image: {image_name}")

        except Exception as e:
            print(f"JSON 파일 로드 실패: {merged_json_path}\n오류: {str(e)}")

    # 모든 샘플의 길이를 동일하게 맞춤 (padding)
    max_length = int(np.percentile([len(sample) for sample in X], 95))  # 95th Percentile
    X_padded = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')

    print(f"총 데이터 샘플: {len(X_padded)}")
    return np.array(X_padded), np.array(y)


def main():
    # 데이터 로드
    X, y = load_json_data()
    print(f"데이터 로드 완료: {len(X)} 샘플")

    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 데이터 불균형 해결 (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost 모델 학습
    model = XGBClassifier(
        max_depth=10,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train_scaled, y_train_resampled)

    # 모델 평가
    train_accuracy = model.score(X_train_scaled, y_train_resampled)
    test_accuracy = model.score(X_test_scaled, y_test)

    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")

    # Confusion Matrix
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    print("\n분류 보고서:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 모델 및 도구 저장
    joblib.dump(model, "gesture_model_xgboost.joblib")
    joblib.dump(scaler, "gesture_scaler_xgboost.joblib")
    joblib.dump(label_encoder, "gesture_label_encoder_xgboost.joblib")
    print("모델 저장 완료!")


if __name__ == "__main__":
    main()
