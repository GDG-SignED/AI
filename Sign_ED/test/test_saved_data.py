import numpy as np
import tensorflow as tf
import os

# 모델 로드
model = tf.keras.models.load_model("./model/Sign_ED_best.keras")

# 저장된 데이터 경로
dataset_path = "./Sign_ED/test/testdata"

# 제스처 클래스 리스트 추가 (오류 해결)
actions = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
           'ㅘ', 'ㅙ', 'ㅝ', 'ㅞ']

# 쌍자음/이중모음 규칙
double_consonants = {'ㄲ': ['ㄱ', 'ㄱ'], 'ㄸ': ['ㄷ', 'ㄷ'], 'ㅃ': ['ㅂ', 'ㅂ'], 'ㅆ': ['ㅅ', 'ㅅ'], 'ㅉ': ['ㅈ', 'ㅈ']}
double_vowels = {'ㅘ': ['ㅗ', 'ㅏ'], 'ㅙ': ['ㅗ', 'ㅐ'], 'ㅝ': ['ㅜ', 'ㅓ'], 'ㅞ': ['ㅜ', 'ㅔ']}

# 저장된 데이터 파일 확인
files = [f for f in os.listdir(dataset_path) if f.startswith("test_landmarks_")]
for file in files:
    label = file.replace("test_landmarks_", "").replace(".npy", "")  # 라벨 추출
    test_data = np.load(os.path.join(dataset_path, file))  # 저장된 데이터 로드
    test_data = np.expand_dims(test_data, axis=0)  # 모델 입력 크기 맞추기

    # 모델 예측
    predictions = model.predict(test_data)[0]
    max_index = np.argmax(predictions)
    predicted_label = actions[max_index]  # 🔥 오류 해결됨!
    confidence = predictions[max_index] * 100

    # 결과 출력
    print(f"📌 저장된 데이터 라벨: {label}")
    print(f"✅ 모델 예측 결과: {predicted_label} ({confidence:.2f}%)")

    # 쌍자음/이중모음 테스트 결과 확인
    if label in double_consonants or label in double_vowels:
        print(f"⚠️ {label}은 쌍자음/이중모음입니다!")
        if label in double_consonants:
            components = double_consonants[label]
        elif label in double_vowels:
            components = double_vowels[label]

        print(f"구성 요소: {components}")
        # 각 구성 요소의 예측 확인 (디버깅용)
        for i, component in enumerate(components):
            component_data = np.load(os.path.join(dataset_path, f"test_landmarks_{component}.npy"))
            component_data = np.expand_dims(component_data, axis=0)
            component_predictions = model.predict(component_data)[0]
            component_max_index = np.argmax(component_predictions)
            component_label = actions[component_max_index]
            component_confidence = component_predictions[component_max_index] * 100
            print(f"  🔹 {i+1}번째 제스처 {component} → 예측: {component_label} ({component_confidence:.2f}%)")

    print("-" * 50)
