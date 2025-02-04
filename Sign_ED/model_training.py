import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, Masking, TimeDistributed, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import pickle # 나중에 추가 ㅜㅜ

# 제스처 클래스 정의 
actions = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
           'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㅘ', 'ㅙ', 'ㅞ', 'ㅝ']

# 데이터 로드
x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')
x_val = np.load('./dataset/x_val.npy')
y_val = np.load('./dataset/y_val.npy')

# ✅ sequence_length 적용
sequence_length = 15  # 기존 30 → 15로 변경

# ✅ 30프레임 중 15프레임씩 슬라이딩 윈도우 적용
def sliding_window(data, seq_length):
    new_sequences = []
    for seq in data:
        for i in range(len(seq) - seq_length + 1):
            new_sequences.append(seq[i:i+seq_length])
    return np.array(new_sequences, dtype='float32')

# ✅ 학습 데이터 변환
x_train = sliding_window(x_train, sequence_length)
x_val = sliding_window(x_val, sequence_length)

# ✅ y_train, y_val도 슬라이딩 적용 후 데이터 개수 맞추기
y_train = np.repeat(y_train, x_train.shape[0] // y_train.shape[0], axis=0)
y_val = np.repeat(y_val, x_val.shape[0] // y_val.shape[0], axis=0)

print(f"🔍 x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
print(f"🔍 x_val.shape: {x_val.shape}, y_val.shape: {y_val.shape}")


# F1-score 커스텀 지표 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    F1score = 2 * precision * recall / (precision + recall + 1e-7)

    return F1score

# 모델 정의
model = Sequential([
    # 패딩된 데이터 무시
    Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2])),

    # Conv1D 추가 (Temporal CNN) → 시퀀스의 공간적 특징 추출
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),

    # Conv1D 출력이 3D 형태로 유지되도록 TimeDistributed 적용
    TimeDistributed(Flatten()),  # Flatten()을 시퀀스 단위로 적용

    # Bidirectional LSTM → 시간 정보를 학습
    Bidirectional(LSTM(64, return_sequences=False)),  # 최종 출력은 (batch_size, 64)
    Dropout(0.3),

    # Fully Connected Layer
    Dense(32, activation='relu'),
    Dropout(0.3),

    # 최종 Softmax Layer
    Dense(len(actions), activation='softmax')
])

# 모델 컴파일 (F1-score 추가됨)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 학습 콜백 설정 (`val_acc` 사용)
early_stopping = EarlyStopping(monitor='val_acc', patience=20, mode='max')
callbacks = [
    ModelCheckpoint('./model/Sign_ED_best.keras', monitor='val_acc', save_best_only=True),
    ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10),  # val_acc 기준으로 학습률 감소
    early_stopping
]

# 모델 학습
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks
)

# 모델 저장 ( keras 로 변경 )
model.save("./model/Sign_ED.keras")

# 모델 구조 확인
model.summary()

# 모델 요약 정보를 파일에 저장
with open("./model/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# 학습 후 history 저장 - 나중에 추가 ㅜㅜ
with open('./model/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
