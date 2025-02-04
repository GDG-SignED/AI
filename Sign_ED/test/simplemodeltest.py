import tensorflow as tf
import numpy as np

# 모델 로드
model = tf.keras.models.load_model("./model/Sign_ED_best.keras")  # 또는 Sign_ED.h5

# 데이터 로드
x_val = np.load('./dataset/x_val.npy')
y_val = np.load('./dataset/y_val.npy')

# 검증 데이터 평가
loss, val_accuracy = model.evaluate(x_val, y_val)
print(f"최종 검증 손실: {loss:.4f}")
print(f"최종 검증 정확도: {val_accuracy:.4f}")

# 데이터 로드
x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')

# 훈련 데이터 평가
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print(f"최종 훈련 손실: {train_loss:.4f}")
print(f"최종 훈련 정확도: {train_accuracy:.4f}")

