import matplotlib.pyplot as plt  # 시각화 위한 matplotlib 임포트
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle

# history 불러오기
with open('./model/history.pkl', 'rb') as f:
    history = pickle.load(f)

# 그래프 출력 등 활용 가능
print(history['accuracy'])  # 학습 정확도 기록

# F1 Score 함수 정의
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score

# 모델 로드
model = load_model('./model/Sign_ED.model', custom_objects={'metric_F1score': metric_F1score})

# Test data load 
x_test = np.load('./dataset/x_test.npy')
y_test = np.load('./dataset/y_test.npy')


# 모델 학습을 수행했을 때 얻은 history 객체를 사용해야 합니다.
# 예시로 여기서는 'history' 객체를 로드했다고 가정합니다.
# history = model.fit(...)

# 손실 함수 및 정확도, F1 score 그래프 그리기
fig, loss_ax = plt.subplots(figsize=(16, 10))  # 16x10 크기의 그래프 그리기
acc_ax = loss_ax.twinx()  # 정확도와 손실을 같은 그래프에 그리기 위해 y축을 공유하는 두 번째 축을 추가

# 손실 그래프
loss_ax.plot(history.history['loss'], 'y', label='train loss')  # 훈련 손실 그래프
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')  # 검증 손실 그래프
loss_ax.set_xlabel('epoch')  # x축 레이블
loss_ax.set_ylabel('loss')  # y축 레이블
loss_ax.legend(loc='upper left')  # 손실 그래프 범례

# 정확도 그래프
acc_ax.plot(history.history['acc'], 'b', label='train acc')  # 훈련 정확도 그래프
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')  # 검증 정확도 그래프
acc_ax.set_ylabel('accuracy')  # y축 레이블
acc_ax.legend(loc='upper left')  # 정확도 그래프 범례

plt.show()  # 그래프 출력

# F1 score 그래프 그리기
fig, loss_ax = plt.subplots(figsize=(16, 10))  # 16x10 크기의 그래프 그리기
acc_ax = loss_ax.twinx()  # 정확도와 F1 score를 같은 그래프에 그리기 위해 y축을 공유하는 두 번째 축을 추가

# F1 score 그래프
acc_ax.plot(history.history['metric_F1score'], 'b', label='train f1')  # 훈련 F1 score 그래프
acc_ax.plot(history.history['val_metric_F1score'], 'g', label='val f1')  # 검증 F1 score 그래프
acc_ax.set_ylabel('f1-score')  # y축 레이블
acc_ax.legend(loc='upper left')  # F1 score 그래프 범례

plt.show()  # 그래프 출력

# 테스트 데이터에 대한 정확도와 F1 score 추가 시각화
test_acc = model.evaluate(x_test, y_test, verbose=0)  # 테스트 데이터 정확도 평가
y_pred = model.predict(x_test)
test_f1score = metric_F1score(y_test, y_pred)  # 테스트 데이터 F1 score 평가

# 테스트 정확도 및 F1 score 출력
print(f"Test Accuracy: {test_acc[1]*100:.2f}%")
print(f"Test F1 Score: {test_f1score:.2f}")

# 테스트 데이터 정확도와 F1 score를 그래프에 추가
fig, ax = plt.subplots(figsize=(16, 10))

ax.plot(history.history['acc'], 'b', label='train acc')  # 훈련 정확도
ax.plot(history.history['val_acc'], 'g', label='val acc')  # 검증 정확도
ax.axhline(test_acc[1], color='r', linestyle='--', label=f'test acc: {test_acc[1]*100:.2f}%')  # 테스트 정확도
ax.set_xlabel('epoch')  # x축 레이블
ax.set_ylabel('accuracy')  # y축 레이블
ax.legend(loc='upper left')  # 범례

plt.show()  # 그래프 출력

fig, ax = plt.subplots(figsize=(16, 10))

ax.plot(history.history['metric_F1score'], 'b', label='train f1')  # 훈련 F1 score
ax.plot(history.history['val_metric_F1score'], 'g', label='val f1')  # 검증 F1 score
ax.axhline(test_f1score, color='r', linestyle='--', label=f'test f1: {test_f1score:.2f}')  # 테스트 F1 score
ax.set_xlabel('epoch')  # x축 레이블
ax.set_ylabel('f1-score')  # y축 레이블
ax.legend(loc='upper left')  # 범례

plt.show()  # 그래프 출력
