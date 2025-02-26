import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# 모델 로드
model = tf.keras.models.load_model("model/Sign_ED_best.keras")

# 제스처 레이블
GESTURES = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
            'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㅘ', 'ㅙ', 'ㅞ', 'ㅝ']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON 데이터 받기
        data = request.get_json()
        input_data = np.array(data["input"])  # (30, 63) 형태

        # 입력 차원 맞추기 (배치 차원 추가)
        input_data = np.expand_dims(input_data, axis=0)  # (1, 30, 63)

        # 모델 예측
        predictions = model.predict(input_data)
        predicted_index = np.argmax(predictions)
        predicted_gesture = GESTURES[predicted_index]

        return jsonify({"gesture": predicted_gesture})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
