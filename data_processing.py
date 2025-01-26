import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 디렉토리 설정
input_dir = "AI/dataset/output_image"  # 이미지 폴더 경로
output_dir = "AI/dataset/output_json"  # JSON 저장 경로
os.makedirs(output_dir, exist_ok=True)

def sanitize_filename(filename):
    """특수문자를 제거하여 파일명을 정리합니다."""
    return "".join(c for c in filename if c.isalnum() or c in ("_", "-"))

def preprocess_image(image_path):
    """이미지 전처리: 크기 정규화 및 RGB 변환"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {image_path}")
        return None

    # 이미지 크기 정규화 (640x480)
    image_resized = cv2.resize(image, (640, 480))

    # BGR -> RGB 변환
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return image_rgb

def extract_landmarks(image_rgb):
    """이미지에서 랜드마크 추출"""
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 데이터를 리스트로 변환
            landmarks.append([{
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            } for lm in hand_landmarks.landmark])
        return landmarks
    return None

def process_images():
    """이미지에서 랜드마크 추출 후 JSON 파일 저장"""
    missing_landmarks_log = []

    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            print(f"[DEBUG] Skipping non-directory: {label_dir}")
            continue

        json_label_dir = os.path.join(output_dir, label)
        os.makedirs(json_label_dir, exist_ok=True)

        for image_file in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
            if not image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                print(f"[DEBUG] Skipping non-image file: {image_file}")
                continue

            # 파일명 정리
            clean_image_file = sanitize_filename(image_file)
            print(f"[DEBUG] Original filename: {image_file}, Cleaned filename: {clean_image_file}")

            image_path = os.path.join(label_dir, image_file)
            json_file = os.path.splitext(clean_image_file)[0] + ".json"
            json_path = os.path.join(json_label_dir, json_file)

            # 기존 JSON이 있으면 건너뜀
            if os.path.exists(json_path):
                print(f"[INFO] JSON already exists, skipping: {json_path}")
                continue

            # 이미지 전처리
            image_rgb = preprocess_image(image_path)
            if image_rgb is None:
                missing_landmarks_log.append(image_path)
                continue

            # 랜드마크 추출
            landmarks = extract_landmarks(image_rgb)
            if landmarks is not None:
                # JSON 저장
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({clean_image_file: landmarks}, f, ensure_ascii=False, indent=4)
            else:
                missing_landmarks_log.append(image_path)

    # 누락된 데이터 로그 출력
    print("\n[랜드마크 추출 실패 로그]")
    for path in missing_landmarks_log:
        print(f"- {path}")

if __name__ == "__main__":
    process_images()
    hands.close()
    print("[INFO] 전처리 완료!")
