import os
import json
import cv2
import mediapipe as mp

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands

# 경로 설정
output_image_path = "AI/dataset/output_image"
output_json_path = "AI/dataset/output_json"
failed_txt_path = "AI/dataset/failed_images.txt"

def crop_to_center(image, target_size=(640, 480)):
    """이미지의 중심을 기준으로 크롭"""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    target_w, target_h = target_size

    start_x = max(center_x - target_w // 2, 0)
    start_y = max(center_y - target_h // 2, 0)
    end_x = min(center_x + target_w // 2, w)
    end_y = min(center_y + target_h // 2, h)

    return image[start_y:end_y, start_x:end_x]

def enhance_contrast(image):
    """이미지 대비를 조정 (CLAHE 적용)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def resize_image(image, target_size=(640, 480)):
    """이미지를 지정된 크기로 리사이즈"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def add_padding(image, target_size=(640, 480), color=(255, 255, 255)):
    """이미지에 패딩 추가"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    delta_w = max(target_w - w, 0)
    delta_h = max(target_h - h, 0)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def preprocess_image(image_path):
    """이미지를 강력하게 전처리"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 이미지 로드 실패: {image_path}")
        return None

    # 중심 크롭
    cropped = crop_to_center(image)

    # 대비 향상
    enhanced = enhance_contrast(cropped)

    # 리사이즈
    resized = resize_image(enhanced)

    # 패딩 추가
    final_image = add_padding(resized)
    return final_image

def preprocess_and_detect(image_path):
    """이미지 전처리 후 Mediapipe로 감지 시도"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 이미지 로드 실패: {image_path}")
        return None

    # 전처리 단계
    cropped = crop_to_center(image)
    enhanced = enhance_contrast(cropped)
    resized = resize_image(enhanced)
    padded = add_padding(resized)

    # Mediapipe 감지
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
        else:
            print(f"[WARNING] Mediapipe 손 랜드마크 감지 실패: {image_path}")
            return None


# 손 랜드마크 추출 함수
def extract_landmarks(image_path, min_confidence=0.3):
    """이미지에서 손 랜드마크를 추출하는 함수"""
    if not os.path.exists(image_path):
        print(f"[ERROR] 이미지 경로가 잘못되었습니다: {image_path}")
        return None

    landmarks = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=min_confidence) as hands:
        # 이미지 전처리
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return None

        # BGR -> RGB 변환
        image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

        # Mediapipe로 손 랜드마크 추출
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks.append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark])
            print(f"[INFO] 손 랜드마크 감지 성공: {image_path}")
        else:
            print(f"[WARNING] Mediapipe 손 랜드마크 감지 실패: {image_path}")

    return landmarks if landmarks else None

# 실패 이미지 기록
def log_failed_image(image_path, failed_list):
    """실패한 이미지를 리스트에 추가"""
    failed_list.append(image_path)
    print(f"[WARNING] 랜드마크 추출 실패: {image_path}")

# 실패 이미지 저장
def save_failed_log(failed_list):
    """실패 로그를 TXT 파일로 저장"""
    with open(failed_txt_path, "w", encoding="utf-8") as f:
        for image_path in failed_list:
            f.write(image_path + "\n")
    print(f"[INFO] 실패 로그 저장 완료: {failed_txt_path}")

# 이미지 처리
def process_images():
    """이미지를 처리하고 JSON 업데이트"""
    failed_list = []
    for folder_name in os.listdir(output_json_path):
        merged_json_path = os.path.join(output_json_path, folder_name, f"{folder_name}_json", f"{folder_name}_merged.json")
        image_folder_path = os.path.join(output_image_path, folder_name)

        if os.path.exists(merged_json_path):
            with open(merged_json_path, "r", encoding="utf-8") as f:
                merged_data = json.load(f)

            for image_file in os.listdir(image_folder_path):
                if image_file.endswith(".jpg") and image_file not in merged_data:
                    image_path = os.path.join(image_folder_path, image_file)
                    landmarks = extract_landmarks(image_path)
                    if landmarks:
                        merged_data[image_file] = landmarks
                        print(f"[INFO] 랜드마크 추출 성공: {image_file}")
                    else:
                        log_failed_image(image_path, failed_list)

            with open(merged_json_path, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, indent=4, ensure_ascii=False)
                print(f"[INFO] JSON 업데이트 완료: {merged_json_path}")
        else:
            print(f"[WARNING] _merged.json 파일이 없습니다: {merged_json_path}")

    save_failed_log(failed_list)

# 실행
process_images()
