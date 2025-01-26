import os
import json
import cv2
import numpy as np
import mediapipe as mp

# 경로 설정
output_image_path = os.path.abspath("AI/dataset/output_image")
output_json_path = os.path.abspath("AI/dataset/output_json")

# Mediapipe 초기화
mp_hands = mp.solutions.hands

def extract_landmarks(image_path):
    """이미지에서 손 랜드마크를 추출하는 함수"""
    try:
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1,    min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as hands:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[ERROR] 이미지 로드 실패: {image_path}")
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
    except Exception as e:
        print(f"[ERROR] 랜드마크 추출 실패: {image_path}, {str(e)}")
    return None

# 폴더 순회
for folder_name in os.listdir(output_image_path):
    image_folder_path = os.path.join(output_image_path, folder_name)
    json_folder_path = os.path.join(output_json_path, folder_name)
    merged_folder_path = os.path.join(json_folder_path, f"{folder_name}_json")

    # 병합 폴더 생성
    if not os.path.exists(merged_folder_path):
        os.makedirs(merged_folder_path)

    # 병합 데이터 초기화
    merged_data = {}
    missing_images = []
    missing_jsons = []

    # 이미지 파일 순회
    if os.path.isdir(image_folder_path):
        for image_file in os.listdir(image_folder_path):
            if image_file.endswith(".jpg"):
                image_name = image_file
                json_found = False

                # JSON 데이터 확인
                if os.path.exists(json_folder_path):
                    for json_file in os.listdir(json_folder_path):
                        if json_file.endswith(".json"):
                            json_file_path = os.path.join(json_folder_path, json_file)
                            with open(json_file_path, "r", encoding="utf-8") as f:
                                try:
                                    json_data = json.load(f)

                                    # JSON 데이터가 딕셔너리인지 리스트인지 확인
                                    if isinstance(json_data, dict) and image_name in json_data:
                                        merged_data[image_name] = json_data[image_name]
                                        json_found = True
                                    elif isinstance(json_data, list):
                                        for entry in json_data:
                                            if entry.get("filename") == image_name:
                                                merged_data[image_name] = entry.get("landmarks", [])
                                                json_found = True
                                                break
                                except json.JSONDecodeError:
                                    print(f"[ERROR] JSON 파일 로드 실패: {json_file_path}")

                # Mediapipe로 랜드마크 추출 및 경서님 JSON 생성
                if not json_found:
                    image_path = os.path.join(image_folder_path, image_name)
                    landmarks = extract_landmarks(image_path)
                    if landmarks:
                        print(f"[INFO] 랜드마크 추출 성공: {image_name}")
                        merged_data[image_name] = [landmarks]
                        
                        # 경서님 JSON 파일 생성
                        kyoungseo_json_path = os.path.join(json_folder_path, f"{folder_name}_목경서.json")
                        if not os.path.exists(kyoungseo_json_path):
                            with open(kyoungseo_json_path, "w", encoding="utf-8") as f:
                                json.dump({}, f, indent=4, ensure_ascii=False)

                        # 경서님의 JSON 파일에 데이터 추가
                        with open(kyoungseo_json_path, "r+", encoding="utf-8") as f:
                            kyoungseo_data = json.load(f)
                            kyoungseo_data[image_name] = [landmarks]
                            f.seek(0)
                            json.dump(kyoungseo_data, f, indent=4, ensure_ascii=False)
                            f.truncate()
                    else:
                        print(f"[WARNING] 랜드마크 추출 실패: {image_name}")
                        missing_images.append(image_name)

    # 병합된 JSON 데이터 저장
    merged_file_path = os.path.join(merged_folder_path, f"{folder_name}_merged.json")
    with open(merged_file_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] 병합 완료: {merged_file_path}")
    print(f"[INFO] 누락된 이미지: {len(missing_images)}개")
    print("-" * 50)
