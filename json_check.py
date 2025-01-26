import os 
import json
import cv2

# 경로 설정
output_image_path = "AI/dataset/output_image"
output_json_path = "AI/dataset/output_json"

# json, image가 서로 연결되어 있는지 체크하기 
def check_merged_json_and_images():
    """각 자모음 폴더의 _merged.json과 이미지의 유효성을 확인"""
    for folder_name in os.listdir(output_json_path):  # 자모음 폴더 순회
        merged_json_path = os.path.join(output_json_path, folder_name, f"{folder_name}_json", f"{folder_name}_merged.json")
        image_folder_path = os.path.join(output_image_path, folder_name)

        # _merged.json 파일 확인
        if os.path.exists(merged_json_path):
            print(f"\n[INFO] _merged.json 파일 확인 중: {merged_json_path}")
            try:
                with open(merged_json_path, "r", encoding="utf-8") as f:
                    merged_data = json.load(f)

                    # JSON 데이터가 딕셔너리인지 확인
                    if not isinstance(merged_data, dict):
                        print(f"[ERROR] _merged.json의 데이터가 딕셔너리가 아닙니다: {merged_json_path}")
                        continue

                    # JSON 데이터의 키(이미지 파일명)와 실제 이미지 파일 비교
                    for image_name in merged_data.keys():
                        image_path = os.path.join(image_folder_path, image_name)
                        if not os.path.exists(image_path):
                            print(f"[WARNING] JSON에 존재하지만 이미지가 없는 파일: {image_name}")

                    # 이미지 폴더의 파일과 JSON 데이터 키 비교
                    for image_file in os.listdir(image_folder_path):
                        if image_file.endswith(".jpg"):
                            if image_file not in merged_data:
                                print(f"[WARNING] 이미지 폴더에 존재하지만 JSON에 없는 파일: {image_file}")

            except json.JSONDecodeError:
                print(f"[ERROR] _merged.json 파일 로드 실패: {merged_json_path}")
        else:
            print(f"[WARNING] _merged.json 파일이 없습니다: {merged_json_path}")


# 함수 실행
check_merged_json_and_images()

