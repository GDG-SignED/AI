import os
import json

def process_merged_json_files(base_directory):
    # 모든 지모음 폴더 탐색
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path):  # 폴더인지 확인
            json_folder = os.path.join(folder_path, f"{folder}_json")  # 예: ㄱ_json
            
            # 해당 폴더 내에서 "merged_json" 파일만 처리
            merged_json_path = os.path.join(json_folder, f"{folder}_merged.json")
            if os.path.exists(merged_json_path):
                print(f"Processing: {merged_json_path}")
                with open(merged_json_path, "r", encoding="utf-8") as json_file:
                    try:
                        data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"[ERROR] {merged_json_path} is not a valid JSON file. Skipping.")
                        continue
                
                # 데이터 수정
                modified = False
                for key, value in data.items():
                    # "지모음_조예인" 형식의 키만 처리
                    if key.startswith(f"{folder}_조예인") and isinstance(value, list):
                        for i, frame_set in enumerate(value):
                            if isinstance(frame_set, dict):  # 딕셔너리를 리스트로 변환
                                value[i] = [frame_set]
                                modified = True
                
                # 수정된 데이터 저장
                if modified:
                    with open(merged_json_path, "w", encoding="utf-8") as json_file:
                        json.dump(data, json_file, ensure_ascii=False, indent=4)
                    print(f"Updated: {merged_json_path}")
                else:
                    print(f"No changes needed: {merged_json_path}")

# 지모음 JSON 데이터가 저장된 최상위 디렉토리
base_directory = "AI/dataset/output_json"
process_merged_json_files(base_directory)
