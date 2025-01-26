import cv2
import numpy as np
import os

file_path = "AI/dataset/output_image/ㅕ/ㅕ_이수연_56.jpg"
try:
    # 파일 경로를 UTF-8로 변환
    normalized_path = os.path.abspath(file_path)
    img = cv2.imdecode(np.fromfile(normalized_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"[ERROR] 이미지 로드 실패: {normalized_path}")
    else:
        print(f"[SUCCESS] 이미지 로드 성공: {normalized_path}")
except Exception as e:
    print(f"[ERROR] 이미지 처리 중 에러 발생: {e}")
