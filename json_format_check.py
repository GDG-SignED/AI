import os
import json
import numpy as np

def inspect_json_files(data_dir="AI/dataset/output_json"):
    """
    JSON 파일을 검사하여 오류가 있는 데이터 구조를 출력
    :param data_dir: JSON 파일들이 있는 상위 폴더 경로
    """
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):  # 폴더가 아니면 무시
            continue

        json_folder = os.path.join(label_path, f"{label_folder}_json")
        if not os.path.exists(json_folder):
            print(f"[WARNING] JSON 폴더가 없습니다: {json_folder}")
            continue

        merged_json_path = os.path.join(json_folder, f"{label_folder}_merged.json")
        if not os.path.exists(merged_json_path):
            print(f"[WARNING] _merged.json 파일이 없습니다: {merged_json_path}")
            continue

        try:
            with open(merged_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for image_name, frames in data.items():
                    if not isinstance(frames, list):
                        print(f"[ERROR] '{image_name}'의 frames가 리스트가 아닙니다. 파일 경로: {merged_json_path}")
                        continue

                    for i, frame_set in enumerate(frames):
                        if not isinstance(frame_set, list):
                            print(f"[ERROR] '{image_name}'의 frame_set[{i}]가 리스트가 아닙니다. 파일 경로: {merged_json_path}")
                            print(f"frame_set 내용: {frame_set}")
                            continue

                        for j, frame in enumerate(frame_set):
                            if not isinstance(frame, dict):
                                print(f"[ERROR] '{image_name}'의 frame_set[{i}][{j}]가 딕셔너리가 아닙니다. 내용: {frame}")
                                print(f"파일 경로: {merged_json_path}")
                            elif not all(k in frame for k in ["x", "y", "z"]):
                                print(f"[ERROR] '{image_name}'의 frame_set[{i}][{j}]에 키가 누락되었습니다. 내용: {frame}")
                                print(f"파일 경로: {merged_json_path}")

        except json.JSONDecodeError:
            print(f"[ERROR] JSON 파일 파싱 실패: {merged_json_path}")
        except Exception as e:
            print(f"[ERROR] JSON 파일 검사 중 예외 발생: {merged_json_path}")
            print(f"오류 메시지: {str(e)}")

def load_json_data(data_dir="AI/dataset/output_json"):
    """
    JSON 데이터를 로드하여 X, y 배열을 반환
    :param data_dir: JSON 파일들이 있는 상위 폴더 경로
    :return: X, y 배열
    """
    X, y = [], []
    
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        json_folder = os.path.join(label_path, f"{label_folder}_json")
        if not os.path.exists(json_folder):
            continue

        merged_json_path = os.path.join(json_folder, f"{label_folder}_merged.json")
        if not os.path.exists(merged_json_path):
            continue

        try:
            with open(merged_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for image_name, frames in data.items():
                    if isinstance(frames, list):
                        X.append(frames)
                        y.append(label_folder)
        except json.JSONDecodeError:
            print(f"[ERROR] JSON 파일 파싱 실패: {merged_json_path}")
        except Exception as e:
            print(f"[ERROR] JSON 파일 처리 중 예외 발생: {merged_json_path}")
            print(f"오류 메시지: {str(e)}")
    
    # 데이터 구조 점검
    print(f"X 데이터 샘플: {X[:2]}")
    print(f"y 데이터 샘플: {y[:2]}")
    return np.array(X, dtype=object), np.array(y)

def main():
    # JSON 검사 실행
    inspect_json_files()

    # JSON 데이터 로드
    try:
        X, y = load_json_data()
        print(f"X의 크기: {len(X)}")
        print(f"y의 크기: {len(y)}")
        print(f"X 샘플 데이터 타입: {type(X[0])}")
    except ValueError as e:
        print(f"[ERROR] 데이터 변환 중 에러 발생: {e}")
    except Exception as e:
        print(f"[ERROR] 예기치 못한 에러 발생: {e}")

if __name__ == "__main__":
    main()
