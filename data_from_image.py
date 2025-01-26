import cv2
import mediapipe as mp
import json
import os
import traceback
from PIL import Image
import numpy as np

def extract_landmarks_from_images(input_dir="dataset/output_image"):
    try:
        print("Starting landmark extraction process...")
        print(f"Input directory: {input_dir}")
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
        print("Initializing MediaPipe...")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, 
                             max_num_hands=1, 
                             min_detection_confidence=0.5)

        all_landmarks = {}
        folders = os.listdir(input_dir)
        print(f"Found folders: {folders}")
        
        for char_folder in folders:
            char_path = os.path.join(input_dir, char_folder)
            print(f"\nProcessing folder: {char_folder}")
            
            if not os.path.isdir(char_path):
                print(f"Skipping {char_folder} as it's not a directory")
                continue
                
            all_landmarks[char_folder] = []
            files = os.listdir(char_path)
            print(f"Found {len(files)} files in {char_folder}")
            
            processed_count = 0
            error_count = 0
            
            for file in files:
                if file.endswith('.jpg'):
                    input_path = os.path.join(char_path, file)
                    
                    try:
                        if not os.path.exists(input_path):
                            print(f"File does not exist: {input_path}")
                            continue
                            
                        # PIL을 사용하여 이미지 읽기 시도
                        try:
                            pil_image = Image.open(input_path)
                            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                            pil_image.close()
                        except Exception as e:
                            print(f"Error reading image with PIL: {input_path}")
                            print(f"Error details: {str(e)}")
                            continue
                            
                        if image is None:
                            print(f"Cannot read image (returned None): {input_path}")
                            print(f"File size: {os.path.getsize(input_path)} bytes")
                            continue
                            
                        print(f"Processing {file} - Image shape: {image.shape}")
                            
                        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                        if results.multi_hand_landmarks:
                            frame_landmarks = []
                            for hand_landmarks in results.multi_hand_landmarks:
                                landmarks = [
                                    {
                                        "x": landmark.x,
                                        "y": landmark.y,
                                        "z": landmark.z
                                    }
                                    for landmark in hand_landmarks.landmark
                                ]
                                frame_landmarks.append({"landmarks": landmarks})
                                
                            all_landmarks[char_folder].extend(frame_landmarks)
                            processed_count += 1
                        else:
                            print(f"No hand landmarks detected in {file}")
                        
                    except Exception as e:
                        error_count += 1
                        print(f"Error processing {input_path}:")
                        print(traceback.format_exc())
                        continue
            
            print(f"\nFolder {char_folder} summary:")
            print(f"Successfully processed: {processed_count} images")
            print(f"Errors encountered: {error_count} images")
            
            if all_landmarks[char_folder]:
                try:
                    output_dir = "dataset/output_json"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_path = os.path.join(output_dir, char_folder)
                    os.makedirs(output_path, exist_ok=True)
                    
                    json_path = os.path.join(output_path, f"{char_folder}_landmarks.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(all_landmarks[char_folder], f, ensure_ascii=False, indent=4)
                    
                    print(f"Successfully saved landmarks for {char_folder} to {json_path}")
                
                except Exception as e:
                    print(f"Error saving JSON for {char_folder}:")
                    print(traceback.format_exc())
        
        hands.close()
        print("\nLandmark extraction process completed!")
        return all_landmarks

    except Exception as e:
        print("Critical error in extraction process:")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    try:
        print("Starting main process...")
        print(f"Current working directory: {os.getcwd()}")
        
        input_dir = "dataset/output_image"
        if os.path.exists(input_dir):
            print(f"Input directory contents: {os.listdir(input_dir)}")
        else:
            print(f"Input directory {input_dir} not found")
        
        landmarks = extract_landmarks_from_images()
        
        if landmarks is not None:
            print("Process completed successfully!")
            print(f"Processed {len(landmarks)} character folders")
        else:
            print("Process failed!")
            
    except Exception as e:
        print("Critical error in main process:")
        print(traceback.format_exc())