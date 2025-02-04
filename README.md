
---

# Sign Language Recognition System

This project introduces a complete framework for collecting, training, and testing a sign language recognition system Each module has been carefully designed to ensure robust data collection, efficient model training, and real-time gesture recognition using a webcam interface

---

## Key Features

### 1. Data Collection Module (`making_dataset.py`)
- Uses MediaPipe for hand landmark detection
- Supports data collection for simple gestures (e.g., ㄱ, ㅏ), double consonants (e.g., ㄲ, ㅆ), and double vowels (e.g., ㅘ, ㅙ)
- Configurable for varying dataset sizes, such as increasing data collection for ㅜ gestures to improve accuracy
- Automatically organizes data into gesture-specific folders and saves them as .npy files
- Provides flexibility to handle special gestures and allows dynamic user input for labels

---

### 2. Data Preprocessing Module (`preprocessing_data.py`)
- Normalizes all gesture sequences to a fixed frame length of 30 frames
- Includes data augmentation like flipping gestures such as ㅜ to improve model robustness
- Automatically splits data into training and validation sets and saves processed files for future use

---

### 3. Model Training Module (`model_training.py`)
- Implements a Bidirectional LSTM combined with Conv1D to extract both spatial and temporal features
- Features a custom F1-score metric to monitor performance on imbalanced datasets
- Configured with early stopping, learning rate reduction, and model checkpoint saving for best performance
- Reduces computational overhead by applying sliding window techniques for sequence handling

---

### 4. Model Evaluation Module (`model_test.py` & `simplemodeltest.py`)
- Visualizes training progress with graphs for loss, accuracy, and F1-score
- Allows quick evaluation of trained models using saved test datasets

---

### 5. Real-Time Recognition Module (`model_test_webcam.py`)
- Provides an interactive PyQt6-based user interface for real-time gesture recognition
- Features a quiz mode for users to practice recognizing gestures dynamically
- Includes timer-based feedback for incorrect or delayed responses
- Displays gesture predictions with accuracy percentages and tracks user progress

---

### 6. Testing and Debugging Module (`test_saved_data.py` & `webcam_data_capture.py`)
- Enables testing of pre-collected gesture datasets
- Provides utilities for debugging predictions on complex gestures like ㄲ and ㅙ
- Includes tools to collect new gesture data directly from a webcam

---

## Installation

1 Clone this repository
   ```bash
   git clone https://github.com/GDG-SignED/AI.git
   ```
2 Navigate to the project directory
   ```bash
   cd AI
   ```
3 Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Collection
Run the data collection script to start capturing gesture data
```bash
python making_dataset.py
```

### 2. Preprocess Data
Preprocess the collected data for model training
```bash
python preprocessing_data.py
```

### 3. Train the Model
Train the gesture recognition model
```bash
python model_training.py
```

### 4. Evaluate the Model
Evaluate the trained model
```bash
python model_test.py
```

### 5. Real-Time Gesture Recognition
Run the real-time recognition system
```bash
python model_test_webcam.py
```

---

## Project Structure

```
AI_NEW/
│
├── dataset/                  # Original dataset containing images, JSON, and npy files
│   ├── output_image/         # Gesture image files
│   ├── output_json/          # JSON label files
│   ├── output_npy/           # Processed npy files
│
├── model/                    # Saved models and logs
│   ├── Sign_ED_best.keras    # Best-trained model
│   ├── history.pkl           # Training history
│   ├── model_summary.txt     # Model architecture summary
│
├── Sign_ED/                  # Core scripts for model evaluation and testing
│   ├── test/                 # Testing utilities
│   │   ├── testdata/         # Test datasets
│   │   ├── model_test_webcam.py # Real-time recognition script
│   │   ├── simplemodeltest.py   # Quick model test script
│   │   ├── test_saved_data.py   # Pre-collected dataset testing
│   │   ├── webcam_data_capture.py # Webcam-based data collection
│   └── making_dataset.py     # Data collection script
│
├── suyoun_dataset/           # Updated gesture dataset (customized)
│   ├── ㄱ/                   # Data for gesture ㄱ
│   ├── ㄴ/                   # Data for gesture ㄴ
│   ├── ...                   # Additional gesture-specific folders
│
├── model_test.py             # Model evaluation script
├── model_training.py         # Model training script
├── preprocessing_data.py     # Data preprocessing script
└── README.md                 # Project documentation
```

---

## Notes on `dataset` and `suyoun_dataset`

- `dataset/` contains the original dataset with images, JSON labels, and `.npy` files that were merged from the `main` branch. This dataset is preserved to avoid losing important information
- `suyoun_dataset/` is the customized dataset used for training and testing, where gestures are organized into separate folders
- Ensure that modifications to either dataset are clearly documented and updated in their respective directories

---

## License

This project is licensed under the MIT License

---

## Contact

For questions or support, please contact
- **GitHub:** [GDG-SignED](https://github.com/GDG-SignED)

---

