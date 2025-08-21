# Real-Time Sign Language Interpreter

Welcome to the Real-Time Sign Language Interpreter! This project uses machine learning and computer vision to recognize and interpret American Sign Language (ASL) gestures in real time. The interpreter currently recognizes individual letters and signed words, providing both on-screen visualization and audio output.

## Key Features

- **Hand Gesture Detection:** The project uses MediaPipe for accurate and efficient hand landmark detection. This allows the system to track hand movements and gestures in real-time using just a webcam.

- **Letter Recognition:** A Logistic Regression model is trained on a custom dataset of hand landmarks to classify individual ASL letters. The system can recognize most static letters of the ASL alphabet.

- **Word Recognition:** A Random Forest Classifier is used to recognize dynamic hand movements corresponding to signed words. The model is trained on sequences of hand movements, allowing it to recognize words that require motion, such as 'hello' and 'help'.

- **Audio Feedback:** The project provides audio output for recognized letters and words using gTTS (Google Text-to-Speech) and pyGame. This allows the system to 'speak' the interpreted signs, making it accessible for both deaf and hearing individuals.

- **Data Collection and Augmentation:** The system allows for real-time data collection and augmentation. Users can capture new hand gestures and automatically add them to the training dataset, facilitating the continuous improvement of the models.

## How It Works

1. **Hand Detection:** The `handDetector` class (located in `hand_detector2.py`) uses MediaPipe to detect and track hand landmarks in real time. The detected landmarks are used as inputs for the letter and word recognition models.

2. **Letter Recognition:** The `letter_interpreter.py` script captures hand landmarks and uses a trained Logistic Regression model to classify individual letters. When a letter is recognized, it is displayed on the screen and added to the current word being spelled. If the same letter is detected consistently, it is added to the word.

3. **Word Recognition:** The `word_interpreter.py` script tracks sequences of hand movements over multiple frames. These sequences are used to classify signed words with the help of a Random Forest Classifier. Recognized words are displayed and can be dictated via audio output.

4. **Audio Output:** The system uses gTTS to convert recognized words into speech. The audio is played using pyGame, making the system interactive and user-friendly.

5. **Data Collection:** The system allows users to collect new gesture data by pressing specific keys. This data is automatically saved and can be used to retrain the models, enhancing the system's recognition capabilities.

## Getting Started
### Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/sign-language-interpreter.git
   ```
2. Start a virtual environment
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

4. Run the interpreter scripts:

   - For letter recognition:
     ```
     python letter_interpreter.py
     ```

   - For word recognition:
     ```
     python word_interpreter.py
     ```

### Usage

- Press `q` to quit the interpreter.
- Press `c` to capture new gesture data for training.

## Future Improvements

- **Full ASL Alphabet:** Incorporate dynamic letters like 'J' and 'Z' to complete the ASL alphabet.
- **Expanded Vocabulary:** Train the system to recognize a broader set of words and phrases.
- **User Profiles:** Implement user profiles to allow personalized gesture models.
- **Mobile Integration:** Develop a mobile app version to make the interpreter portable.

## Acknowledgments

This project is inspired by the desire to make communication more accessible and inclusive for the deaf community.


