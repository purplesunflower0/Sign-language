import hand_detector2 as hdm
import cv2
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time
from gtts import gTTS
import io
import pygame
import warnings

warnings.filterwarnings("ignore")

# ==================== Load or Train Model ====================
try:
    # Load saved model if exists
    with open("word_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Loaded trained model from word_model.pkl")

except FileNotFoundError:
    # Train model from dataset if no saved model
    print("Training new RandomForest model...")
    data = pd.read_csv("sign_language.csv")

    # Drop index-like columns if they exist
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))

    # Save for future use
    with open("word_model.pkl", "wb") as f:
        pickle.dump(model, f)

# ==================== Speech Function ====================
def speech(text):
    myobj = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    myobj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# ==================== Main Loop ====================
def main():
    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()

    sequence_length = 15
    current_sequence = []
    inactive_frames = 0
    inactivity_reset_length = 3

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        key = cv2.waitKey(1) & 0xFF

        img = detector.find_hands(img, draw=False)
        landmarks = detector.find_position(img)

        if landmarks:
            left_hand = [0] * 42
            right_hand = [0] * 42

            for hand in landmarks:
                handedness = hand[0]
                lmlist = hand[1]
                location_vector = [coord for lm in lmlist for coord in lm[1:3]]
                if handedness == 'Left':
                    left_hand = location_vector
                elif handedness == 'Right':
                    right_hand = location_vector

            combined_vector = left_hand + right_hand
            current_sequence.append(combined_vector)

            if len(current_sequence) == sequence_length:
                sequence_array = np.array(current_sequence).flatten().reshape(1, -1)

                # === Fix feature mismatch ===
                expected_features = model.n_features_in_
                if sequence_array.shape[1] < expected_features:
                    diff = expected_features - sequence_array.shape[1]
                    sequence_array = np.pad(sequence_array, ((0,0),(0,diff)), mode="constant")
                elif sequence_array.shape[1] > expected_features:
                    sequence_array = sequence_array[:, :expected_features]

                prediction = model.predict(sequence_array)[0]
                print("Predicted word:", prediction)
                speech(prediction)
                current_sequence = []

        else:
            inactive_frames += 1
            if inactive_frames >= inactivity_reset_length:
                current_sequence = []
                inactive_frames = 0

        cv2.imshow("Image", img)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
