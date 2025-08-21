import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Sign Language Interpreter")
st.write("📷 This is a demo. Webcam access may not work in Streamlit Cloud yet, but we’ll set it up.")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()

while run:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
