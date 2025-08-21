# server.py
import hand_detector2 as hdm
import cv2
import numpy as np
import pickle
import pandas as pd
from io import BytesIO
from PIL import Image
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------- Model load / fallback train ----------------
def load_or_train_model():
    try:
        with open("word_model.pkl", "rb") as f:
            m = pickle.load(f)
        print("Loaded model from word_model.pkl")
        return m
    except FileNotFoundError:
        print("word_model.pkl not found. Training RandomForest from sign_language.csv ...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        df = pd.read_csv("sign_language.csv")
        # drop any accidental index cols
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        print(f"Trained model. Accuracy: {acc:.3f}")
        # Optional: save (Render's disk is ephemeral, but helpful locally)
        with open("word_model.pkl", "wb") as f:
            pickle.dump(m, f)
        return m

model = load_or_train_model()
expected_features = model.n_features_in_

# --------------- MediaPipe hand detector ----------------
detector = hdm.handDetector()

# --------------- API app + CORS ----------------
app = FastAPI(title="Sign Language Word API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # lock this down later to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Session buffer for sequences ----------------
# each session_id -> list of 84-d frame feature vectors
SEQUENCE_LEN = 15
sessions: Dict[str, List[np.ndarray]] = {}

# --------------- helpers ----------------
def frame_features_from_image(rgb_np: np.ndarray) -> np.ndarray:
    """Return 84-d feature vector [Left(42) + Right(42)] from a single RGB image."""
    # convert to BGR for cv2 functions used inside your detector
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    bgr = detector.find_hands(bgr, draw=False)
    landmarks = detector.find_position(bgr)

    left_hand = [0] * 42
    right_hand = [0] * 42

    if landmarks:
        for handedness, lmlist in landmarks:
            # flatten (x,y) for the 21 landmarks -> 42 numbers
            loc_vec = [coord for lm in lmlist for coord in lm[1:3]]
            if handedness == "Left":
                left_hand = loc_vec
            elif handedness == "Right":
                right_hand = loc_vec

    combined = np.array(left_hand + right_hand, dtype=np.float32)  # shape (84,)
    return combined

def align_features(vec: np.ndarray) -> np.ndarray:
    """Pad/truncate to match model.n_features_in_."""
    x = vec.reshape(1, -1)
    n = x.shape[1]
    if n < expected_features:
        x = np.pad(x, ((0,0),(0, expected_features - n)), mode="constant")
    elif n > expected_features:
        x = x[:, :expected_features]
    return x

# --------------- endpoints ----------------
@app.get("/")
def root():
    return {"ok": True, "message": "API running. POST /predict_frame with image + session_id."}

@app.post("/predict_frame")
async def predict_frame(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Send one frame at a time with a session_id.
    Server buffers 15 frames, predicts, returns JSON, then clears the buffer.
    """
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    rgb_np = np.array(image)

    feat84 = frame_features_from_image(rgb_np)

    buf = sessions.get(session_id, [])
    buf.append(feat84)
    sessions[session_id] = buf

    if len(buf) < SEQUENCE_LEN:
        return JSONResponse({
            "sequence_len": len(buf),
            "sequence_needed": SEQUENCE_LEN,
            "sequence_complete": False
        })

    # have 15 frames -> make 1260-d vector
    seq = np.stack(buf[:SEQUENCE_LEN], axis=0).reshape(-1)  # (15,84) -> (1260,)
    sessions[session_id] = []  # reset

    X = align_features(seq)
    pred = model.predict(X)[0]
    # Optionally: proba = model.predict_proba(X).max()

    return JSONResponse({
        "sequence_len": SEQUENCE_LEN,
        "sequence_complete": True,
        "prediction": pred
    })

@app.post("/reset_session")
async def reset_session(session_id: str = Form(...)):
    sessions[session_id] = []
    return {"ok": True, "message": f"session {session_id} cleared"}

# --------------- local run ----------------
if __name__ == "__main__":
    # Render will run via the start command; this is only for local testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
