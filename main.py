import uvicorn
import io
import base64
import librosa
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# --- 1. CONFIGURATION & MODEL LOADING ---
MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

print(f"Loading Model: {MODEL_ID} ...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Auto-detect label mapping
    fake_id, real_id = 1, 0
    labels = model.config.id2label
    if labels:
        for k, v in labels.items():
            text = str(v).lower()
            if "fake" in text or "spoof" in text or "ai" in text:
                fake_id = int(k)
            elif "real" in text or "human" in text:
                real_id = int(k)

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model = None
    fake_id, real_id = 1, 0

# --- 2. API SETUP ---
app = FastAPI(title="Deepfake Voice Detector")
API_KEY = "hackathon-secret-key-123"

# --- 3. UPDATED DATA MODEL (THE FIX IS HERE) ---
class AudioPayload(BaseModel):
    # We accept BOTH formats to be safe
    audio_base64: Optional[str] = None 
    audioBase64: Optional[str] = None  # <--- This is what the tester sends!
    
    # Accept extra fields so we don't crash
    language: Optional[str] = None
    audioFormat: Optional[str] = None

# --- 4. HELPER FUNCTIONS ---
def preprocess_audio(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1] # Remove data:audio/mp3;base64 header
        audio_bytes = base64.b64decode(base64_str)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=16000)
        return y
    except Exception as e:
        raise ValueError(f"Audio decode failed: {str(e)}")

async def verify_key(x_api_key: str = Header(None)): # Make optional initially to debug
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- 5. ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>✅ API is Live</h1><p>Send POST to /detect</p>"

@app.post("/detect", dependencies=[Depends(verify_key)])
async def analyze_audio(payload: AudioPayload):
    if not model:
        raise HTTPException(status_code=500, detail="Model loading...")

    try:
        # SMART DETECTION: Check which variable holds the data
        audio_data = payload.audio_base64 or payload.audioBase64
        
        if not audio_data:
            raise HTTPException(status_code=422, detail="MISSING DATA: Send 'audio_base64' or 'audioBase64'")

        # Process Audio
        audio_array = preprocess_audio(audio_data)
        
        # Prepare inputs
        inputs = feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=16000*10
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Results
        probs = torch.nn.functional.softmax(logits, dim=-1)
        score_fake = probs[0][fake_id].item()
        score_real = probs[0][real_id].item()
        
        if score_fake > score_real:
            result = "AI_GENERATED"
            confidence = score_fake
            explanation = "High probability of synthetic audio artifacts."
        else:
            result = "HUMAN"
            confidence = score_real
            explanation = "Natural acoustic features detected."

        return {
            "classification": result,
            "confidence_score": round(confidence, 4),
            "explanation": explanation
        }

    except Exception as e:
        # Return 500 but with the actual error message so you can see it
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)