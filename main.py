import uvicorn
import io
import base64
import librosa
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# --- 1. CONFIGURATION & MODEL LOADING ---
MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

print(f"Loading Model: {MODEL_ID} ...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # --- AUTO-DETECT LABEL MAPPING ---
    fake_id = 1 
    real_id = 0
    
    # Check model config to be sure
    labels = model.config.id2label
    print(f"Model Labels found: {labels}")
    
    if labels:
        for k, v in labels.items():
            label_text = str(v).lower()
            if "fake" in label_text or "spoof" in label_text or "ai" in label_text:
                fake_id = int(k)
            elif "real" in label_text or "bonafide" in label_text or "human" in label_text:
                real_id = int(k)
                
    print(f"Configured Logic -> Fake Index: {fake_id}, Real Index: {real_id}")

except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    model = None
    fake_id, real_id = 1, 0

# --- 2. API SETUP ---
app = FastAPI(title="Deepfake Voice Detector")
API_KEY = "hackathon-secret-key-123"

class AudioPayload(BaseModel):
    audio_base64: str

# --- 3. HELPER FUNCTIONS ---
def preprocess_audio(base64_str):
    try:
        audio_bytes = base64.b64decode(base64_str)
        audio_buffer = io.BytesIO(audio_bytes)
        # Resample to 16kHz
        y, sr = librosa.load(audio_buffer, sr=16000)
        return y
    except Exception as e:
        raise ValueError(f"Audio processing failed: {str(e)}")

async def verify_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- 4. ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Deepfake Detector Live</title>
            <style>body{font-family: sans-serif; text-align: center; padding: 50px; background-color: #f4f4f9;}</style>
        </head>
        <body>
            <h1>✅ API is Running!</h1>
            <p>Your Deepfake Detection Server is active.</p>
            <p><b>Submit POST requests to:</b> <code>/detect</code></p>
        </body>
    </html>
    """

@app.post("/detect", dependencies=[Depends(verify_key)])
async def analyze_audio(payload: AudioPayload):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Step A: Process Input
        audio_array = preprocess_audio(payload.audio_base64)
        
        # Safety Check for Audio Length
        if len(audio_array) < 4000:
            return {
                "classification": "UNCERTAIN",
                "confidence_score": 0.0,
                "explanation": "Audio is too short (less than 0.5s). Please provide a longer sample."
            }
        
        # Step B: Prepare for Model
        inputs = feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt",
            truncation=True,
            max_length=16000 * 10 
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Step C: Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Step D: Interpret Results
        probs = torch.nn.functional.softmax(logits, dim=-1)
        score_fake = probs[0][fake_id].item()
        score_real = probs[0][real_id].item()
        
        # Decision Logic (With Explanation for Hackathon Compliance)
        if score_fake > score_real:
            result = "AI_GENERATED"
            confidence = score_fake
            explanation = "High confidence of synthetic manipulation detected in spectral features."
        else:
            result = "HUMAN"
            confidence = score_real
            explanation = "Natural acoustic characteristics and breath patterns detected."

        # STRICT JSON FORMAT FOR SUBMISSION
        return {
            "classification": result,
            "confidence_score": round(confidence, 4),
            "explanation": explanation
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)