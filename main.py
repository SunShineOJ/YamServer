# server_fixed.py
import os
import logging
import sqlite3
from datetime import datetime
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import tempfile

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cough_server")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Database ----
DB_PATH = "cough_db.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cough_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            filename TEXT,
            probability REAL,
            cough_detected INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# ---- Models ----
OUR_MODEL = None
YAMNET_MODEL = None

def load_models():
    global OUR_MODEL, YAMNET_MODEL
    try:
        OUR_MODEL = tf.keras.models.load_model('model.keras', compile=False)
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("✅ Models loaded")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")

load_models()

def process_audio_like_colab(audio_bytes: bytes, filename: str):
    """ОБРАБАТЫВАЕМ ТОЧНО КАК В КОЛАБЕ"""
    try:
        # Сохраняем во временный файл и обрабатываем как в колабе
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # ТОЧНО ТАК ЖЕ КАК В ОБУЧЕНИИ!
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)  # duration=1.0 ВАЖНО!
        
        # ТОЧНО ТАК ЖЕ КАК В ОБУЧЕНИИ!
        target_length = 16000
        if len(waveform) < target_length:
            padding = target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding))
        else:
            waveform = waveform[:target_length]

        # ТОЧНО ТАК ЖЕ КАК В ОБУЧЕНИИ!
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        os.unlink(tmp_path)
        return waveform
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """АНАЛИЗ ТОЧНО КАК В КОЛАБЕ"""
    if not OUR_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Model not loaded"}
    
    try:
        # 1. Обрабатываем аудио ТОЧНО как в колабе
        waveform = process_audio_like_colab(audio_bytes, filename)
        
        # 2. Проверка на тишину (как в колабе)
        rms = np.sqrt(np.mean(waveform**2))
        if rms < 0.001:
            return {"probability": 0.0, "cough_detected": False, "message": "Silence"}
        
        # 3. Извлекаем фичи ТОЧНО как в колабе
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = YAMNET_MODEL(waveform_tf)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
        
        # 4. Предсказание
        prediction = OUR_MODEL.predict(avg_embedding, verbose=0)
        prob = float(prediction[0][0]) if hasattr(prediction[0], '__len__') else float(prediction[0])
        
        # 5. Порог как в колабе (0.5 или найденный оптимальный)
        is_cough = prob > 0.5
        
        logger.info(f"🎯 Analysis: {filename} | prob={prob:.3f} | rms={rms:.4f} | cough={is_cough}")
        
        return {
            "probability": prob,
            "cough_detected": is_cough,
            "confidence": prob,
            "message": "COUGH" if is_cough else "NO_COUGH",
            "rms_level": rms
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"probability": 0.0, "cough_detected": False, "message": f"Error: {str(e)}"}

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    logger.info(f"📥 Upload: {audio.filename}")
    
    try:
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(400, "Empty file")
        
        # Анализируем
        result = analyze_audio(raw, audio.filename)
        
        # Сохраняем в базу
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records (device_id, filename, probability, cough_detected, timestamp)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (device_id, audio.filename, result["probability"], int(result["cough_detected"])))
        conn.commit()
        conn.close()
        
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute('''
            SELECT COUNT(*), SUM(cough_detected) FROM cough_records 
            WHERE device_id=? AND date(timestamp)=?
        ''', (device_id, today))
        
        row = cursor.fetchone()
        total = row[0] or 0
        coughs = row[1] or 0
        
        return {
            "today_stats": {
                "total_recordings": total,
                "total_coughs": coughs
            },
            "device_id": device_id,
            "date": today
        }
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": OUR_MODEL is not None}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
