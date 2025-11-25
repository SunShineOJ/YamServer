# server_final_working.py
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
import noisereduce as nr
import scipy.signal as signal

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
    logger.info("✅ Database initialized")

init_db()

# ---- Models ----
OUR_MODEL = None
YAMNET_MODEL = None

def load_models():
    global OUR_MODEL, YAMNET_MODEL
    try:
        OUR_MODEL = tf.keras.models.load_model('improved_cough_model.keras', compile=False)
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("✅ Improved models loaded")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")

load_models()

def preprocess_audio(waveform, sr):
    """Точно такая же предобработка как при обучении"""
    try:
        # 1. Preemphasis
        waveform = librosa.effects.preemphasis(waveform)
        
        # 2. Noise reduction
        if len(waveform) > 8000:
            try:
                noise_sample = waveform[:4000]
                waveform = nr.reduce_noise(y=waveform, sr=sr, y_noise=noise_sample, prop_decrease=0.7)
            except:
                pass
        
        # 3. Bandpass filter для кашля (80-4000 Hz)
        sos = signal.butter(4, [80, 4000], 'bandpass', fs=sr, output='sos')
        waveform = signal.sosfilt(sos, waveform)
        
        # 4. Нормализация с защитой от тишины
        max_amp = np.max(np.abs(waveform))
        if max_amp < 0.01:  # Тишина
            return None
        waveform = waveform / max_amp * 0.9
        
        return waveform
    except:
        return None

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """Анализ с улучшенной моделью"""
    if not OUR_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Model not loaded"}
    
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Загружаем и обрабатываем как при обучении
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        os.unlink(tmp_path)
        
        # Проверка RMS
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms < 0.01:  # Тишина
            return {"probability": 0.0, "cough_detected": False, "message": "Silence", "rms": rms}
        
        # Предобработка
        waveform = preprocess_audio(waveform, sr)
        if waveform is None:
            return {"probability": 0.0, "cough_detected": False, "message": "Too quiet after processing"}
        
        # Дополняем до 1 секунды
        target_length = 16000
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        
        # YAMNet + MFCC фичи (как при обучении)
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = YAMNET_MODEL(waveform_tf)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        # MFCC фичи
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_features = np.mean(mfcc, axis=1)
        
        # Объединяем фичи
        combined_features = np.concatenate([avg_embedding, mfcc_features]).reshape(1, -1)
        
        # Предсказание
        prediction = OUR_MODEL.predict(combined_features, verbose=0)
        prob = float(prediction[0][0])
        
        # ПОРОГ 0.62 как при обучении
        is_cough = prob > 0.62
        
        logger.info(f"🎯 IMPROVED MODEL: {filename} | prob={prob:.3f} | cough={is_cough}")
        
        return {
            "probability": prob,
            "cough_detected": is_cough,
            "confidence": prob,
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH",
            "threshold_used": 0.62,
            "model_version": "improved"
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
        
        # Статистика по часам
        hourly = []
        for hour in range(24):
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records
                WHERE device_id=? AND cough_detected=1 AND date(timestamp)=? 
                AND strftime('%H', timestamp)=?
            ''', (device_id, today, f"{hour:02d}"))
            count = cursor.fetchone()[0] or 0
            hourly.append({"hour": f"{hour:02d}:00", "count": count})
        
        conn.close()
        
        return {
            "today_stats": {
                "total_recordings": total,
                "total_coughs": coughs
            },
            "hourly_stats": hourly,
            "device_id": device_id,
            "date": today
        }
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy", 
        "model_loaded": OUR_MODEL is not None,
        "model_version": "improved",
        "accuracy": "81%",
        "threshold": 0.62,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting IMPROVED server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
