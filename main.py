# server_working.py
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
    logger.info("✅ Database initialized")

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

def process_audio_like_colab(audio_bytes: bytes):
    """ОБРАБАТЫВАЕМ ТОЧНО КАК В КОЛАБЕ"""
    tmp_path = None
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # ТОЧНО ТАК ЖЕ КАК В ОБУЧЕНИИ!
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        
        target_length = 16000
        if len(waveform) < target_length:
            padding = target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding))
        else:
            waveform = waveform[:target_length]

        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        return waveform
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def analyze_audio(audio_bytes: bytes, filename: str) -> dict:
    """АНАЛИЗ ТОЧНО КАК В КОЛАБЕ"""
    if not OUR_MODEL:
        return {"probability": 0.0, "cough_detected": False, "message": "Model not loaded"}
    
    try:
        # 1. Обрабатываем аудио ТОЧНО как в колабе
        waveform = process_audio_like_colab(audio_bytes)
        
        # 2. Проверка на тишину
        rms = float(np.sqrt(np.mean(waveform**2)))  # ФИКС: конвертируем в float
        if rms < 0.001:
            return {"probability": 0.0, "cough_detected": False, "message": "Silence"}
        
        # 3. Извлекаем фичи
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = YAMNET_MODEL(waveform_tf)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
        
        # 4. Предсказание
        prediction = OUR_MODEL.predict(avg_embedding, verbose=0)
        prob = float(prediction[0][0])  # ФИКС: конвертируем в Python float
        
        # 5. Порог
        is_cough = prob > 0.5
        
        logger.info(f"🎯 Analysis: {filename} | prob={prob:.3f} | rms={rms:.4f} | cough={is_cough}")
        
        # ФИКС: все значения должны быть JSON-сериализуемы
        return {
            "probability": float(prob),
            "cough_detected": bool(is_cough),
            "confidence": float(prob),
            "message": "COUGH" if is_cough else "NO_COUGH",
            "rms_level": float(rms)
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
        ''', (
            device_id, 
            audio.filename, 
            float(result["probability"]),  # ФИКС: конвертируем
            int(result["cough_detected"])
        ))
        conn.commit()
        conn.close()
        
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """РАБОЧАЯ статистика"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Основная статистика
        cursor.execute('''
            SELECT 
                COUNT(*) as total_recordings,
                SUM(cough_detected) as total_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND date(timestamp)=?
        ''', (device_id, today))
        
        row = cursor.fetchone()
        total_recordings = int(row[0] or 0) if row else 0
        total_coughs = int(row[1] or 0) if row else 0
        avg_probability = float(row[2] or 0.0) if row and row[2] is not None else 0.0
        
        # Статистика по часам
        hourly_stats = []
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records
                WHERE device_id=? AND cough_detected=1 AND date(timestamp)=? 
                AND strftime('%H', timestamp)=?
            ''', (device_id, today, f"{hour:02d}"))
            count_row = cursor.fetchone()
            count = int(count_row[0] or 0) if count_row else 0
            hourly_stats.append({"hour": hour_str, "count": count})
        
        # Последние кашли
        cursor.execute('''
            SELECT timestamp, probability FROM cough_records
            WHERE device_id=? AND cough_detected=1
            ORDER BY timestamp DESC LIMIT 10
        ''', (device_id,))
        recent_coughs = [
            {"time": row[0], "probability": float(row[1])} 
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        result = {
            "today_stats": {
                "total_recordings": total_recordings,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_probability, 3)
            },
            "hourly_stats": hourly_stats,
            "recent_coughs": recent_coughs,
            "device_id": device_id,
            "date": today
        }
        
        logger.info(f"📊 Stats for {device_id}: {total_coughs} coughs today")
        return result
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/debug/stats/{device_id}")
async def debug_stats(device_id: str):
    """Отладочная статистика"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*), SUM(cough_detected) FROM cough_records 
            WHERE device_id=?
        ''', (device_id,))
        
        row = cursor.fetchone()
        total = row[0] if row else 0
        coughs = row[1] if row else 0
        
        cursor.execute('''
            SELECT filename, probability, cough_detected, timestamp 
            FROM cough_records 
            WHERE device_id=? 
            ORDER BY timestamp DESC LIMIT 5
        ''', (device_id,))
        
        recent = [
            {
                "filename": row[0],
                "probability": float(row[1]),
                "cough_detected": bool(row[2]),
                "timestamp": row[3]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            "device_id": device_id,
            "total_records": total,
            "total_coughs": coughs,
            "recent_entries": recent
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy", 
        "model_loaded": OUR_MODEL is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/")
async def root():
    return {"message": "Cough Detection Server", "status": "running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
