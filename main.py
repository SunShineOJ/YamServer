# server_compatible.py
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
import pytz

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cough_server")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Configuration ----
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')

def get_current_datetime():
    """ТОЧНО КАК В СТАРОМ СЕРВЕРЕ"""
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    """ТОЧНО КАК В СТАРОМ СЕРВЕРЕ"""
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Database ----
DB_PATH = "cough_db.db"
def init_db():
    """ТОЧНО КАК В СТАРОМ СЕРВЕРЕ"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cough_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            filename TEXT,
            file_path TEXT,
            probability REAL,
            cough_detected INTEGER,
            message TEXT,
            top_classes TEXT,
            cough_stats TEXT,
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
    """Улучшенная предобработка"""
    try:
        waveform = librosa.effects.preemphasis(waveform)
        
        if len(waveform) > 8000:
            try:
                noise_sample = waveform[:4000]
                waveform = nr.reduce_noise(y=waveform, sr=sr, y_noise=noise_sample, prop_decrease=0.7)
            except:
                pass
        
        sos = signal.butter(4, [80, 4000], 'bandpass', fs=sr, output='sos')
        waveform = signal.sosfilt(sos, waveform)
        
        max_amp = np.max(np.abs(waveform))
        if max_amp < 0.01:
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        waveform, sr = librosa.load(tmp_path, sr=16000, duration=1.0)
        os.unlink(tmp_path)
        
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms < 0.01:
            return {"probability": 0.0, "cough_detected": False, "message": "Silence"}
        
        waveform = preprocess_audio(waveform, sr)
        if waveform is None:
            return {"probability": 0.0, "cough_detected": False, "message": "Too quiet after processing"}
        
        target_length = 16000
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        else:
            waveform = waveform[:target_length]
        
        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = YAMNET_MODEL(waveform_tf)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_features = np.mean(mfcc, axis=1)
        
        combined_features = np.concatenate([avg_embedding, mfcc_features]).reshape(1, -1)
        
        prediction = OUR_MODEL.predict(combined_features, verbose=0)
        prob = float(prediction[0][0])
        
        # БАЛАНСИРОВАННЫЙ ПОРОГ
        is_cough = prob > 0.60
        
        logger.info(f"🎯 IMPROVED MODEL: {filename} | prob={prob:.3f} | cough={is_cough}")
        
        return {
            "probability": prob,
            "cough_detected": is_cough,
            "confidence": prob,
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH"
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"probability": 0.0, "cough_detected": False, "message": f"Error: {str(e)}"}

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    logger.info(f"📥 Upload: {audio.filename}, device_id: {device_id}")
    
    try:
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(400, "Empty file")
        
        # ТОЧНО КАК В СТАРОМ СЕРВЕРЕ - с временем московским
        current_datetime = get_current_datetime()
        current_date = get_current_date()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{device_id}_{audio.filename}"
        
        result = analyze_audio(raw, audio.filename)
        
        # ТОЧНО КАК В СТАРОМ СЕРВЕРЕ - полная структура
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records 
            (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, 
            filename,
            "",  # file_path
            float(result["probability"]),
            int(result["cough_detected"]),
            result["message"],
            "[]",  # top_classes
            "{}",  # cough_stats
            current_datetime  # ТОЧНО КАК В СТАРОМ СЕРВЕРЕ
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Analysis result: {result}")
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """СТАРАЯ РАБОЧАЯ СТАТИСТИКА"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # ТОЧНО КАК В СТАРОМ СЕРВЕРЕ
        today = get_current_date()
        logger.info(f"📊 Запрос статистики для device_id: {device_id}, дата: {today}")
        
        # Основная статистика за сегодня
        cursor.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN cough_detected=1 THEN 1 ELSE 0 END),
                   AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END)
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp)=?
        ''', (device_id, today))
        
        stats = cursor.fetchone()
        total = int(stats[0] or 0) if stats else 0
        total_coughs = int(stats[1] or 0) if stats else 0
        avg_prob = float(stats[2] or 0.0) if stats and stats[2] is not None else 0.0
        
        logger.info(f"📊 Статистика сегодня: total={total}, coughs={total_coughs}, avg_prob={avg_prob}")
        
        # Статистика по часам
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hr, COUNT(*) 
            FROM cough_records
            WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=?
            GROUP BY hr
        ''', (device_id, today))
        rows = cursor.fetchall()
        hourly = [{"hour": f"{h}:00", "count": c} for h, c in rows]
        
        # Заполняем пропущенные часы нулями
        for hh in range(24):
            hs = f"{hh:02d}:00"
            if not any(item["hour"] == hs for item in hourly):
                hourly.append({"hour": hs, "count": 0})
        hourly.sort(key=lambda x: x["hour"])
        
        # Последние случаи кашля
        cursor.execute('''
            SELECT timestamp, probability FROM cough_records
            WHERE device_id=? AND cough_detected=1
            ORDER BY timestamp DESC LIMIT 10
        ''', (device_id,))
        recent_coughs = [{"time": row[0], "probability": float(row[1])} for row in cursor.fetchall()]
        
        # Анализ паттернов
        peak_hours = "Нет данных"
        cough_frequency = "0 раз/день"
        intensity = "Низкая"
        trend = "📊"
        
        if total_coughs > 0:
            if hourly:
                max_hour = max(hourly, key=lambda x: x["count"])
                peak_hours = f"{max_hour['hour']} ({max_hour['count']} раз)"
            
            cough_frequency = f"{total_coughs} раз/день"
            
            if avg_prob > 0.7:
                intensity = "Высокая"
            elif avg_prob > 0.3:
                intensity = "Средняя"
            else:
                intensity = "Низкая"
            
            # Тренд
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records 
                WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=DATE('now', '-1 day')
            ''', (device_id,))
            yesterday_coughs = cursor.fetchone()[0] or 0
            
            if total_coughs > yesterday_coughs:
                trend = "📈 Растет"
            elif total_coughs < yesterday_coughs:
                trend = "📉 Снижается"
            else:
                trend = "➡️ Стабильно"
        
        conn.close()
        
        result = {
            "today_stats": {
                "total_recordings": total,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_prob, 3),
                "intensity": intensity
            },
            "hourly_stats": hourly,
            "recent_coughs": recent_coughs,
            "patterns": {
                "peak_hours": peak_hours,
                "cough_frequency": cough_frequency,
                "intensity": intensity,
                "trend": trend
            }
        }
        
        logger.info(f"📊 Финальный результат статистики: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"Stats error: {e}")
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/debug/stats/{device_id}")
async def debug_stats(device_id: str):
    """Отладочная статистика"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = get_current_date()
        
        logger.info(f"🔍 DEBUG STATS: device_id={device_id}, today={today}")
        
        cursor.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN cough_detected=1 THEN 1 ELSE 0 END),
                   AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END)
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp)=?
        ''', (device_id, today))
        
        stats = cursor.fetchone()
        total = int(stats[0] or 0) if stats else 0
        total_coughs = int(stats[1] or 0) if stats else 0
        avg_prob = float(stats[2] or 0.0) if stats and stats[2] is not None else 0.0
        
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
            "today_stats": {
                "total_recordings": total,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_prob, 3)
            },
            "recent_entries": recent,
            "device_id": device_id,
            "date": today
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting COMPATIBLE server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
