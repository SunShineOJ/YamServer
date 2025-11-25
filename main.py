# server_final_fixed.py
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
        
        # НЕМНОГО ПОНИЗИМ ПОРОГ для лучшего recall (с 0.62 до 0.58)
        is_cough = prob > 0.58
        
        logger.info(f"🎯 IMPROVED MODEL: {filename} | prob={prob:.3f} | cough={is_cough} | threshold=0.58")
        
        return {
            "probability": prob,
            "cough_detected": is_cough,
            "confidence": prob,
            "message": "COUGH_DETECTED" if is_cough else "NO_COUGH",
            "threshold_used": 0.58,
            "model_version": "improved_v2"
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
        
        # Сохраняем в базу С ТОЙ ЖЕ СТРУКТУРОЙ как в старом сервере
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records 
            (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (
            device_id, 
            audio.filename,
            "",  # file_path - оставляем пустым для совместимости
            float(result["probability"]),
            int(result["cough_detected"]),
            result["message"],
            "[]",  # top_classes
            "{}"   # cough_stats
        ))
        conn.commit()
        conn.close()
        
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """РАБОЧАЯ статистика как в старом сервере"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"📊 Статистика для {device_id}, дата: {today}")
        
        # Основная статистика за сегодня
        cursor.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN cough_detected=1 THEN 1 ELSE 0 END),
                   AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END)
            FROM cough_records 
            WHERE device_id=? AND date(timestamp)=?
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
            WHERE device_id=? AND cough_detected=1 AND date(timestamp)=?
            GROUP BY hr
        ''', (device_id, today))
        rows = cursor.fetchall()
        hourly = [{"hour": f"{int(h)}:00", "count": c} for h, c in rows]
        
        # Заполняем пропущенные часы нулями
        existing_hours = {item["hour"] for item in hourly}
        for hh in range(24):
            hs = f"{hh:02d}:00"
            if hs not in existing_hours:
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
            # Находим пиковые часы
            if hourly:
                max_hour = max(hourly, key=lambda x: x["count"])
                peak_hours = f"{max_hour['hour']} ({max_hour['count']} раз)"
            
            # Частота кашля
            cough_frequency = f"{total_coughs} раз/день"
            
            # Интенсивность
            if avg_prob > 0.7:
                intensity = "Высокая"
            elif avg_prob > 0.3:
                intensity = "Средняя"
            else:
                intensity = "Низкая"
            
            # Тренд
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records 
                WHERE device_id=? AND cough_detected=1 AND date(timestamp)=date(?, '-1 day')
            ''', (device_id, today))
            yesterday_result = cursor.fetchone()
            yesterday_coughs = int(yesterday_result[0]) if yesterday_result and yesterday_result[0] is not None else 0
            
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
        return JSONResponse(
            {"status": "error", "message": f"Stats error: {str(e)}"}, 
            status_code=500
        )

@app.get("/debug/stats/{device_id}")
async def debug_stats(device_id: str):
    """Отладочная статистика"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*), SUM(cough_detected) FROM cough_records WHERE device_id=?', (device_id,))
        row = cursor.fetchone()
        total = row[0] if row else 0
        coughs = row[1] if row else 0
        
        cursor.execute('SELECT filename, probability, cough_detected, timestamp FROM cough_records WHERE device_id=? ORDER BY timestamp DESC LIMIT 5', (device_id,))
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
        "model_version": "improved_v2",
        "accuracy": "81%",
        "threshold": 0.58,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/")
async def root():
    return {"message": "Improved Cough Detection Server", "status": "running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting IMPROVED server v2 on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
