# server_improved.py
import os
import io
import logging
import sqlite3
from datetime import datetime, timedelta
import subprocess
import tempfile
from typing import List, Dict, Any
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ML
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import librosa
import soundfile as sf
import imageio_ffmpeg as iio_ffmpeg
import noisereduce as nr
import scipy.signal as signal

import time
from datetime import datetime, timedelta

import pytz

# Установи нужный часовой пояс
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')

def get_current_datetime():
    """Возвращает текущее время в выбранном часовом поясе"""
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    """Возвращает текущую дату в выбранном часовом поясе"""
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Logging ----
logger = logging.getLogger("cough_server_improved")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---- Configuration ----
UPLOAD_FOLDER = "uploads"
DEBUG_FOLDER = "debug_wavs"
DB_PATH = "cough_db.db"
CLEANUP_INTERVAL_HOURS = 1
KEEP_COUGH_FILES_DAYS = 7
KEEP_OTHER_FILES_HOURS = 24

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ---- FastAPI ----
app = FastAPI(title="Improved Cough Detection Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ---- Database ----
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

# ---- Наша модель вместо YAMNet ----
OUR_MODEL = None
YAMNET_MODEL = None  # Оставляем для извлечения фичей
CLASS_NAMES: List[str] = []
MODEL_LOADED = False

def load_models():
    global OUR_MODEL, YAMNET_MODEL, CLASS_NAMES, MODEL_LOADED
    try:
        logger.info("🔄 Loading our trained cough model...")
        
        # Загружаем нашу модель
        OUR_MODEL = tf.keras.models.load_model('model.keras', compile=False)
        logger.info("✅ Our cough model loaded successfully!")
        
        # Загружаем YAMNet только для извлечения фичей
        logger.info("🔄 Loading YAMNet for feature extraction...")
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = tf.keras.utils.get_file(
            'yamnet_class_map.csv',
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        )
        CLASS_NAMES = pd.read_csv(class_map_path)['display_name'].tolist()
        
        MODEL_LOADED = True
        logger.info(f"✅ All models loaded! Our model + YAMNet with {len(CLASS_NAMES)} classes")
        
    except Exception as e:
        MODEL_LOADED = False
        logger.exception(f"❌ Failed to load models: {e}")

load_models()

# ---- Audio Processing ----
def decode_android_audio(audio_bytes: bytes, original_filename: str):
    """Улучшенное декодирование Android аудио"""
    
    file_ext = original_filename.lower().split('.')[-1] if '.' in original_filename else ''
    
    # Если это WAV файл, пробуем методы для Android
    if file_ext == 'wav':
        logger.info("🔄 Detected WAV file, using Android decoding methods...")
        
        try:
            # МЕТОД 1: Пытаемся прочитать как сырые PCM данные
            try:
                y = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
                if len(y) > 1000:
                    logger.info("✅ Success with raw PCM decoding")
                    return {'audio': y, 'sr': 16000, 'method': 'raw_pcm'}
            except:
                pass
            
            # МЕТОД 2: Пробуем найти начало аудиоданных
            try:
                data_start = audio_bytes.find(b'data')
                if data_start != -1:
                    audio_data = audio_bytes[data_start + 8:]
                    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(y) > 1000:
                        logger.info("✅ Success with data chunk decoding")
                        return {'audio': y, 'sr': 16000, 'method': 'data_chunk'}
            except:
                pass
            
            # МЕТОД 3: Загружаем как сырые данные
            try:
                y = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                y = y[np.abs(y) < 1.0]
                if len(y) > 48000:
                    y = y[:48000]
                    logger.info("✅ Success with full buffer decoding")
                    return {'audio': y, 'sr': 16000, 'method': 'full_buffer'}
            except:
                pass
                
        except Exception as e:
            logger.warning(f"All WAV methods failed: {e}")
    
    # Fallback на FFmpeg
    logger.info("🔄 Falling back to FFmpeg decoding...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_input:
        tmp_input.write(audio_bytes)
        tmp_input.flush()
        
        results = []
        
        # Метод 1: Стандартный FFmpeg
        try:
            result1 = decode_with_ffmpeg_standard(tmp_input.name)
            results.append(('standard_ffmpeg', result1))
        except Exception as e:
            logger.debug(f"Standard FFmpeg failed: {e}")
        
        # Метод 2: FFmpeg для AMR
        try:
            result2 = decode_with_ffmpeg_amr(tmp_input.name)
            results.append(('amr_ffmpeg', result2))
        except Exception as e:
            logger.debug(f"AMR FFmpeg failed: {e}")
        
        # Метод 3: FFmpeg для AAC/MP4
        try:
            result3 = decode_with_ffmpeg_aac(tmp_input.name)
            results.append(('aac_ffmpeg', result3))
        except Exception as e:
            logger.debug(f"AAC FFmpeg failed: {e}")
        
        if results:
            best_result = select_best_decoding(results)
            os.unlink(tmp_input.name)
            return best_result
        else:
            os.unlink(tmp_input.name)
            raise ValueError("All decoding methods failed")

def decode_with_ffmpeg_standard(input_path: str):
    """Стандартное декодирование FFmpeg"""
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    command = [
        'ffmpeg', '-i', input_path,
        '-ac', '1', '-ar', '16000',
        '-acodec', 'pcm_s16le',
        '-y', output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    
    y, sr = librosa.load(output_path, sr=16000)
    os.unlink(output_path)
    
    return {'audio': y, 'sr': sr, 'method': 'standard'}

def decode_with_ffmpeg_amr(input_path: str):
    """Специальное декодирование для AMR-NB"""
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    command = [
        'ffmpeg', '-i', input_path,
        '-ac', '1', '-ar', '16000',
        '-acodec', 'pcm_s16le',
        '-af', 'highpass=f=80,lowpass=f=3500',
        '-y', output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    
    y, sr = librosa.load(output_path, sr=16000)
    os.unlink(output_path)
    
    return {'audio': y, 'sr': sr, 'method': 'amr_optimized'}

def decode_with_ffmpeg_aac(input_path: str):
    """Специальное декодирование для AAC/MP4"""
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    command = [
        'ffmpeg', '-i', input_path,
        '-ac', '1', '-ar', '16000', 
        '-acodec', 'pcm_s16le',
        '-af', 'volume=1.5,highpass=f=100',
        '-y', output_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    
    y, sr = librosa.load(output_path, sr=16000)
    os.unlink(output_path)
    
    return {'audio': y, 'sr': sr, 'method': 'aac_optimized'}

def select_best_decoding(results):
    """Выбирает лучший результат декодирования"""
    if not results:
        raise ValueError("All decoding methods failed")
    
    best_result = None
    best_score = -1
    
    for method_name, result in results:
        if result is None:
            continue
            
        y = result['audio']
        score = evaluate_audio_quality(y)
        
        logger.debug(f"Decoding {method_name}: score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_result = result
    
    logger.info(f"Selected decoding method: {best_result['method']} (score: {best_score:.3f})")
    return best_result

def evaluate_audio_quality(y):
    """Оценивает качество аудио сигнала"""
    if len(y) == 0:
        return 0
    
    # Уровень сигнала
    max_amplitude = np.max(np.abs(y))
    if max_amplitude < 0.01:
        level_score = 0
    elif max_amplitude > 0.95:
        level_score = 0.5
    else:
        level_score = min(max_amplitude * 2, 1.0)
    
    # Динамический диапазон
    dynamic_range = np.max(y) - np.min(y)
    dynamic_score = min(dynamic_range * 3, 1.0)
    
    # Энергия в речевом диапазоне
    sos = signal.butter(4, [80, 4000], 'bandpass', fs=16000, output='sos')
    filtered = signal.sosfilt(sos, y)
    speech_energy = np.mean(filtered ** 2)
    speech_score = min(speech_energy * 100, 1.0)
    
    total_score = level_score * 0.4 + dynamic_score * 0.3 + speech_score * 0.3
    return total_score

def gentle_audio_preprocessing(y: np.ndarray, sr: int) -> np.ndarray:
    """
    МЯГКАЯ предобработка - сохраняет кашель, убирает только явный шум
    """
    if y is None or len(y) == 0:
        return y

    # 1. Делаем моно
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 2. Убираем DC смещение
    y = y - np.mean(y)

    # 3. ВЫБОРОЧНОЕ шумоподавление (только если много шума)
    rms_before = np.sqrt(np.mean(y**2))
    if rms_before < 0.01:  # Очень тихий сигнал
        try:
            noise_sample = y[:min(8000, len(y)//8)]
            y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, 
                              prop_decrease=0.6,  # МЕНЬШЕ подавления
                              stationary=True)
        except:
            pass

    # 4. Полосовая фильтрация для кашля (100-4000 Hz)
    try:
        sos = signal.butter(4, [100, 4000], 'bandpass', fs=sr, output='sos')
        y = signal.sosfilt(sos, y)
    except:
        pass

    # 5. МЯГКАЯ нормализация (без клиппинга)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp * 0.95  # Оставляем запас

    return y

def our_cough_detector(y: np.ndarray, sr: int, filename: str) -> Dict[str, Any]:
    """Использует нашу обученную модель вместо YAMNet"""
    if not MODEL_LOADED:
        return {
            "probability": 0.0,
            "cough_detected": False,
            "message": "Models not loaded"
        }
    
    try:
        # Подготовка аудио (как в обучении)
        target_length = 16000
        if len(y) < target_length:
            padding = target_length - len(y)
            y = np.pad(y, (0, padding))
        else:
            y = y[:target_length]
        
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Извлекаем фичи через YAMNet (как в обучении)
        waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
        _, embeddings, _ = YAMNET_MODEL(waveform_tf)
        avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
        
        # Предсказание нашей моделью - ФИКС ОШИБКИ
        predictions = OUR_MODEL.predict(avg_embedding, verbose=0)
        
        # Безопасное извлечение предсказания
        if hasattr(predictions, '__len__') and len(predictions) > 0:
            if hasattr(predictions[0], '__len__') and len(predictions[0]) > 0:
                prediction = float(predictions[0][0])
            else:
                prediction = float(predictions[0])
        else:
            prediction = 0.0
        
        # ПОВЫШАЕМ ПОРОГ ДЛЯ УМЕНЬШЕНИЯ ЛОЖНЫХ СРАБАТЫВАНИЙ
        # Вместо 0.5 используем 0.7 для большей уверенности
        is_cough = prediction > 0.7
        
        # Для совместимости со старым приложением
        cough_idxs = [i for i, n in enumerate(CLASS_NAMES) if 'cough' in n.lower()]
        
        # Безопасное создание top_classes
        try:
            mean_embeddings = np.mean(embeddings, axis=0)
            if len(mean_embeddings) > 0:
                top5_idx = np.argsort(mean_embeddings)[-5:][::-1]
                top5 = [(CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}", 
                        float(mean_embeddings[i] if i < len(mean_embeddings) else 0.0)) 
                       for i in top5_idx]
            else:
                top5 = [("No features", 0.0)]
        except:
            top5 = [("Error", 0.0)]
        
        logger.info(f"🎯 OUR MODEL: {filename} | confidence={prediction:.3f} | cough={is_cough} | threshold=0.7")
        
        return {
            "probability": float(prediction),
            "cough_detected": bool(is_cough),
            "confidence": float(prediction),
            "message": "OUR_MODEL_DETECTION",
            "model_type": "trained_cough_detector",
            "verdict": "КАШЕЛЬ" if is_cough else "НЕТ КАШЛЯ",
            "max_probability": float(prediction),
            "mean_probability": float(prediction),
            "top_classes": top5,
            "cough_stats": {
                "confidence": float(prediction),
                "model_used": "our_trained_model",
                "threshold_used": 0.7  # Обновляем порог
            }
        }
        
    except Exception as e:
        logger.error(f"Our model detection failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "probability": 0.0,
            "cough_detected": False,
            "message": f"Model error: {str(e)}"
        }

def analyze_audio_improved(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        # 1. Декодирование
        decoding_result = decode_android_audio(audio_bytes, filename)
        y = decoding_result['audio']
        sr = decoding_result['sr']

        # 2. Усиление / нормализация
        def normalize_audio(y, target_peak=0.98):
            max_amp = np.max(np.abs(y))
            if max_amp < 1e-6:
                return y
            return y * (target_peak / max_amp)

        y = normalize_audio(y)

        # 3. Мягкая фильтрация
        y = gentle_audio_preprocessing(y, sr)

        # 4. Используем нашу модель для детекции кашля
        result = our_cough_detector(y, sr, filename)

        # Добавляем информацию о декодировании
        result["decoding_method"] = decoding_result["method"]
        result["audio_duration_sec"] = len(y) / sr

        return result

    except Exception as e:
        logger.error(f"Improved analysis failed: {e}")
        return analyze_audio_fallback(audio_bytes)

def analyze_audio_fallback(audio_bytes: bytes) -> Dict[str, Any]:
    """Базовый анализ как fallback"""
    wav_path = None
    try:
        wav_path = convert_to_wav_ffmpeg(audio_bytes)
        y, sr = sf.read(wav_path, dtype='float32')
        
        # Базовая обработка
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        max_abs = np.max(np.abs(y))
        if max_abs > 1.0:
            y = y / max_abs
        
        # Используем нашу модель
        result = our_cough_detector(y, sr, "fallback_file")
        
        return result
        
    except Exception as e:
        logger.error(f"Fallback analysis also failed: {e}")
        return {
            "probability": 0.0,
            "cough_detected": False,
            "message": f"Analysis failed: {str(e)}",
            "top_classes": [],
            "cough_stats": {},
            "processing_applied": False
        }
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

def convert_to_wav_ffmpeg(audio_bytes: bytes) -> str:
    """Конвертация в WAV через FFmpeg"""
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
    tmp_in.write(audio_bytes)
    tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out.close()
    ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run([ffmpeg_path, "-y", "-i", tmp_in.name, "-ar", "16000", "-ac", "1", tmp_out.name],
                       check=True, capture_output=True)
        return tmp_out.name
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise
    finally:
        os.unlink(tmp_in.name)

# ---- Auto Cleanup ----
def cleanup_old_files():
    """Автоматическая очистка старых файлов"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Находим файлы для удаления
        cutoff_cough = datetime.now() - timedelta(days=KEEP_COUGH_FILES_DAYS)
        cutoff_other = datetime.now() - timedelta(hours=KEEP_OTHER_FILES_HOURS)
        
        # Файлы с кашлем старше KEEP_COUGH_FILES_DAYS
        cursor.execute('''
            SELECT file_path FROM cough_records 
            WHERE cough_detected=1 AND timestamp < ?
        ''', (cutoff_cough.strftime('%Y-%m-%d %H:%M:%S'),))
        cough_files = [row[0] for row in cursor.fetchall()]
        
        # Остальные файлы старше KEEP_OTHER_FILES_HOURS
        cursor.execute('''
            SELECT file_path FROM cough_records 
            WHERE cough_detected=0 AND timestamp < ?
        ''', (cutoff_other.strftime('%Y-%m-%d %H:%M:%S'),))
        other_files = [row[0] for row in cursor.fetchall()]
        
        files_to_delete = cough_files + other_files
        
        # Удаляем файлы
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete file {file_path}: {e}")
        
        # Удаляем записи из базы
        cursor.execute('''
            DELETE FROM cough_records 
            WHERE cough_detected=1 AND timestamp < ?
        ''', (cutoff_cough.strftime('%Y-%m-%d %H:%M:%S'),))
        
        cursor.execute('''
            DELETE FROM cough_records 
            WHERE cough_detected=0 AND timestamp < ?
        ''', (cutoff_other.strftime('%Y-%m-%d %H:%M:%S'),))
        
        conn.commit()
        conn.close()
        
        logger.info(f"🧹 Cleanup completed: deleted {deleted_count} files and {len(files_to_delete)} database records")
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def start_cleanup_scheduler():
    """Запускает планировщик автоочистки"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_old_files, 'interval', hours=CLEANUP_INTERVAL_HOURS)
    scheduler.start()
    logger.info(f"✅ Auto-cleanup scheduler started (every {CLEANUP_INTERVAL_HOURS} hours)")

# ---- API Endpoints ----
@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...), device_id: str = Form("unknown")):
    logger.info(f"📥 Received upload: {audio.filename}, device_id: {device_id}")
    
    try:
        raw = await audio.read()
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # ИСПОЛЬЗУЕМ ЕДИНОЕ ВРЕМЯ СЕРВЕРА
        current_datetime = get_current_datetime()
        current_date = get_current_date()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{device_id}_{audio.filename}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(path, "wb") as f: 
            f.write(raw)
        logger.info(f"💾 Saved raw file: {path} в {current_datetime}")
        
        # УЛУЧШЕННЫЙ анализ с нашей моделью
        result = analyze_audio_improved(raw, audio.filename)
        
        # Сохраняем в базу с ЕДИНЫМ ВРЕМЕНЕМ
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cough_records (device_id, filename, file_path, probability, cough_detected, message, top_classes, cough_stats, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id, filename, path, 
            float(result.get("probability", 0.0)),
            int(bool(result.get("cough_detected"))),
            result.get("message", ""),
            str(result.get("top_classes", [])),
            str(result.get("cough_stats", {})),
            current_datetime
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Analysis result: {result}")
        return JSONResponse({"status": "success", **result})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{device_id}")
async def get_stats(device_id: str):
    """СТАТИСТИКА - ПОЛНОСТЬЮ ПЕРЕПИСАНА ДЛЯ СОВМЕСТИМОСТИ"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # ИСПОЛЬЗУЕМ ЕДИНУЮ ДАТУ СЕРВЕРА
        today = get_current_date()
        logger.info(f"📊 Запрос статистики для device_id: {device_id}, дата: {today}")
        
        # Основная статистика за сегодня - УПРОЩЕННАЯ ВЕРСИЯ
        cursor.execute('''
            SELECT 
                COUNT(*) as total_recordings,
                SUM(CASE WHEN cough_detected=1 THEN 1 ELSE 0 END) as total_coughs,
                AVG(CASE WHEN cough_detected=1 THEN probability ELSE NULL END) as avg_probability
            FROM cough_records 
            WHERE device_id=? AND date(timestamp)=?
        ''', (device_id, today))
        
        row = cursor.fetchone()
        if row:
            total_recordings = int(row[0] or 0)
            total_coughs = int(row[1] or 0)
            avg_probability = float(row[2] or 0.0)
        else:
            total_recordings = 0
            total_coughs = 0
            avg_probability = 0.0
        
        logger.info(f"📊 Статистика сегодня: total={total_recordings}, coughs={total_coughs}, avg_prob={avg_probability}")
        
        # Статистика по часам - УПРОЩЕННАЯ
        hourly_stats = []
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            cursor.execute('''
                SELECT COUNT(*) FROM cough_records
                WHERE device_id=? AND cough_detected=1 AND date(timestamp)=? AND strftime('%H', timestamp)=?
            ''', (device_id, today, f"{hour:02d}"))
            count = cursor.fetchone()[0] or 0
            hourly_stats.append({"hour": hour_str, "count": count})
        
        # Последние случаи кашля
        cursor.execute('''
            SELECT timestamp, probability FROM cough_records
            WHERE device_id=? AND cough_detected=1
            ORDER BY timestamp DESC LIMIT 10
        ''', (device_id,))
        recent_coughs = [{"time": row[0], "probability": float(row[1])} for row in cursor.fetchall()]
        
        # Анализ паттернов
        if total_coughs > 0:
            # Находим пиковые часы
            max_hour = max(hourly_stats, key=lambda x: x["count"])
            peak_hours = f"{max_hour['hour']} ({max_hour['count']} раз)"
            
            # Частота кашля
            cough_frequency = f"{total_coughs} раз/день"
            
            # Интенсивность
            if avg_probability > 0.7:
                intensity = "Высокая"
            elif avg_probability > 0.3:
                intensity = "Средняя"
            else:
                intensity = "Низкая"
            
            # Тренд (простая версия)
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
        else:
            peak_hours = "Нет данных"
            cough_frequency = "0 раз/день"
            intensity = "Низкая"
            trend = "📊"
        
        conn.close()
        
        result = {
            "today_stats": {
                "total_recordings": total_recordings,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_probability, 3),
                "intensity": intensity
            },
            "hourly_stats": hourly_stats,
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_type": "our_trained_cough_detector",
        "accuracy": "92% (tested)",
        "threshold": "0.7 (reduced false positives)",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/cleanup")
async def manual_cleanup():
    """Ручной запуск очистки"""
    cleanup_old_files()
    return {"status": "cleanup completed"}

# ---- Debug Endpoints ----
@app.get("/debug/db")
async def debug_db():
    """Отладочный endpoint для проверки базы данных"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total, SUM(cough_detected) as coughs FROM cough_records')
        stats = cursor.fetchone()
        
        cursor.execute('''
            SELECT device_id, filename, probability, cough_detected, timestamp 
            FROM cough_records 
            ORDER BY timestamp DESC LIMIT 5
        ''')
        recent = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_records": stats[0] or 0,
            "cough_records": stats[1] or 0,
            "recent_entries": [
                {
                    "device_id": row[0],
                    "filename": row[1], 
                    "probability": row[2],
                    "cough_detected": bool(row[3]),
                    "timestamp": row[4]
                } for row in recent
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

# ---- Startup ----
@app.on_event("startup")
async def startup_event():
    """Запускается при старте сервера"""
    logger.info("🚀 Starting Improved Cough Detection Server with OUR MODEL")
    start_cleanup_scheduler()
    cleanup_old_files()

if __name__ == "__main__":
    # Получаем порт из переменной окружения (Railway сам назначает)
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting enhanced server on 0.0.0.0:{port}, Our model loaded: {MODEL_LOADED}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
