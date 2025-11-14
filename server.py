# server_enhanced.py
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

import pytz  # Установи: pip install pytz

# Установи нужный часовой пояс
SERVER_TIMEZONE = pytz.timezone('Europe/Moscow')  # Или 'UTC' если хочешь универсально

def get_current_datetime():
    """Возвращает текущее время в выбранном часовом поясе"""
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    """Возвращает текущую дату в выбранном часовом поясе"""
    return datetime.now(SERVER_TIMEZONE).strftime("%Y-%m-%d")

# ---- Logging ----
logger = logging.getLogger("cough_server_enhanced")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---- Configuration ----
UPLOAD_FOLDER = "uploads"
DEBUG_FOLDER = "debug_wavs"
DB_PATH = "cough_db.db"
CLEANUP_INTERVAL_HOURS = 1  # Автоочистка каждые 1 час
KEEP_COUGH_FILES_DAYS = 7   # Хранить файлы с кашлем 7 дней
KEEP_OTHER_FILES_HOURS = 24 # Хранить остальные файлы 24 часа

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ---- FastAPI ----
app = FastAPI(title="Enhanced Cough Detection Server")
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

# ---- YAMNet ----
YAMNET_MODEL = None
CLASS_NAMES: List[str] = []
YAMNET_LOADED = False

def load_yamnet():
    global YAMNET_MODEL, CLASS_NAMES, YAMNET_LOADED
    try:
        logger.info("🔄 Loading YAMNet...")
        YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = tf.keras.utils.get_file(
            'yamnet_class_map.csv',
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        )
        CLASS_NAMES = pd.read_csv(class_map_path)['display_name'].tolist()
        YAMNET_LOADED = True
        logger.info(f"✅ YAMNet loaded with {len(CLASS_NAMES)} classes")
    except Exception as e:
        YAMNET_LOADED = False
        logger.exception("❌ Failed to load YAMNet: %s", e)

load_yamnet()

def find_cough_indices() -> List[int]:
    return [i for i, n in enumerate(CLASS_NAMES) if 'cough' in n.lower()]

# ---- Audio Processing ----
def decode_android_audio(audio_bytes: bytes, original_filename: str):
    """РАДИКАЛЬНОЕ решение - обход битых WAV заголовков"""
    
    file_ext = original_filename.lower().split('.')[-1] if '.' in original_filename else ''
    
    # Если это WAV файл, пробуем РАДИКАЛЬНЫЕ методы
    if file_ext == 'wav':
        logger.info("🔄 Detected WAV file, using radical decoding methods...")
        
        try:
            # МЕТОД 1: Пытаемся прочитать как сырые PCM данные
            # Предполагаем стандартные параметры: 16kHz, 16-bit, mono
            try:
                # Пробуем интерпретировать как PCM 16-bit
                y = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
                if len(y) > 1000:  # Если получили разумное количество samples
                    logger.info("✅ Success with raw PCM decoding")
                    return {'audio': y, 'sr': 16000, 'method': 'raw_pcm'}
            except:
                pass
            
            # МЕТОД 2: Пробуем найти начало аудиоданных (пропускаем заголовок)
            try:
                # Ищем данные после 'data' chunk (обычно 44 байта)
                data_start = audio_bytes.find(b'data')
                if data_start != -1:
                    audio_data = audio_bytes[data_start + 8:]  # +8 чтобы пропустить 'data' и размер
                    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(y) > 1000:
                        logger.info("✅ Success with data chunk decoding")
                        return {'audio': y, 'sr': 16000, 'method': 'data_chunk'}
            except:
                pass
            
            # МЕТОД 3: Пробуем загрузить как сырые данные без заголовка
            try:
                # Просто берем все данные как PCM
                y = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # Фильтруем только разумные значения (избегаем шум)
                y = y[np.abs(y) < 1.0]  # убираем выбросы
                if len(y) > 48000:  # 3 секунды при 16kHz
                    y = y[:48000]  # обрезаем до 3 секунд
                    logger.info("✅ Success with full buffer decoding")
                    return {'audio': y, 'sr': 16000, 'method': 'full_buffer'}
            except:
                pass
                
        except Exception as e:
            logger.warning(f"All radical WAV methods failed: {e}")
    
    # Если WAV методы не сработали или это не WAV, используем старую логику
    logger.info("🔄 Falling back to standard decoding...")
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
        '-af', 'volume=2.0,highpass=f=100',
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

def enhanced_audio_processing(y, sr):
    """Улучшенная обработка аудио для Android записей"""
    
    # 1. Шумоподавление
    try:
        noise_sample = y[:min(16000, len(y)//4)]
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.8, stationary=True)
    except:
        y_denoised = y
    
    # 2. Агрессивное усиление для тихих записей
    current_max = np.max(np.abs(y_denoised))
    if current_max < 0.1:
        gain = 10.0
    elif current_max < 0.3:
        gain = 5.0
    else:
        gain = 2.0
    
    y_amplified = y_denoised * gain
    
    # 3. Компрессия
    threshold = 0.3
    ratio = 4
    y_compressed = np.where(np.abs(y_amplified) > threshold, 
                           threshold + (y_amplified - threshold) / ratio, 
                           y_amplified)
    
    # 4. Полосовая фильтрация для кашля
    sos_low = signal.butter(4, 100, 'high', fs=sr, output='sos')
    sos_high = signal.butter(4, 4000, 'low', fs=sr, output='sos')
    
    y_filtered = signal.sosfilt(sos_low, y_compressed)
    y_filtered = signal.sosfilt(sos_high, y_filtered)
    
    # 5. Финальная нормализация
    max_amp = np.max(np.abs(y_filtered))
    if max_amp > 0:
        y_final = y_filtered / max_amp * 0.9
    else:
        y_final = y_filtered
    
    return y_final

# ---- Enhanced Analysis ----
def run_yamnet(waveform: np.ndarray):
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = YAMNET_MODEL(waveform_tf)
    return scores.numpy(), embeddings.numpy(), spectrogram.numpy()

def aggressive_cough_detector_enhanced(y, sr, scores, filename):
    """УЛУЧШЕННЫЙ агрессивный детектор кашля для Android записей"""
    
    cough_idxs = find_cough_indices()
    
    if not cough_idxs:
        return 0.0, False, "No cough classes in YAMNet"
    
    cough_scores = scores[:, cough_idxs]
    per_frame_cough = np.max(cough_scores, axis=1)
    
    # БАЗОВЫЕ МЕТРИКИ YAMNet
    max_prob = np.max(per_frame_cough)
    mean_prob = np.mean(per_frame_cough)
    
    # Android-specific: более агрессивные пороги
    very_weak_frames = np.sum(per_frame_cough > 0.005)
    weak_frames = np.sum(per_frame_cough > 0.01)
    medium_frames = np.sum(per_frame_cough > 0.03)
    strong_frames = np.sum(per_frame_cough > 0.08)
    
    total_frames = len(per_frame_cough)
    
    # АНАЛИЗ ЭНЕРГЕТИЧЕСКИХ ПАТТЕРНОВ
    energy_features = analyze_energy_patterns(y, sr)
    
    # АГРЕССИВНАЯ ЛОГИКА ДЛЯ ANDROID
    detection_reasons = []
    base_prob = 0.0
    
    # Основные критерии
    if strong_frames >= 1:
        base_prob += 0.5
        detection_reasons.append(f"strong({strong_frames})")
    elif medium_frames >= 2:
        base_prob += 0.4
        detection_reasons.append(f"medium({medium_frames})")
    elif weak_frames >= 3:
        base_prob += 0.3
        detection_reasons.append(f"weak({weak_frames})")
    elif very_weak_frames >= 5:
        base_prob += 0.2
        detection_reasons.append(f"vweak({very_weak_frames})")
    
    # Бонус за энергетические паттерны
    if energy_features['valid_cough_like_events'] >= 1:
        base_prob += 0.2
        detection_reasons.append(f"energy_events({energy_features['valid_cough_like_events']})")
    
    # Бонус за максимальную вероятность
    if max_prob > 0.05:
        base_prob += max_prob
        detection_reasons.append(f"maxP({max_prob:.3f})")
    
    final_prob = min(base_prob, 0.95)
    
    # ОЧЕНЬ АГРЕССИВНОЕ РЕШЕНИЕ ДЛЯ ANDROID
    cough_detected = (
        strong_frames >= 1 or
        medium_frames >= 2 or 
        weak_frames >= 3 or
        (very_weak_frames >= 4 and energy_features['valid_cough_like_events'] >= 1) or
        final_prob > 0.25
    )
    
    reason = " + ".join(detection_reasons) if detection_reasons else "marginal_signals"
    
    logger.info(f"Enhanced detection: {filename} - prob: {final_prob:.3f}, detected: {cough_detected}, reason: {reason}")
    
    return final_prob, cough_detected, reason

def analyze_energy_patterns(y, sr):
    """Анализ энергетических паттернов характерных для кашля"""
    frame_len = int(0.02 * sr)
    hop_len = frame_len // 2
    
    energies = []
    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i:i + frame_len]
        energies.append(np.sqrt(np.mean(frame**2)))
    
    energies = np.array(energies)
    
    # Ищем резкие короткие всплески
    threshold = np.percentile(energies, 80)
    spikes = energies > threshold
    
    # Группируем смежные всплески
    cough_like_events = 0
    in_event = False
    event_start = 0
    
    for i, is_spike in enumerate(spikes):
        if is_spike and not in_event:
            in_event = True
            event_start = i
        elif not is_spike and in_event:
            in_event = False
            event_duration = (i - event_start) * (hop_len / sr)
            if 0.05 < event_duration < 1.0:
                cough_like_events += 1
    
    return {
        'valid_cough_like_events': cough_like_events,
        'total_spikes': np.sum(spikes),
        'max_energy': np.max(energies)
    }

def analyze_audio_enhanced(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    """УЛУЧШЕННЫЙ анализ аудио с гибридным подходом"""
    try:
        # Улучшенное декодирование
        decoding_result = decode_android_audio(audio_bytes, filename)
        y = decoding_result['audio']
        sr = decoding_result['sr']
        
        logger.info(f"Decoded: {len(y)} samples, SR: {sr}, method: {decoding_result['method']}")
        
        # Улучшенная обработка
        y_processed = enhanced_audio_processing(y, sr)
        
        # Анализ YAMNet
        scores, _, _ = run_yamnet(y_processed)
        
        # Топ классы
        mean_scores = np.mean(scores, axis=0)
        top5_idx = np.argsort(mean_scores)[-5:][::-1]
        top5 = [(CLASS_NAMES[i], float(mean_scores[i])) for i in top5_idx]
        
        # Улучшенный детектор кашля
        final_prob, detected, reason = aggressive_cough_detector_enhanced(y_processed, sr, scores, filename)
        
        # Детальная статистика
        cough_idxs = find_cough_indices()
        cough_stats = {}
        if cough_idxs:
            cough_scores = scores[:, cough_idxs]
            per_frame = np.max(cough_scores, axis=1)
            cough_stats = {
                "max_cough": float(np.max(per_frame)),
                "mean_cough": float(np.mean(per_frame)),
                "cough_frames": int(np.sum(per_frame > 0.05)),
                "total_frames": len(per_frame)
            }
        
        result = {
            "probability": round(final_prob, 3),
            "cough_detected": detected,
            "message": f"Enhanced detection: {reason}",
            "top_classes": top5,
            "cough_stats": cough_stats,
            "decoding_method": decoding_result['method'],
            "processing_applied": True
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        # Fallback на базовый анализ
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
        
        scores, _, _ = run_yamnet(y)
        mean_scores = np.mean(scores, axis=0)
        
        cough_idxs = find_cough_indices()
        cough_prob = np.max(mean_scores[cough_idxs]) if cough_idxs else 0.0
        
        return {
            "probability": round(float(cough_prob), 3),
            "cough_detected": cough_prob > 0.1,
            "message": "Fallback analysis",
            "top_classes": [],
            "cough_stats": {},
            "processing_applied": False
        }
        
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

def convert_numpy_types(obj):
    """Рекурсивно преобразует numpy типы в стандартные Python типы"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

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
        
        # УЛУЧШЕННЫЙ анализ
        result = analyze_audio_enhanced(raw, audio.filename)
        
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
            current_datetime  # ЯВНО УКАЗЫВАЕМ ВРЕМЯ СЕРВЕРА
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
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # ИСПОЛЬЗУЕМ ЕДИНУЮ ДАТУ СЕРВЕРА
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
        
        # Проверяем что вообще есть в базе
        cursor.execute('SELECT COUNT(*) FROM cough_records WHERE device_id=?', (device_id,))
        total_device_records = cursor.fetchone()[0] or 0
        logger.info(f"📊 Всего записей для устройства {device_id}: {total_device_records}")
        
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
            
            # Тренд (простая логика)
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "yamnet_loaded": YAMNET_LOADED,
        "timestamp": datetime.now().isoformat(),
        "upload_folder_size": sum(os.path.getsize(os.path.join(UPLOAD_FOLDER, f)) for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)))
    }

@app.post("/cleanup")
async def manual_cleanup():
    """Ручной запуск очистки"""
    cleanup_old_files()
    return {"status": "cleanup completed"}

# ---- Startup ----
@app.on_event("startup")
async def startup_event():
    """Запускается при старте сервера"""
    logger.info("🚀 Starting Enhanced Cough Detection Server")
    start_cleanup_scheduler()
    # Сразу запускаем очистку при старте
    cleanup_old_files()

# Добавь эти endpoint'ы после существующих

@app.get("/debug/db")
async def debug_db():
    """Отладочный endpoint для проверки базы данных"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Проверяем все записи
        cursor.execute('SELECT COUNT(*) as total, SUM(cough_detected) as coughs FROM cough_records')
        stats = cursor.fetchone()
        
        # Последние 5 записей
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

@app.get("/debug/stats/{device_id}")
async def debug_stats(device_id: str):
    """Отладочная версия статистики с подробными логами"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🔍 DEBUG STATS: device_id={device_id}, today={today}")
        
        # Проверяем какие записи есть в базе для этого устройства
        cursor.execute('''
            SELECT COUNT(*), device_id, DATE(timestamp) 
            FROM cough_records 
            WHERE device_id=? 
            GROUP BY device_id, DATE(timestamp)
        ''', (device_id,))
        device_stats = cursor.fetchall()
        
        logger.info(f"🔍 DEBUG: Записи для устройства {device_id}: {device_stats}")
        
        # Основная статистика за сегодня - ИСПРАВЛЕННЫЙ ЗАПРОС
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
        
        logger.info(f"🔍 DEBUG: Статистика сегодня - total: {total}, coughs: {total_coughs}, avg_prob: {avg_prob}")
        
        # Проверяем все записи за сегодня
        cursor.execute('''
            SELECT filename, cough_detected, probability, timestamp 
            FROM cough_records 
            WHERE device_id=? AND DATE(timestamp)=?
            ORDER BY timestamp DESC
        ''', (device_id, today))
        today_records = cursor.fetchall()
        
        logger.info(f"🔍 DEBUG: Записи за сегодня: {len(today_records)}")
        for record in today_records:
            logger.info(f"🔍 DEBUG: {record}")
        
        # Остальной код статистики...
        # Статистика по часам
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hr, COUNT(*) 
            FROM cough_records
            WHERE device_id=? AND cough_detected=1 AND DATE(timestamp)=?
            GROUP BY hr
        ''', (device_id, today))
        rows = cursor.fetchall()
        hourly = [{"hour": f"{h}:00", "count": c} for h, c in rows]
        
        # Заполняем все часы
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
        
        conn.close()
        
        result = {
            "today_stats": {
                "total_recordings": total,
                "total_coughs": total_coughs,
                "avg_probability": round(avg_prob, 3)
            },
            "hourly_stats": hourly,
            "recent_coughs": recent_coughs,
            "patterns": {
                "peak_hours": "Нет данных" if total_coughs == 0 else f"{hourly[0]['hour']} ({max([h['count'] for h in hourly])} раз)",
                "cough_frequency": f"{total_coughs} раз/день",
                "intensity": "Высокая" if avg_prob > 0.7 else "Средняя" if avg_prob > 0.3 else "Низкая",
                "trend": "📊"
            },
            "debug_info": {
                "device_id": device_id,
                "today": today,
                "today_records_count": len(today_records),
                "all_records_for_device": device_stats
            }
        }
        
        logger.info(f"🔍 DEBUG: Финальный результат: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"DEBUG Stats error: {e}")
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/debug/time")
async def debug_time():
    """Показать текущее время сервера"""
    return {
        "server_time": get_current_datetime(),
        "server_date": get_current_date(),
        "timezone": "Europe/Moscow"  # или тот что используешь
    }

if __name__ == "__main__":
    # Получаем порт из переменной окружения (Railway сам назначает)
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting enhanced server on 0.0.0.0:{port}, YAMNet loaded: {YAMNET_LOADED}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")