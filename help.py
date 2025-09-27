
"""
AURA Vision System v2.0 - Enhanced for Python 3.11.9
Advanced AI-powered face recognition and intelligent assistant system
Features: Real-time face recognition, emotion analysis, voice control, security monitoring
"""

import asyncio
import aiofiles
import concurrent.futures
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional, Callable, TypeVar, Generic
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports
import cv2
import numpy as np
import threading
import time
import json
import sqlite3
import logging
import requests
import face_recognition
import random
from datetime import datetime, timedelta
import pickle
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import hashlib
from collections import defaultdict, deque
import platform
import psutil
import subprocess
import sys
import os

# Enhanced imports for Python 3.11.9
try:
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Text-to-speech not available")

try:
    import speech_recognition as sr
    import pyaudio
    VOICE_RECOGNITION_AVAILABLE = True
except ImportError:
    VOICE_RECOGNITION_AVAILABLE = False
    print("Voice recognition not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available")

try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Type definitions
T = TypeVar('T')

class SecurityLevel(Enum):
    GUEST = 1
    MEMBER = 2
    TRUSTED = 3
    ADMIN = 4
    OWNER = 5

class EmotionState(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    FEAR = "fear"
    DISGUST = "disgust"
    CONFUSED = "confused"

class SystemStatus(Enum):
    INITIALIZING = auto()
    ACTIVE = auto()
    MONITORING = auto()
    ALERT = auto()
    MAINTENANCE = auto()
    SHUTDOWN = auto()

@dataclass
class FaceData:
    """Enhanced face data structure"""
    identity: Dict[str, Any]
    position: Tuple[int, int, int, int]
    confidence: float
    emotion: EmotionState
    age_estimate: int
    gender_estimate: str
    timestamp: float
    tracking_id: str
    stability_score: float
    facial_landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    security_clearance: SecurityLevel = SecurityLevel.GUEST

@dataclass
class PerformanceMetrics:
    """System performance tracking"""
    fps: float = 0.0
    frames_processed: int = 0
    faces_detected: int = 0
    recognitions_made: int = 0
    avg_processing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    last_updated: float = field(default_factory=time.time)

class AsyncDatabaseManager:
    """Enhanced async database manager with connection pooling"""
    
    def __init__(self, db_path: str = "aura_members.db", pool_size: int = 5):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self._lock = asyncio.Lock()
        self._pool: List[sqlite3.Connection] = []
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize database connection pool"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._pool.append(conn)
        
        # Initialize database schema
        self._init_schema()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        async with self._lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                # Create temporary connection if pool is empty
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        try:
            yield conn
        finally:
            async with self._lock:
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                else:
                    conn.close()
    
    def _init_schema(self):
        """Initialize enhanced database schema"""
        conn = self._pool[0]
        cursor = conn.cursor()
        
        # Enhanced members table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                age INTEGER,
                gender TEXT,
                relationship TEXT,
                face_encoding BLOB,
                face_embedding BLOB,
                image_path TEXT,
                access_level INTEGER DEFAULT 1,
                phone TEXT,
                email TEXT,
                notes TEXT,
                greeting_preference TEXT DEFAULT 'formal',
                security_clearance INTEGER DEFAULT 1,
                last_location TEXT,
                visit_frequency INTEGER DEFAULT 0,
                emergency_contact TEXT,
                biometric_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Enhanced visit logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration INTEGER DEFAULT 0,
                emotion_detected TEXT,
                confidence_score REAL,
                location TEXT,
                activity_type TEXT,
                weather_condition TEXT,
                device_used TEXT,
                FOREIGN KEY (member_id) REFERENCES members (id) ON DELETE CASCADE
            )
        ''')
        
        # System events with enhanced categorization
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_category TEXT DEFAULT 'general',
                event_data TEXT,
                severity TEXT DEFAULT 'info',
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES members (id)
            )
        ''')
        
        # Security incidents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_type TEXT NOT NULL,
                severity_level INTEGER DEFAULT 1,
                description TEXT,
                image_path TEXT,
                resolved BOOLEAN DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolved_by INTEGER,
                FOREIGN KEY (resolved_by) REFERENCES members (id)
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fps REAL,
                cpu_usage REAL,
                memory_usage REAL,
                gpu_usage REAL,
                frames_processed INTEGER,
                faces_detected INTEGER
            )
        ''')
        
        # Create optimized indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_members_name ON members(name)",
            "CREATE INDEX IF NOT EXISTS idx_members_active ON members(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_visits_member_timestamp ON visit_logs(member_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON system_events(event_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_security_severity ON security_incidents(severity_level, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)"
        ]
        
        for index in indexes:
            cursor.execute(index)
        
        conn.commit()

class AdvancedImageProcessor:
    """Enhanced image processing with multiple backends"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        
        # Initialize YOLO if available
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n-face.pt')  # Face detection model
            except:
                self.yolo_model = None
        else:
            self.yolo_model = None
        
        # Image enhancement pipeline
        self.enhancement_pipeline = self._create_enhancement_pipeline()
    
    def _create_enhancement_pipeline(self) -> List[Callable]:
        """Create image enhancement pipeline"""
        def denoise(img):
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        def enhance_contrast(img):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        def sharpen(img):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(img, -1, kernel)
        
        return [denoise, enhance_contrast, sharpen]
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply enhancement pipeline to image"""
        try:
            enhanced = image.copy()
            for enhancement in self.enhancement_pipeline:
                enhanced = enhancement(enhanced)
            return enhanced
        except Exception as e:
            logging.error(f"Image enhancement failed: {e}")
            return image
    
    def detect_faces_multi_backend(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using multiple backends for better accuracy"""
        faces = []
        
        # Method 1: face_recognition library
        try:
            face_locations = face_recognition.face_locations(image, model="cnn" if torch.cuda.is_available() else "hog")
            for (top, right, bottom, left) in face_locations:
                faces.append((left, top, right - left, bottom - top))
        except Exception as e:
            logging.warning(f"face_recognition detection failed: {e}")
        
        # Method 2: MediaPipe
        if MEDIAPIPE_AVAILABLE and self.face_detection:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_image)
                
                if results.detections:
                    h, w = image.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        faces.append((x, y, width, height))
            except Exception as e:
                logging.warning(f"MediaPipe detection failed: {e}")
        
        # Method 3: YOLO (if available)
        if self.yolo_model:
            try:
                results = self.yolo_model(image, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            faces.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            except Exception as e:
                logging.warning(f"YOLO detection failed: {e}")
        
        # Remove duplicates and merge nearby detections
        return self._merge_detections(faces)
    
    def _merge_detections(self, detections: List[Tuple[int, int, int, int]], 
                         overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping face detections"""
        if not detections:
            return []
        
        # Convert to (x1, y1, x2, y2) format for easier IoU calculation
        boxes = []
        for x, y, w, h in detections:
            boxes.append([x, y, x + w, y + h])
        
        boxes = np.array(boxes, dtype=np.float32)
        
        # Simple non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            [1.0] * len(boxes),  # All have same confidence
            score_threshold=0.0,
            nms_threshold=overlap_threshold
        )
        
        if len(indices) > 0:
            merged = []
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                merged.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            return merged
        
        return detections[:1]  # Return best detection if NMS fails

class EnhancedEmotionAnalyzer:
    """Advanced emotion analysis using multiple techniques"""
    
    def __init__(self):
        self.emotion_history = defaultdict(lambda: deque(maxlen=10))
        
        # Facial landmark indices for emotion detection
        self.emotion_landmarks = {
            'mouth_corners': [61, 291, 39, 269],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52],
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125]
        }
        
        # Load emotion classification model if available
        self._load_emotion_model()
    
    def _load_emotion_model(self):
        """Load pre-trained emotion recognition model"""
        try:
            if TF_AVAILABLE:
                # Try to load a lightweight emotion model
                # This is a placeholder - in practice you'd load a real model
                self.emotion_model = None
                logging.info("Emotion model placeholder loaded")
            else:
                self.emotion_model = None
        except Exception as e:
            logging.warning(f"Could not load emotion model: {e}")
            self.emotion_model = None
    
    def analyze_emotion(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> EmotionState:
        """Analyze emotion using multiple methods"""
        try:
            # Method 1: Landmark-based analysis
            if landmarks is not None and MEDIAPIPE_AVAILABLE:
                landmark_emotion = self._analyze_landmarks_emotion(landmarks)
            else:
                landmark_emotion = EmotionState.NEUTRAL
            
            # Method 2: Histogram and texture analysis
            texture_emotion = self._analyze_texture_emotion(face_image)
            
            # Method 3: Deep learning model (if available)
            if self.emotion_model is not None:
                model_emotion = self._analyze_model_emotion(face_image)
            else:
                model_emotion = EmotionState.NEUTRAL
            
            # Combine results with weighted voting
            emotions = [landmark_emotion, texture_emotion, model_emotion]
            emotion_weights = [0.4, 0.3, 0.3]
            
            # Simple voting mechanism
            emotion_scores = defaultdict(float)
            for emotion, weight in zip(emotions, emotion_weights):
                emotion_scores[emotion] += weight
            
            # Return emotion with highest score
            final_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            # Smooth emotion over time
            return self._smooth_emotion(final_emotion, face_image)
            
        except Exception as e:
            logging.error(f"Emotion analysis failed: {e}")
            return EmotionState.NEUTRAL
    
    def _analyze_landmarks_emotion(self, landmarks: np.ndarray) -> EmotionState:
        """Analyze emotion based on facial landmarks"""
        try:
            # This is a simplified implementation
            # In practice, you'd use more sophisticated geometric analysis
            
            # Calculate mouth curvature
            mouth_points = landmarks[self.emotion_landmarks['mouth_corners']]
            mouth_curve = np.mean(mouth_points[:, 1])  # Y coordinates
            
            # Calculate eyebrow position
            eyebrow_points = landmarks[self.emotion_landmarks['eyebrows']]
            eyebrow_height = np.mean(eyebrow_points[:, 1])
            
            # Simple heuristic classification
            if mouth_curve > 0.1:
                return EmotionState.HAPPY
            elif mouth_curve < -0.1:
                return EmotionState.SAD
            elif eyebrow_height < -0.1:
                return EmotionState.ANGRY
            else:
                return EmotionState.NEUTRAL
                
        except Exception as e:
            logging.error(f"Landmark emotion analysis failed: {e}")
            return EmotionState.NEUTRAL
    
    def _analyze_texture_emotion(self, face_image: np.ndarray) -> EmotionState:
        """Analyze emotion based on image texture and patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate local binary patterns
            def lbp(image):
                h, w = image.shape
                lbp_image = np.zeros_like(image)
                
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        center = image[i, j]
                        code = 0
                        code |= (image[i-1, j-1] >= center) << 7
                        code |= (image[i-1, j] >= center) << 6
                        code |= (image[i-1, j+1] >= center) << 5
                        code |= (image[i, j+1] >= center) << 4
                        code |= (image[i+1, j+1] >= center) << 3
                        code |= (image[i+1, j] >= center) << 2
                        code |= (image[i+1, j-1] >= center) << 1
                        code |= (image[i, j-1] >= center) << 0
                        lbp_image[i, j] = code
                
                return lbp_image
            
            lbp_image = lbp(gray)
            
            # Calculate histogram features
            hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
            hist_features = hist.flatten()
            
            # Simple classification based on texture patterns
            uniformity = np.sum(hist_features ** 2)
            contrast = np.var(hist_features)
            
            if uniformity > 1000 and contrast > 500:
                return EmotionState.SURPRISED
            elif uniformity < 500:
                return EmotionState.SAD
            elif contrast > 800:
                return EmotionState.ANGRY
            else:
                return EmotionState.NEUTRAL
                
        except Exception as e:
            logging.error(f"Texture emotion analysis failed: {e}")
            return EmotionState.NEUTRAL
    
    def _analyze_model_emotion(self, face_image: np.ndarray) -> EmotionState:
        """Analyze emotion using deep learning model"""
        try:
            if self.emotion_model is None:
                return EmotionState.NEUTRAL
            
            # Preprocess image for model
            resized = cv2.resize(face_image, (48, 48))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized = gray / 255.0
            input_data = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
            
            # Predict emotion
            predictions = self.emotion_model.predict(input_data)
            emotion_classes = list(EmotionState)
            predicted_emotion = emotion_classes[np.argmax(predictions)]
            
            return predicted_emotion
            
        except Exception as e:
            logging.error(f"Model emotion analysis failed: {e}")
            return EmotionState.NEUTRAL
    
    def _smooth_emotion(self, current_emotion: EmotionState, face_key: str) -> EmotionState:
        """Smooth emotion predictions over time"""
        try:
            history = self.emotion_history[face_key]
            history.append(current_emotion)
            
            if len(history) < 3:
                return current_emotion
            
            # Count occurrences of each emotion in recent history
            emotion_counts = defaultdict(int)
            for emotion in list(history)[-5:]:  # Last 5 frames
                emotion_counts[emotion] += 1
            
            # Return most frequent emotion
            return max(emotion_counts.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logging.error(f"Emotion smoothing failed: {e}")
            return current_emotion

class IntelligentVoiceAssistant:
    """Enhanced voice assistant with NLP and contextual understanding"""
    
    def __init__(self):
        self.tts_engine = self._initialize_tts()
        self.speech_queue = asyncio.Queue()
        self.is_speaking = False
        self._speech_lock = asyncio.Lock()
        
        # Enhanced personality system
        self.personality = {
            'formality_level': 0.7,
            'enthusiasm': 0.8,
            'humor': 0.3,
            'empathy': 0.9
        }
        
        # Context-aware responses
        self.context_memory = deque(maxlen=50)
        self.user_preferences = {}
        
        # Multi-language support
        self.languages = {
            'en': 'english',
            'hi': 'hindi',
            'mixed': 'hinglish'
        }
        
        # Advanced response templates
        self._load_response_templates()
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine with enhanced settings"""
        if not SPEECH_AVAILABLE:
            return None
        
        try:
            engine = pyttsx3.init()
            
            # Enhanced voice settings
            voices = engine.getProperty('voices')
            if voices:
                # Prefer female voices for better user experience
                female_voices = [v for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()]
                if female_voices:
                    engine.setProperty('voice', female_voices[0].id)
                else:
                    engine.setProperty('voice', voices[0].id)
            
            # Optimized speech parameters
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 0.9)
            
            return engine
        except Exception as e:
            logging.error(f"TTS initialization failed: {e}")
            return None
    
    def _load_response_templates(self):
        """Load context-aware response templates"""
        self.templates = {
            'greeting': {
                'family': {
                    'en': ["Welcome home {name}! How was your day?", "Hi {name}! Great to see you back."],
                    'hi': ["घर वापसी मुबारक {name}! आपका दिन कैसा रहा?", "नमस्ते {name}! आपको देखकर खुशी हुई।"],
                    'mixed': ["Welcome home {name}! कैसा रहा आपका दिन?", "Hi {name}! मिलकर खुशी हुई।"]
                },
                'guest': {
                    'en': ["Welcome to our home! I'm AURA, your intelligent assistant.", "Hello! Please make yourself comfortable."],
                    'hi': ["हमारे घर में आपका स्वागत है! मैं AURA हूँ।", "नमस्ते! कृपया घर समझकर बैठिए।"],
                    'mixed': ["Welcome! मैं AURA हूँ, आपकी सेवा में।"]
                }
            },
            'emotion_response': {
                EmotionState.HAPPY: {
                    'en': ["You look wonderful today!", "Your smile brightens the room!"],
                    'hi': ["आज आप बहुत खुश लग रहे हैं!", "आपकी मुस्कान से कमरा रोशन हो गया!"],
                    'mixed': ["You look happy! आपकी खुशी देखकर अच्छा लगा।"]
                },
                EmotionState.SAD: {
                    'en': ["Is everything alright? I'm here if you need to talk.", "You seem a bit down. Can I help?"],
                    'hi': ["क्या सब ठीक है? अगर बात करना चाहें तो मैं हूँ।", "आप उदास लग रहे हैं। कुछ मदद चाहिए?"],
                    'mixed': ["आप परेशान लग रहे हैं। Is everything okay?"]
                }
            }
        }
    
    async def speak_async(self, text: str, emotion: EmotionState = EmotionState.NEUTRAL, 
                         priority: bool = False, language: str = 'mixed'):
        """Asynchronous speech with emotion and priority handling"""
        if not self.tts_engine:
            print(f"AURA: {text}")
            return
        
        speech_item = {
            'text': text,
            'emotion': emotion,
            'language': language,
            'priority': priority,
            'timestamp': time.time()
        }
        
        if priority:
            # Clear queue for high priority messages
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        await self.speech_queue.put(speech_item)
        
        # Process speech queue
        await self._process_speech_queue()
    
    async def _process_speech_queue(self):
        """Process speech queue asynchronously"""
        async with self._speech_lock:
            if self.is_speaking:
                return
            
            try:
                speech_item = self.speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            
            self.is_speaking = True
            
            try:
                # Add emotional inflection
                text = self._add_emotional_inflection(speech_item['text'], speech_item['emotion'])
                
                # Adjust speech parameters based on emotion
                self._adjust_speech_params(speech_item['emotion'])
                
                print(f"AURA ({speech_item['emotion'].value}): {text}")
                
                # Synthesize speech in thread to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._synthesize_speech, text)
                
                # Store in context memory
                self.context_memory.append({
                    'type': 'speech',
                    'content': text,
                    'emotion': speech_item['emotion'],
                    'timestamp': time.time()
                })
                
            finally:
                self.is_speaking = False
                
                # Process next item if queue not empty
                if not self.speech_queue.empty():
                    await self._process_speech_queue()
    
    def _add_emotional_inflection(self, text: str, emotion: EmotionState) -> str:
        """Add emotional context to speech"""
        if emotion == EmotionState.HAPPY:
            return f"😊 {text}"
        elif emotion == EmotionState.SAD:
            return f"😔 {text}"
        elif emotion == EmotionState.SURPRISED:
            return f"😲 {text}!"
        elif emotion == EmotionState.ANGRY:
            return f"😠 {text}"
        return text
    
    def _adjust_speech_params(self, emotion: EmotionState):
        """Adjust speech parameters based on emotion"""
        if not self.tts_engine:
            return
        
        if emotion == EmotionState.HAPPY:
            self.tts_engine.setProperty('rate', 170)
            self.tts_engine.setProperty('volume', 0.95)
        elif emotion == EmotionState.SAD:
            self.tts_engine.setProperty('rate', 140)
            self.tts_engine.setProperty('volume', 0.8)
        elif emotion == EmotionState.ANGRY:
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.9)
        elif emotion == EmotionState.SURPRISED:
            self.tts_engine.setProperty('rate', 190)
            self.tts_engine.setProperty('volume', 1.0)
        else:  # NEUTRAL
            self.tts_engine.setProperty('rate', 160)
            self.tts_engine.setProperty('volume', 0.9)
    
    def _synthesize_speech(self, text: str):
        """Synthesize speech in separate thread"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"Speech synthesis error: {e}")
    
    def generate_contextual_response(self, person_name: str, emotion: EmotionState, 
                                   relationship: str = 'guest', language: str = 'mixed') -> str:
        """Generate contextual response based on person and situation"""
        try:
            # Determine appropriate template category
            if relationship.lower() in ['family', 'parent', 'child', 'sibling', 'spouse']:
                category = 'family'
            else:
                category = 'guest'
            
            # Get base greeting
            greeting_templates = self.templates['greeting'][category][language]
            greeting = random.choice(greeting_templates).format(name=person_name)
            
            # Add emotion-specific response
            if emotion != EmotionState.NEUTRAL and emotion in self.templates['emotion_response']:
                emotion_templates = self.templates['emotion_response'][emotion][language]
                emotion_response = random.choice(emotion_templates)
                greeting += " " + emotion_response
            
            return greeting
            
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return f"Hello {person_name}!"

class AdvancedSecuritySystem:
    """Enhanced security system with threat assessment and alerts"""
    
    def __init__(self):
        self.threat_levels = {
            SecurityLevel.GUEST: 1,
            SecurityLevel.MEMBER: 2,
            SecurityLevel.TRUSTED: 3,
            SecurityLevel.ADMIN: 4,
            SecurityLevel.OWNER: 5
        }
        
        self.security_events = deque(maxlen=1000)
        self.unknown_faces_tracker = defaultdict(lambda: {'count': 0, 'first_seen': 0, 'last_seen': 0})
        self.alert_cooldown = {}
        self.suspicious_patterns = []
        
        # Advanced threat detection parameters
        self.config = {
            'max_unknown_threshold': 3,
            'suspicious_behavior_window': 300,  # 5 minutes
            'alert_escalation_time': 600,  # 10 minutes
            'facial_occlusion_threshold': 0.3,
            'unusual_hours_start': 22,  # 10 PM
            'unusual_hours_end': 6,     # 6 AM
        }
    
    def assess_security_risk(self, face_data: FaceData) -> Dict[str, Any]:
        """Comprehensive security risk assessment"""
        risk_factors = {
            'unknown_person': 0,
            'facial_occlusion': 0,
            'unusual_time': 0,
            'suspicious_behavior': 0,
            'low_confidence': 0
        }
        
        current_time = datetime.now()
        
        # Unknown person risk
        if face_data.identity['name'] == 'Unknown':
            face_key = self._generate_face_key(face_data)
            self.unknown_faces_tracker[face_key]['count'] += 1
            self.unknown_faces_tracker[face_key]['last_seen'] = time.time()
            
            if self.unknown_faces_tracker[face_key]['first_seen'] == 0:
                self.unknown_faces_tracker[face_key]['first_seen'] = time.time()
            
            if self.unknown_faces_tracker[face_key]['count'] > self.config['max_unknown_threshold']:
                risk_factors['unknown_person'] = 0.8
            else:
                risk_factors['unknown_person'] = 0.4
        
        # Facial occlusion detection
        occlusion_score = self._detect_facial_occlusion(face_data)
        if occlusion_score > self.config['facial_occlusion_threshold']:
            risk_factors['facial_occlusion'] = occlusion_score
        
        # Unusual time detection
        hour = current_time.hour
        if (hour >= self.config['unusual_hours_start'] or 
            hour <= self.config['unusual_hours_end']):
            risk_factors['unusual_time'] = 0.6
        
        # Low confidence detection
        if face_data.confidence < 0.5:
            risk_factors['low_confidence'] = 0.3
        
        # Calculate overall risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        risk_level = self._categorize_risk(risk_score)
        
        # Log security event
        security_event = {
            'timestamp': current_time.isoformat(),
            'face_id': face_data.tracking_id,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'location': 'main_camera',
            'person_name': face_data.identity.get('name', 'Unknown')
        }
        
        self.security_events.append(security_event)
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommended_action': self._get_recommended_action(risk_level),
            'alert_required': risk_score > 0.6
        }
    
    def _generate_face_key(self, face_data: FaceData) -> str:
        """Generate unique key for face tracking"""
        # Use position and basic features to create a tracking key
        x, y, w, h = face_data.position
        return f"unknown_{x}_{y}_{w}_{h}_{face_data.age_estimate}"
    
    def _detect_facial_occlusion(self, face_data: FaceData) -> float:
        """Detect if face is partially occluded (mask, sunglasses, etc.)"""
        try:
            # This is a simplified implementation
            # In practice, you'd use more sophisticated detection
            
            if face_data.facial_landmarks is not None:
                # Check if key facial features are visible
                visible_features = 0
                total_features = 4  # eyes, nose, mouth
                
                # This would involve checking landmark visibility
                # Placeholder implementation
                occlusion_score = random.uniform(0.0, 0.5)  # Simplified
                return occlusion_score
            
            return 0.0
        except Exception as e:
            logging.error(f"Occlusion detection error: {e}")
            return 0.0
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level based on score"""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        elif risk_score >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level"""
        actions = {
            'CRITICAL': 'IMMEDIATE_ALERT_AND_RECORD',
            'HIGH': 'ALERT_AND_MONITOR',
            'MEDIUM': 'ENHANCED_MONITORING',
            'LOW': 'STANDARD_MONITORING',
            'MINIMAL': 'NO_ACTION'
        }
        return actions.get(risk_level, 'STANDARD_MONITORING')
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        current_time = time.time()
        recent_events = [
            event for event in self.security_events 
            if current_time - datetime.fromisoformat(event['timestamp']).timestamp() < 86400  # Last 24 hours
        ]
        
        # Analyze patterns
        risk_distribution = defaultdict(int)
        for event in recent_events:
            risk_distribution[event['risk_level']] += 1
        
        # Calculate security metrics
        total_events = len(recent_events)
        high_risk_events = risk_distribution['HIGH'] + risk_distribution['CRITICAL']
        security_score = max(0, 100 - (high_risk_events / max(1, total_events)) * 100)
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'total_events_24h': total_events,
            'risk_distribution': dict(risk_distribution),
            'security_score': security_score,
            'unknown_faces_detected': len(self.unknown_faces_tracker),
            'recommendations': self._generate_recommendations(recent_events),
            'system_status': 'SECURE' if security_score > 80 else 'ATTENTION_NEEDED'
        }
    
    def _generate_recommendations(self, recent_events: List[Dict]) -> List[str]:
        """Generate security recommendations based on recent events"""
        recommendations = []
        
        high_risk_count = sum(1 for event in recent_events if event['risk_level'] in ['HIGH', 'CRITICAL'])
        unknown_count = sum(1 for event in recent_events if event['person_name'] == 'Unknown')
        
        if high_risk_count > 5:
            recommendations.append("Consider reviewing access controls and security protocols")
        
        if unknown_count > 10:
            recommendations.append("High number of unknown visitors detected - consider visitor registration system")
        
        if not recommendations:
            recommendations.append("Security status normal - continue monitoring")
        
        return recommendations

class EnhancedFaceRecognitionSystem:
    """Advanced face recognition with multiple algorithms and optimization"""
    
    def __init__(self):
        self.database = AsyncDatabaseManager()
        self.image_processor = AdvancedImageProcessor()
        self.emotion_analyzer = EnhancedEmotionAnalyzer()
        self.security_system = AdvancedSecuritySystem()
        self.voice_assistant = IntelligentVoiceAssistant()
        
        # Face tracking and matching
        self.face_trackers = {}
        self.known_encodings = []
        self.known_names = []
        self.known_metadata = []
        
        # Performance optimization
        self.face_cache = {}
        self.encoding_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 300  # 5 minutes
        
        # Advanced matching parameters
        self.recognition_config = {
            'tolerance': 0.6,
            'model': 'large',  # 'small', 'large'
            'distance_metric': 'euclidean',  # 'euclidean', 'cosine'
            'min_confidence': 0.4,
            'stability_threshold': 0.7
        }
        
        # Load known faces
        asyncio.create_task(self._load_known_faces())
    
    async def _load_known_faces(self):
        """Load known faces from database asynchronously"""
        try:
            async with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, face_encoding, face_embedding, relationship, 
                           access_level, security_clearance, greeting_preference 
                    FROM members WHERE is_active = 1
                """)
                
                members = cursor.fetchall()
                
                self.known_encodings = []
                self.known_names = []
                self.known_metadata = []
                
                for member in members:
                    member_id, name, encoding_blob, embedding_blob, relationship, \
                    access_level, security_clearance, greeting_pref = member
                    
                    if encoding_blob:
                        try:
                            encoding = pickle.loads(encoding_blob)
                            self.known_encodings.append(encoding)
                            self.known_names.append(name)
                            self.known_metadata.append({
                                'id': member_id,
                                'name': name,
                                'relationship': relationship,
                                'access_level': access_level,
                                'security_clearance': SecurityLevel(security_clearance),
                                'greeting_preference': greeting_pref
                            })
                        except Exception as e:
                            logging.error(f"Failed to load encoding for {name}: {e}")
                
                logging.info(f"Loaded {len(self.known_encodings)} known faces")
                
        except Exception as e:
            logging.error(f"Failed to load known faces: {e}")
    
    async def process_faces(self, frame: np.ndarray) -> List[FaceData]:
        """Process faces in frame with advanced recognition"""
        try:
            # Enhance image quality
            enhanced_frame = self.image_processor.enhance_image(frame)
            
            # Detect faces using multiple methods
            face_locations = self.image_processor.detect_faces_multi_backend(enhanced_frame)
            
            if not face_locations:
                return []
            
            # Extract face encodings
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                [(y, x+w, y+h, x) for x, y, w, h in face_locations],
                model=self.recognition_config['model']
            )
            
            results = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Process individual face
                face_data = await self._process_single_face(
                    enhanced_frame, rgb_frame, face_location, face_encoding, i
                )
                results.append(face_data)
            
            return results
            
        except Exception as e:
            logging.error(f"Face processing error: {e}")
            return []
    
    async def _process_single_face(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                 location: Tuple[int, int, int, int], 
                                 encoding: np.ndarray, face_id: int) -> FaceData:
        """Process individual face with comprehensive analysis"""
        x, y, w, h = location
        face_img = frame[y:y+h, x:x+w]
        
        # Face recognition
        identity = await self._recognize_face(encoding, face_img)
        
        # Extract facial landmarks if MediaPipe is available
        landmarks = None
        if MEDIAPIPE_AVAILABLE:
            landmarks = self._extract_facial_landmarks(rgb_frame[y:y+h, x:x+w])
        
        # Emotion analysis
        emotion = self.emotion_analyzer.analyze_emotion(face_img, landmarks)
        
        # Age and gender estimation (enhanced)
        age, gender = self._estimate_demographics(face_img, landmarks)
        
        # Generate tracking ID
        tracking_id = f"face_{face_id}_{int(time.time())}"
        
        # Calculate stability score
        stability_score = self._calculate_face_stability(tracking_id, location)
        
        # Create face data object
        face_data = FaceData(
            identity=identity,
            position=location,
            confidence=identity.get('confidence', 0.0),
            emotion=emotion,
            age_estimate=age,
            gender_estimate=gender,
            timestamp=time.time(),
            tracking_id=tracking_id,
            stability_score=stability_score,
            facial_landmarks=landmarks,
            embedding=encoding,
            security_clearance=SecurityLevel(identity.get('security_clearance', 1))
        )
        
        # Security assessment
        security_assessment = self.security_system.assess_security_risk(face_data)
        
        # Handle recognition result
        await self._handle_face_recognition(face_data, security_assessment)
        
        return face_data
    
    async def _recognize_face(self, face_encoding: np.ndarray, face_img: np.ndarray) -> Dict[str, Any]:
        """Advanced face recognition with multiple matching strategies"""
        try:
            # Check if we have known faces loaded
            if not self.known_encodings:
                return await self._handle_unknown_face(face_img)
            
            # Calculate distances using configured metric
            if self.recognition_config['distance_metric'] == 'euclidean':
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            else:  # cosine similarity
                distances = self._calculate_cosine_distances(self.known_encodings, face_encoding)
            
            # Find best matches within tolerance
            tolerance = self.recognition_config['tolerance']
            matches = distances <= tolerance
            
            if not any(matches):
                return await self._handle_unknown_face(face_img)
            
            # Get best match
            best_match_idx = np.argmin(distances)
            if matches[best_match_idx]:
                confidence = float(1.0 - distances[best_match_idx])
                
                if confidence >= self.recognition_config['min_confidence']:
                    metadata = self.known_metadata[best_match_idx]
                    
                    # Update database
                    await self._update_member_visit(metadata['id'])
                    
                    return {
                        'name': metadata['name'],
                        'relationship': metadata['relationship'],
                        'confidence': confidence,
                        'source': 'local_database',
                        'member_id': metadata['id'],
                        'access_level': metadata['access_level'],
                        'security_clearance': metadata['security_clearance'].value,
                        'greeting_preference': metadata['greeting_preference']
                    }
            
            return await self._handle_unknown_face(face_img)
            
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return await self._handle_unknown_face(face_img)
    
    def _calculate_cosine_distances(self, known_encodings: List[np.ndarray], 
                                  target_encoding: np.ndarray) -> np.ndarray:
        """Calculate cosine distances for face matching"""
        try:
            known_encodings = np.array(known_encodings)
            target_encoding = np.array(target_encoding)
            
            # Normalize vectors
            known_encodings = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
            target_encoding = target_encoding / np.linalg.norm(target_encoding)
            
            # Calculate cosine similarity
            similarities = np.dot(known_encodings, target_encoding)
            
            # Convert to distances (1 - similarity)
            distances = 1.0 - similarities
            
            return distances
            
        except Exception as e:
            logging.error(f"Cosine distance calculation error: {e}")
            return np.array([1.0] * len(known_encodings))
    
    async def _handle_unknown_face(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Handle unknown face detection"""
        return {
            'name': 'Unknown',
            'relationship': 'Unknown',
            'confidence': 0.0,
            'source': 'unknown',
            'member_id': None,
            'access_level': 1,
            'security_clearance': 1,
            'greeting_preference': 'formal'
        }
    
    def _extract_facial_landmarks(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks using MediaPipe"""
        try:
            if not MEDIAPIPE_AVAILABLE:
                return None
            
            results = self.image_processor.face_mesh.process(face_img)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                # Convert to numpy array
                landmark_points = []
                for landmark in landmarks.landmark:
                    landmark_points.append([landmark.x, landmark.y, landmark.z])
                return np.array(landmark_points)
            
            return None
            
        except Exception as e:
            logging.error(f"Landmark extraction error: {e}")
            return None
    
    def _estimate_demographics(self, face_img: np.ndarray, 
                             landmarks: Optional[np.ndarray] = None) -> Tuple[int, str]:
        """Enhanced age and gender estimation"""
        try:
            # Improve this with actual ML models
            # For now, using enhanced heuristics
            
            h, w = face_img.shape[:2]
            
            # Analyze face structure
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Texture analysis for age estimation
            # Calculate wrinkle density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            # Skin smoothness analysis
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            smoothness = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            
            # Age estimation based on multiple factors
            if edge_density > 0.15 or smoothness > 20:
                age = random.randint(45, 70)  # Older
            elif edge_density > 0.08 or smoothness > 12:
                age = random.randint(30, 50)  # Middle-aged
            else:
                age = random.randint(18, 35)  # Younger
            
            # Gender estimation (simplified - in practice use proper ML model)
            # Analyze facial proportions
            if landmarks is not None and len(landmarks) > 100:
                # Use landmark ratios for better estimation
                # This is still simplified
                jaw_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
                face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
                ratio = jaw_width / face_height if face_height > 0 else 1
                
                gender = 'Male' if ratio > 0.75 else 'Female'
            else:
                gender = random.choice(['Male', 'Female'])
            
            return age, gender
            
        except Exception as e:
            logging.error(f"Demographics estimation error: {e}")
            return 30, 'Unknown'
    
    def _calculate_face_stability(self, tracking_id: str, 
                                location: Tuple[int, int, int, int]) -> float:
        """Calculate face tracking stability score"""
        try:
            if tracking_id not in self.face_trackers:
                self.face_trackers[tracking_id] = deque(maxlen=10)
            
            self.face_trackers[tracking_id].append({
                'location': location,
                'timestamp': time.time()
            })
            
            tracker_data = list(self.face_trackers[tracking_id])
            
            if len(tracker_data) < 3:
                return 0.5
            
            # Calculate position variance
            positions = [data['location'] for data in tracker_data]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            x_variance = np.var(x_coords)
            y_variance = np.var(y_coords)
            
            # Calculate temporal consistency
            timestamps = [data['timestamp'] for data in tracker_data]
            time_diffs = np.diff(timestamps)
            time_consistency = 1.0 / (1.0 + np.var(time_diffs))
            
            # Combine metrics
            position_stability = 1.0 / (1.0 + (x_variance + y_variance) / 10000.0)
            
            stability_score = (position_stability + time_consistency) / 2.0
            return min(1.0, max(0.0, stability_score))
            
        except Exception as e:
            logging.error(f"Stability calculation error: {e}")
            return 0.5
    
    async def _update_member_visit(self, member_id: int):
        """Update member visit information in database"""
        try:
            async with self.database.get_connection() as conn:
                cursor = conn.cursor()
                current_time = datetime.now().isoformat()
                
                # Update last seen and visit frequency
                cursor.execute("""
                    UPDATE members 
                    SET last_seen = ?, visit_frequency = visit_frequency + 1, updated_at = ?
                    WHERE id = ?
                """, (current_time, current_time, member_id))
                
                # Log the visit
                cursor.execute("""
                    INSERT INTO visit_logs (member_id, emotion_detected, location, device_used)
                    VALUES (?, ?, ?, ?)
                """, (member_id, None, 'main_camera', 'aura_vision_v2'))
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Database update error: {e}")
    
    async def _handle_face_recognition(self, face_data: FaceData, security_assessment: Dict[str, Any]):
        """Handle face recognition result with appropriate actions"""
        try:
            person_name = face_data.identity['name']
            
            # Generate appropriate response
            if person_name != 'Unknown' and face_data.stability_score > self.recognition_config['stability_threshold']:
                # Known person with stable detection
                response = self.voice_assistant.generate_contextual_response(
                    person_name, 
                    face_data.emotion,
                    face_data.identity.get('relationship', 'guest')
                )
                
                # Announce with appropriate emotion
                await self.voice_assistant.speak_async(response, face_data.emotion)
                
                # Log successful recognition
                logging.info(f"Recognized {person_name} with confidence {face_data.confidence:.2f}")
                
            elif security_assessment['alert_required']:
                # Security alert required
                alert_message = "Security alert: Unknown person detected. Enhanced monitoring activated."
                await self.voice_assistant.speak_async(
                    alert_message, 
                    EmotionState.NEUTRAL, 
                    priority=True
                )
                
                # Log security event
                logging.warning(f"Security alert triggered for unknown person")
            
        except Exception as e:
            logging.error(f"Face recognition handling error: {e}")

class AURAVisionSystemV2:
    """Main AURA Vision System V2.0 with enhanced capabilities"""
    
    def __init__(self):
        self.system_status = SystemStatus.INITIALIZING
        self.performance_metrics = PerformanceMetrics()
        
        # Initialize core components
        self.face_recognition_system = EnhancedFaceRecognitionSystem()
        self.voice_assistant = IntelligentVoiceAssistant()
        self.security_system = AdvancedSecuritySystem()
        
        # Camera and video processing
        self.camera = None
        self.camera_index = 0
        self.target_fps = 30
        self.frame_skip = 2  # Process every nth frame for performance
        self.current_frame_count = 0
        
        # Async processing
        self.processing_queue = asyncio.Queue(maxsize=10)
        self.result_queue = asyncio.Queue(maxsize=100)
        
        # System configuration
        self.config = {
            'display_enabled': True,
            'recording_enabled': False,
            'performance_monitoring': True,
            'auto_learning': False,
            'debug_mode': False
        }
        
        # Performance monitoring
        self.fps_calculator = self._FPSCalculator()
        self.performance_monitor_task = None
        
        # Command system
        self.command_handlers = self._initialize_command_handlers()
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
    
    class _FPSCalculator:
        """Internal FPS calculation helper"""
        def __init__(self, window_size: int = 30):
            self.timestamps = deque(maxlen=window_size)
        
        def update(self) -> float:
            current_time = time.time()
            self.timestamps.append(current_time)
            
            if len(self.timestamps) < 2:
                return 0.0
            
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed == 0:
                return 0.0
                
            return (len(self.timestamps) - 1) / elapsed
    
    def _initialize_command_handlers(self) -> Dict[str, Callable]:
        """Initialize voice command handlers"""
        return {
            'add member': self._cmd_add_member,
            'list members': self._cmd_list_members,
            'system status': self._cmd_system_status,
            'security report': self._cmd_security_report,
            'performance stats': self._cmd_performance_stats,
            'toggle recording': self._cmd_toggle_recording,
            'shutdown': self._cmd_shutdown,
            'restart': self._cmd_restart,
            'help': self._cmd_help
        }
    
    async def initialize(self):
        """Initialize the AURA system asynchronously"""
        try:
            logging.info("Initializing AURA Vision System V2.0...")
            
            # Initialize camera
            await self._initialize_camera()
            
            # Start performance monitoring
            if self.config['performance_monitoring']:
                self.performance_monitor_task = asyncio.create_task(
                    self._performance_monitor_loop()
                )
            
            # System ready
            self.system_status = SystemStatus.ACTIVE
            
            # Welcome message
            await self.voice_assistant.speak_async(
                "AURA Vision System 2.0 activated. Enhanced AI capabilities online.",
                EmotionState.HAPPY,
                priority=True
            )
            
            logging.info("AURA Vision System V2.0 initialized successfully")
            
        except Exception as e:
            logging.error(f"System initialization failed: {e}")
            self.system_status = SystemStatus.SHUTDOWN
            raise
    
    async def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Optimal camera settings for performance
            self.camera.set(cv2.CAP_PROP_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Enable auto-exposure and auto-focus if available
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            
            # Verify camera settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            logging.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            raise
    
    async def run(self):
        """Main system loop with async processing"""
        try:
            self.system_status = SystemStatus.MONITORING
            logging.info("AURA Vision System V2.0 started")
            
            # Start processing tasks
            frame_processor_task = asyncio.create_task(self._frame_processing_loop())
            display_task = asyncio.create_task(self._display_loop())
            command_processor_task = asyncio.create_task(self._command_processing_loop())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cancel all tasks
            frame_processor_task.cancel()
            display_task.cancel()
            command_processor_task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(
                frame_processor_task, 
                display_task, 
                command_processor_task,
                return_exceptions=True
            )
            
        except Exception as e:
            logging.error(f"System runtime error: {e}")
        finally:
            await self._cleanup()
    
    async def _frame_processing_loop(self):
        """Async frame processing loop"""
        try:
            while not self.shutdown_event.is_set():
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    await asyncio.sleep(0.1)
                    continue
                
                # Skip frames for performance optimization
                self.current_frame_count += 1
                if self.current_frame_count % self.frame_skip != 0:
                    continue
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add to processing queue (non-blocking)
                try:
                    self.processing_queue.put_nowait({
                        'frame': frame.copy(),
                        'timestamp': time.time(),
                        'frame_id': self.current_frame_count
                    })
                except asyncio.QueueFull:
                    # Skip frame if queue is full
                    pass
                
                # Process frames asynchronously
                if not self.processing_queue.empty():
                    asyncio.create_task(self._process_frame_async())
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(1.0 / self.target_fps)
                
        except Exception as e:
            logging.error(f"Frame processing loop error: {e}")
    
    async def _process_frame_async(self):
        """Process individual frame asynchronously"""
        try:
            frame_data = await self.processing_queue.get()
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            start_time = time.time()
            
            # Face detection and analysis
            faces_data = await self.face_recognition_system.process_faces(frame)
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics.frames_processed += 1
            self.performance_metrics.faces_detected += len(faces_data)
            self.performance_metrics.fps = self.fps_calculator.update()
            
            # Weighted average for processing time
            if self.performance_metrics.avg_processing_time == 0:
                self.performance_metrics.avg_processing_time = processing_time
            else:
                alpha = 0.1  # Smoothing factor
                self.performance_metrics.avg_processing_time = (
                    alpha * processing_time + 
                    (1 - alpha) * self.performance_metrics.avg_processing_time
                )
            
            # Add result to display queue
            try:
                await self.result_queue.put({
                    'frame': frame,
                    'faces': faces_data,
                    'timestamp': timestamp,
                    'processing_time': processing_time
                })
            except asyncio.QueueFull:
                # Remove oldest result if queue is full
                try:
                    self.result_queue.get_nowait()
                    await self.result_queue.put({
                        'frame': frame,
                        'faces': faces_data,
                        'timestamp': timestamp,
                        'processing_time': processing_time
                    })
                except asyncio.QueueEmpty:
                    pass
            
        except Exception as e:
            logging.error(f"Async frame processing error: {e}")
    
    async def _display_loop(self):
        """Async display loop"""
        if not self.config['display_enabled']:
            return
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get latest result
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                    
                    # Create annotated frame
                    annotated_frame = self._create_annotated_frame(
                        result['frame'], 
                        result['faces']
                    )
                    
                    # Display frame
                    cv2.imshow('AURA Vision System V2.0', annotated_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.shutdown_event.set()
                    elif key == ord('r'):
                        self.config['recording_enabled'] = not self.config['recording_enabled']
                        status = "enabled" if self.config['recording_enabled'] else "disabled"
                        await self.voice_assistant.speak_async(f"Recording {status}")
                    elif key == ord('d'):
                        self.config['debug_mode'] = not self.config['debug_mode']
                    elif key == ord('h'):
                        await self._show_help()
                    
                except asyncio.TimeoutError:
                    # No new results, continue
                    continue
                    
        except Exception as e:
            logging.error(f"Display loop error: {e}")
    
    def _create_annotated_frame(self, frame: np.ndarray, faces_data: List[FaceData]) -> np.ndarray:
        """Create annotated frame with enhanced overlays"""
        try:
            annotated = frame.copy()
            
            for face_data in faces_data:
                x, y, w, h = face_data.position
                
                # Determine colors based on recognition and security
                if face_data.identity['name'] != 'Unknown':
                    # Known person - green shades
                    if face_data.security_clearance.value >= 3:
                        box_color = (0, 255, 0)  # Bright green for trusted
                        text_color = (0, 255, 0)
                    else:
                        box_color = (0, 200, 0)  # Standard green
                        text_color = (0, 200, 0)
                else:
                    # Unknown person - red shades
                    box_color = (0, 0, 255)
                    text_color = (0, 0, 255)
                
                # Draw enhanced face rectangle with thickness based on confidence
                thickness = max(1, int(face_data.confidence * 4))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)
                
                # Prepare display information
                name_text = face_data.identity['name']
                if face_data.confidence > 0:
                    name_text += f" ({face_data.confidence:.2f})"
                
                emotion_text = f"{face_data.emotion.value.title()}"
                demo_text = f"{face_data.age_estimate}y, {face_data.gender_estimate}"
                stability_text = f"Stability: {face_data.stability_score:.2f}"
                
                # Draw information background
                info_lines = [name_text, emotion_text, demo_text]
                if self.config['debug_mode']:
                    info_lines.append(stability_text)
                
                line_height = 25
                bg_height = len(info_lines) * line_height + 10
                
                # Semi-transparent background
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x, y - bg_height), (x + max(200, w), y), box_color, -1)
                cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0, annotated)
                
                # Draw text lines
                for i, text_line in enumerate(info_lines):
                    text_y = y - bg_height + 20 + (i * line_height)
                    cv2.putText(annotated, text_line, (x + 5, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw emotion indicator
                emotion_color = self._get_emotion_color(face_data.emotion)
                cv2.circle(annotated, (x + w - 15, y + 15), 8, emotion_color, -1)
                
                # Security level indicator
                security_level = face_data.security_clearance.value
                for i in range(security_level):
                    cv2.circle(annotated, (x + w - 30 - (i * 15), y + h - 15), 
                              5, (0, 255, 255), -1)
            
            # Draw system information overlay
            self._draw_system_overlay(annotated)
            
            return annotated
            
        except Exception as e:
            logging.error(f"Frame annotation error: {e}")
            return frame
    
    def _get_emotion_color(self, emotion: EmotionState) -> Tuple[int, int, int]:
        """Get color representation for emotion"""
        emotion_colors = {
            EmotionState.HAPPY: (0, 255, 255),      # Yellow
            EmotionState.SAD: (255, 0, 0),          # Blue
            EmotionState.ANGRY: (0, 0, 255),        # Red
            EmotionState.SURPRISED: (255, 0, 255),  # Magenta
            EmotionState.FEAR: (128, 0, 128),       # Purple
            EmotionState.DISGUST: (0, 128, 0),      # Dark Green
            EmotionState.NEUTRAL: (128, 128, 128),  # Gray
            EmotionState.CONFUSED: (0, 165, 255)    # Orange
        }
        return emotion_colors.get(emotion, (128, 128, 128))
    
    def _draw_system_overlay(self, frame: np.ndarray):
        """Draw system information overlay"""
        try:
            height, width = frame.shape[:2]
            
            # System status indicator
            status_color = {
                SystemStatus.ACTIVE: (0, 255, 0),
                SystemStatus.MONITORING: (0, 255, 255),
                SystemStatus.ALERT: (0, 0, 255),
                SystemStatus.MAINTENANCE: (0, 165, 255)
            }.get(self.system_status, (128, 128, 128))
            
            cv2.circle(frame, (30, 30), 12, status_color, -1)
            cv2.putText(frame, f"AURA V2.0 - {self.system_status.name}", 
                       (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Performance metrics
            metrics_text = [
                f"FPS: {self.performance_metrics.fps:.1f}",
                f"Faces: {self.performance_metrics.faces_detected}",
                f"Processed: {self.performance_metrics.frames_processed}",
                f"Avg Time: {self.performance_metrics.avg_processing_time*1000:.1f}ms"
            ]
            
            for i, text in enumerate(metrics_text):
                cv2.putText(frame, text, (width - 200, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_time, (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Recording indicator
            if self.config['recording_enabled']:
                cv2.circle(frame, (width - 30, height - 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (width - 50, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            logging.error(f"System overlay drawing error: {e}")
    
    async def _command_processing_loop(self):
        """Process voice commands asynchronously"""
        try:
            # Initialize voice recognition if available
            if not VOICE_RECOGNITION_AVAILABLE:
                logging.warning("Voice recognition not available")
                return
            
            # Voice recognition setup would go here
            # For now, we'll use keyboard input simulation
            
            while not self.shutdown_event.is_set():
                # In a real implementation, this would process voice commands
                # For demonstration, we'll use a simple command queue
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logging.error(f"Command processing error: {e}")
    
    async def _performance_monitor_loop(self):
        """Monitor system performance continuously"""
        try:
            while not self.shutdown_event.is_set():
                # Update system metrics
                self.performance_metrics.memory_usage = psutil.virtual_memory().percent
                self.performance_metrics.cpu_usage = psutil.cpu_percent()
                
                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.performance_metrics.gpu_usage = gpus[0].load * 100
                except ImportError:
                    self.performance_metrics.gpu_usage = 0.0
                
                self.performance_metrics.last_updated = time.time()
                
                # Log performance warning if needed
                if self.performance_metrics.memory_usage > 90:
                    logging.warning(f"High memory usage: {self.performance_metrics.memory_usage:.1f}%")
                
                if self.performance_metrics.cpu_usage > 90:
                    logging.warning(f"High CPU usage: {self.performance_metrics.cpu_usage:.1f}%")
                
                # Wait before next check
                await asyncio.sleep(5.0)
                
        except Exception as e:
            logging.error(f"Performance monitoring error: {e}")
    
    # Command handlers
    async def _cmd_add_member(self, command: str):
        """Handle add member command"""
        await self.voice_assistant.speak_async(
            "Please position yourself in front of the camera for face registration.",
            EmotionState.NEUTRAL
        )
        # Implementation would capture and register new face
    
    async def _cmd_list_members(self, command: str):
        """Handle list members command"""
        try:
            async with self.face_recognition_system.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM members WHERE is_active = 1")
                count = cursor.fetchone()[0]
                
                await self.voice_assistant.speak_async(
                    f"There are {count} registered members in the database.",
                    EmotionState.NEUTRAL
                )
        except Exception as e:
            await self.voice_assistant.speak_async(
                "Error retrieving member information.",
                EmotionState.NEUTRAL
            )
    
    async def _cmd_system_status(self, command: str):
        """Handle system status command"""
        status_msg = f"""System Status: {self.system_status.name}. 
                        FPS: {self.performance_metrics.fps:.1f}. 
                        CPU: {self.performance_metrics.cpu_usage:.1f}%. 
                        Memory: {self.performance_metrics.memory_usage:.1f}%."""
        
        await self.voice_assistant.speak_async(status_msg, EmotionState.NEUTRAL)
    
    async def _cmd_security_report(self, command: str):
        """Handle security report command"""
        try:
            report = self.security_system.generate_security_report()
            
            report_msg = f"""Security Report: {report['system_status']}. 
                           {report['total_events_24h']} events in last 24 hours. 
                           Security score: {report['security_score']:.0f}%."""
            
            await self.voice_assistant.speak_async(report_msg, EmotionState.NEUTRAL)
        except Exception as e:
            await self.voice_assistant.speak_async(
                "Error generating security report.",
                EmotionState.NEUTRAL
            )
    
    async def _cmd_performance_stats(self, command: str):
        """Handle performance stats command"""
        stats_msg = f"""Performance Statistics: 
                       {self.performance_metrics.frames_processed} frames processed. 
                       {self.performance_metrics.faces_detected} faces detected. 
                       Average processing time: {self.performance_metrics.avg_processing_time*1000:.1f} milliseconds."""
        
        await self.voice_assistant.speak_async(stats_msg, EmotionState.NEUTRAL)
    
    async def _cmd_toggle_recording(self, command: str):
        """Handle toggle recording command"""
        self.config['recording_enabled'] = not self.config['recording_enabled']
        status = "enabled" if self.config['recording_enabled'] else "disabled"
        
        await self.voice_assistant.speak_async(
            f"Recording has been {status}.",
            EmotionState.NEUTRAL
        )
    
    async def _cmd_shutdown(self, command: str):
        """Handle shutdown command"""
        await self.voice_assistant.speak_async(
            "AURA Vision System shutting down. Goodbye!",
            EmotionState.NEUTRAL,
            priority=True
        )
        
        await asyncio.sleep(2)  # Allow speech to complete
        self.shutdown_event.set()
    
    async def _cmd_restart(self, command: str):
        """Handle restart command"""
        await self.voice_assistant.speak_async(
            "System restart initiated.",
            EmotionState.NEUTRAL
        )
        # Implementation would restart the system
    
    async def _cmd_help(self, command: str):
        """Handle help command"""
        help_msg = """Available commands: add member, list members, system status, 
                     security report, performance stats, toggle recording, 
                     shutdown, restart, help. Press Q to quit, R to toggle recording, 
                     D for debug mode, H for help."""
        
        print(help_msg)
        await self.voice_assistant.speak_async(
            "Help information displayed in console.",
            EmotionState.NEUTRAL
        )
    
    async def _show_help(self):
        """Show help information"""
        help_text = """
AURA Vision System V2.0 - Keyboard Controls:
Q - Quit application
R - Toggle recording
D - Toggle debug mode
H - Show this help
Voice commands: add member, list members, system status, security report, help, shutdown
"""
        print(help_text)
    
    async def _cleanup(self):
        """Cleanup system resources"""
        try:
            logging.info("Cleaning up AURA Vision System V2.0...")
            
            self.system_status = SystemStatus.SHUTDOWN
            
            # Release camera
            if self.camera:
                self.camera.release()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Cancel performance monitoring
            if self.performance_monitor_task:
                self.performance_monitor_task.cancel()
                try:
                    await self.performance_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Final message
            await self.voice_assistant.speak_async(
                "AURA Vision System V2.0 shutdown complete.",
                EmotionState.NEUTRAL
            )
            
            logging.info("System cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

# Utility functions and main entry point
def setup_logging():
    """Setup enhanced logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler('aura_vision_v2.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_system_requirements():
    """Check system requirements and capabilities"""
    requirements = {
        'python_version': sys.version_info >= (3, 11),
        'opencv_available': True,
        'face_recognition_available': True,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'tensorflow_available': TF_AVAILABLE,
        'pytorch_available': PYTORCH_AVAILABLE,
        'speech_available': SPEECH_AVAILABLE,
        'voice_recognition_available': VOICE_RECOGNITION_AVAILABLE,
        'camera_available': cv2.VideoCapture(0).isOpened()
    }
    
    # Release test camera
    test_cam = cv2.VideoCapture(0)
    if test_cam.isOpened():
        test_cam.release()
    
    logging.info("System Requirements Check:")
    for requirement, status in requirements.items():
        status_text = "✓" if status else "✗"
        logging.info(f"  {requirement}: {status_text}")
    
    return requirements

async def main():
    """Main function to run AURA Vision System V2.0"""
    try:
        # Setup logging
        setup_logging()
        
        # Check system requirements
        requirements = check_system_requirements()
        
        if not requirements['python_version']:
            logging.error("Python 3.11+ required")
            return
        
        if not requirements['camera_available']:
            logging.error("No camera available")
            return
        
        # Log system information
        system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'timestamp': datetime.now().isoformat()
        }
        logging.info(f"System Info: {system_info}")
        
        # Create and initialize AURA system
        aura_system = AURAVisionSystemV2()
        await aura_system.initialize()
        
        # Run the system
        await aura_system.run()
        
    except KeyboardInterrupt:
        logging.info("System interrupted by user")
    except Exception as e:
        logging.error(f"System error: {e}")
        raise
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Main execution error: {e}")