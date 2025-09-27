import cv2
import numpy as np
import os
import threading
import time
import json
import sqlite3
import logging
import requests
import face_recognition
from typing import Dict, List, Any, Union, Tuple, Optional
import random
from datetime import datetime, timedelta
import pickle
import base64
from io import BytesIO
from PIL import Image
import pyttsx3
import speech_recognition as sr
import queue
import pyaudio
import configparser
import hashlib
from collections import defaultdict
import asyncio
import concurrent.futures
import platform
import psutil
import subprocess
import sys

# Python 3.8.10 compatible imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available, using basic emotion detection")

try:
    import tensorflow as tf
    if hasattr(tf, '__version__') and tf.__version__.startswith('2.'):
        TF_AVAILABLE = True
    else:
        TF_AVAILABLE = False
except ImportError:
    TF_AVAILABLE = False

class SystemInfo:
    @staticmethod
    def get_system_info():
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'timestamp': datetime.now().isoformat()
        }

class ConfigManager:
    def __init__(self, config_file='aura_config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_default_config()
        self._load_config()
    
    def _load_default_config(self):
        self.config['AURA'] = {
            'wake_word': 'aura',
            'voice_rate': '160',
            'voice_volume': '0.9',
            'greeting_language': 'mixed',
            'announcement_cooldown': '30',
            'face_recognition_tolerance': '0.6',
            'camera_index': '0',
            'log_level': 'INFO'
        }
        
        self.config['DATABASE'] = {
            'path': 'aura_members.db',
            'backup_enabled': 'true',
            'backup_interval': '24'
        }
        
        self.config['SECURITY'] = {
            'max_unknown_alerts': '5',
            'alert_cooldown': '300',
            'require_confirmation': 'true'
        }
    
    def _load_config(self):
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.save_config()
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section, key, fallback=0):
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section, key, fallback=0.0):
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getboolean(self, section, key, fallback=False):
        return self.config.getboolean(section, key, fallback=fallback)

class LogManager:
    @staticmethod
    def setup_logger(name: str, log_file: str = "aura_vision.log", level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            
            # File handler with rotation
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Could not setup file logging: {e}")
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

logger = LogManager.setup_logger('AURA.Vision')
config = ConfigManager()

class SecurityManager:
    def __init__(self):
        self.unknown_faces_count = 0
        self.last_alert_time = 0
        self.suspicious_activity_log = []
        self.max_unknown_alerts = config.getint('SECURITY', 'max_unknown_alerts', 5)
        self.alert_cooldown = config.getint('SECURITY', 'alert_cooldown', 300)
    
    def check_suspicious_activity(self, unknown_faces_detected: int) -> bool:
        current_time = time.time()
        
        if unknown_faces_detected > 0:
            self.unknown_faces_count += unknown_faces_detected
            
            if (self.unknown_faces_count >= self.max_unknown_alerts and 
                current_time - self.last_alert_time > self.alert_cooldown):
                
                self.suspicious_activity_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'unknown_faces': self.unknown_faces_count,
                    'alert_triggered': True
                })
                
                self.last_alert_time = current_time
                self.unknown_faces_count = 0
                return True
        
        return False
    
    def generate_security_report(self) -> Dict[str, Any]:
        return {
            'total_alerts': len(self.suspicious_activity_log),
            'recent_activity': self.suspicious_activity_log[-10:],
            'system_status': 'secure' if len(self.suspicious_activity_log) < 5 else 'attention_needed'
        }

class EnhancedMemberDatabase:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.get('DATABASE', 'path', 'aura_members.db')
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()
        self._create_indexes()
        
        # Backup management
        if config.getboolean('DATABASE', 'backup_enabled', True):
            self._schedule_backup()
        
    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                relationship TEXT,
                face_encoding BLOB,
                image_path TEXT,
                access_level TEXT DEFAULT 'guest',
                phone TEXT,
                email TEXT,
                notes TEXT,
                greeting_preference TEXT DEFAULT 'formal',
                favorite_time TEXT,
                security_clearance INTEGER DEFAULT 1,
                last_location TEXT,
                visit_frequency INTEGER DEFAULT 0,
                emergency_contact TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS visit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration INTEGER,
                emotion_detected TEXT,
                location TEXT,
                confidence_score REAL,
                FOREIGN KEY (member_id) REFERENCES members (id)
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                event_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                severity TEXT DEFAULT 'info'
            )
        ''')
        
        self.conn.commit()
    
    def _create_indexes(self):
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_members_name ON members(name)",
            "CREATE INDEX IF NOT EXISTS idx_members_last_seen ON members(last_seen)",
            "CREATE INDEX IF NOT EXISTS idx_visit_logs_member_id ON visit_logs(member_id)",
            "CREATE INDEX IF NOT EXISTS idx_visit_logs_timestamp ON visit_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)"
        ]
        
        for index in indexes:
            self.cursor.execute(index)
        self.conn.commit()
    
    def _schedule_backup(self):
        def backup_worker():
            while True:
                time.sleep(config.getint('DATABASE', 'backup_interval', 24) * 3600)
                self.create_backup()
        
        backup_thread = threading.Thread(target=backup_worker, daemon=True)
        backup_thread.start()
    
    def create_backup(self):
        backup_name = f"aura_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        try:
            import shutil
            shutil.copy2(self.db_path, backup_name)
            logger.info(f"Database backup created: {backup_name}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
    
    def add_member(self, name: str, face_encoding: np.ndarray, **kwargs) -> int:
        with self._lock:
            encoding_blob = pickle.dumps(face_encoding)
            self.cursor.execute('''
                INSERT INTO members (name, face_encoding, age, gender, relationship, 
                                   access_level, phone, email, notes, image_path, 
                                   greeting_preference, security_clearance, emergency_contact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, encoding_blob, kwargs.get('age'), kwargs.get('gender'), 
                  kwargs.get('relationship', 'Unknown'), kwargs.get('access_level', 'guest'),
                  kwargs.get('phone'), kwargs.get('email'), kwargs.get('notes'),
                  kwargs.get('image_path'), kwargs.get('greeting_preference', 'formal'),
                  kwargs.get('security_clearance', 1), kwargs.get('emergency_contact')))
            
            member_id = self.cursor.lastrowid
            self.conn.commit()
            
            # Log the event
            self.log_system_event('member_added', f"Added member: {name} (ID: {member_id})")
            return member_id
    
    def get_all_members(self) -> List[Dict]:
        with self._lock:
            self.cursor.execute('SELECT * FROM members ORDER BY name')
            rows = self.cursor.fetchall()
            
            columns = [desc[0] for desc in self.cursor.description]
            members = []
            
            for row in rows:
                member = dict(zip(columns, row))
                if member['face_encoding']:
                    try:
                        member['face_encoding'] = pickle.loads(member['face_encoding'])
                    except Exception as e:
                        logger.error(f"Failed to load face encoding for member {member['name']}: {e}")
                        member['face_encoding'] = None
                members.append(member)
                
            return members
    
    def update_last_seen(self, member_id: int, emotion: str = None, location: str = None):
        with self._lock:
            current_time = datetime.now().isoformat()
            self.cursor.execute('''
                UPDATE members 
                SET last_seen = ?, visit_frequency = visit_frequency + 1, updated_at = ?
                WHERE id = ?
            ''', (current_time, current_time, member_id))
            
            # Log the visit
            self.cursor.execute('''
                INSERT INTO visit_logs (member_id, emotion_detected, location)
                VALUES (?, ?, ?)
            ''', (member_id, emotion, location))
            
            self.conn.commit()
    
    def log_system_event(self, event_type: str, event_data: str, severity: str = 'info'):
        with self._lock:
            self.cursor.execute('''
                INSERT INTO system_events (event_type, event_data, severity)
                VALUES (?, ?, ?)
            ''', (event_type, event_data, severity))
            self.conn.commit()
    
    def get_member_statistics(self) -> Dict[str, Any]:
        with self._lock:
            stats = {}
            
            # Total members
            self.cursor.execute('SELECT COUNT(*) FROM members')
            stats['total_members'] = self.cursor.fetchone()[0]
            
            # Members by relationship
            self.cursor.execute('''
                SELECT relationship, COUNT(*) 
                FROM members 
                GROUP BY relationship
            ''')
            stats['by_relationship'] = dict(self.cursor.fetchall())
            
            # Recent visits (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            self.cursor.execute('''
                SELECT COUNT(*) FROM visit_logs 
                WHERE timestamp > ?
            ''', (week_ago,))
            stats['recent_visits'] = self.cursor.fetchone()[0]
            
            # Most frequent visitors
            self.cursor.execute('''
                SELECT m.name, m.visit_frequency 
                FROM members m 
                ORDER BY m.visit_frequency DESC 
                LIMIT 5
            ''')
            stats['frequent_visitors'] = self.cursor.fetchall()
            
            return stats
    
    def delete_member(self, member_id: int):
        with self._lock:
            # Get member name for logging
            self.cursor.execute('SELECT name FROM members WHERE id = ?', (member_id,))
            result = self.cursor.fetchone()
            name = result[0] if result else f"ID:{member_id}"
            
            # Delete member and related records
            self.cursor.execute('DELETE FROM visit_logs WHERE member_id = ?', (member_id,))
            self.cursor.execute('DELETE FROM members WHERE id = ?', (member_id,))
            self.conn.commit()
            
            self.log_system_event('member_deleted', f"Deleted member: {name}")

class AdvancedVoiceAssistant:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self._setup_voice()
        except Exception as e:
            logger.error(f"Failed to initialize voice engine: {e}")
            self.engine = None
            
        self.speaking = False
        self._lock = threading.Lock()
        
        # Voice personality settings
        self.personality_responses = {
            'greeting': [
                "नमस्ते! मैं AURA हूँ, आपकी सेवा में।",
                "Hello! I'm AURA, your intelligent assistant.",
                "स्वागत है! AURA यहाँ है आपकी मदद के लिए।"
            ],
            'acknowledgment': [
                "जी हाँ, समझ गया।", "Got it!", "ठीक है।", "Understood."
            ],
            'confusion': [
                "माफ करिए, मुझे समझ नहीं आया।", "Sorry, I didn't understand that.",
                "कृपया फिर से कहें।", "Could you please repeat?"
            ]
        }
    
    def _setup_voice(self):
        if not self.engine:
            return
            
        try:
            # Voice configuration
            rate = config.getint('AURA', 'voice_rate', 160)
            volume = config.getfloat('AURA', 'voice_volume', 0.9)
            
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Try to set a female voice
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['female', 'zira', 'hazel']):
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
                    
        except Exception as e:
            logger.error(f"Voice setup error: {e}")
    
    def speak(self, text: str, interrupt: bool = False, emotion: str = 'neutral'):
        if not self.engine:
            print(f"AURA (text only): {text}")
            return
            
        if interrupt and self.speaking:
            self.engine.stop()
        
        with self._lock:
            try:
                self.speaking = True
                
                # Add emotional context to speech
                if emotion == 'happy':
                    text = f"😊 {text}"
                elif emotion == 'concerned':
                    text = f"🤔 {text}"
                elif emotion == 'alert':
                    text = f"⚠️ {text}"
                
                print(f"AURA: {text}")
                logger.info(f"AURA speaking: {text}")
                
                self.engine.say(text)
                self.engine.runAndWait()
                
            except Exception as e:
                logger.error(f"Voice synthesis error: {e}")
                print(f"AURA (fallback): {text}")
            finally:
                self.speaking = False
    
    def speak_async(self, text: str, emotion: str = 'neutral'):
        thread = threading.Thread(target=self.speak, args=(text, False, emotion), daemon=True)
        thread.start()
        return thread
    
    def get_personality_response(self, category: str) -> str:
        responses = self.personality_responses.get(category, [""])
        return random.choice(responses)

class EnhancedVoiceRecognition:
    def __init__(self, wake_word: str = None):
        self.wake_word = (wake_word or config.get('AURA', 'wake_word', 'aura')).lower()
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.mic_available = True
        except Exception as e:
            logger.error(f"Microphone initialization failed: {e}")
            self.mic_available = False
            
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.listening_thread = None
        self.confidence_threshold = 0.7
        self.last_command_time = 0
        self.command_history = []
        
        if self.mic_available:
            # Enhanced recognition settings
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            
            # Adjust for ambient noise
            try:
                with self.microphone as source:
                    logger.info("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                    logger.info("Voice recognition calibrated")
            except Exception as e:
                logger.error(f"Microphone calibration failed: {e}")
    
    def start_listening(self):
        if not self.mic_available:
            logger.warning("Microphone not available, voice recognition disabled")
            return
            
        if not self.is_listening:
            self.is_listening = True
            self.listening_thread = threading.Thread(target=self._listen_continuously, daemon=True)
            self.listening_thread.start()
            logger.info("Enhanced voice recognition started")
    
    def stop_listening(self):
        if self.is_listening:
            self.is_listening = False
            if self.listening_thread:
                self.listening_thread.join(timeout=3)
            logger.info("Voice recognition stopped")
    
    def _listen_continuously(self):
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=8)
                
                try:
                    # Use Google Speech Recognition with multiple attempts
                    for attempt in range(2):
                        try:
                            text = self.recognizer.recognize_google(audio, 
                                                                 language='en-IN',
                                                                 show_all=False).lower()
                            
                            if text:
                                self._process_audio_input(text)
                                break
                                
                        except sr.UnknownValueError:
                            if attempt == 0:
                                # Try with different language settings
                                try:
                                    text = self.recognizer.recognize_google(audio, 
                                                                         language='hi-IN').lower()
                                    if text:
                                        self._process_audio_input(text)
                                        break
                                except:
                                    pass
                            
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {e}")
                    time.sleep(5)  # Wait before retrying
                    
            except sr.WaitTimeoutError:
                pass  # Continue listening
            except Exception as e:
                logger.error(f"Listening error: {e}")
                time.sleep(1)
    
    def _process_audio_input(self, text: str):
        current_time = time.time()
        
        # Prevent duplicate commands
        if (current_time - self.last_command_time < 2 and 
            self.command_history and text == self.command_history[-1]):
            return
        
        logger.info(f"Audio input processed: {text}")
        
        # Check for wake word with fuzzy matching
        if self._contains_wake_word(text):
            command = self._extract_command(text)
            if command:
                self.command_queue.put(command)
                self.command_history.append(text)
                self.last_command_time = current_time
                
                # Keep history manageable
                if len(self.command_history) > 10:
                    self.command_history.pop(0)
                
                logger.info(f"Command queued: {command}")
    
    def _contains_wake_word(self, text: str) -> bool:
        # Direct match
        if self.wake_word in text:
            return True
        
        # Fuzzy matching for variations
        variations = ['aura', 'aurora', 'aora', 'ora']
        return any(var in text for var in variations)
    
    def _extract_command(self, text: str) -> str:
        # Remove wake word and clean up
        for wake_var in ['aura', 'aurora', 'aora', 'ora']:
            text = text.replace(wake_var, '').strip()
        
        # Remove common filler words
        fillers = ['please', 'कृपया', 'जी', 'अब', 'now']
        words = text.split()
        cleaned_words = [w for w in words if w not in fillers]
        
        return ' '.join(cleaned_words).strip()
    
    def get_command(self) -> Optional[str]:
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

class AdvancedFaceProcessor:
    def __init__(self):
        self.member_db = EnhancedMemberDatabase()
        self.global_api = GlobalFaceAPI()
        self.voice = AdvancedVoiceAssistant()
        self.security = SecurityManager()
        self.known_members = self.member_db.get_all_members()
        self.last_announcements = {}
        self.face_tracking = defaultdict(list)
        self.emotion_history = defaultdict(list)
        
        # Enhanced greeting system
        self.greeting_templates = {
            'family': {
                'hindi': [
                    "आपका हार्दिक स्वागत है {name}! घर आकर अच्छा लगा होगा।",
                    "नमस्ते {name}! कैसे हैं आप? बहुत दिन बाद मिले।",
                    "अरे {name}! खुशी हुई आपको देखकर। कैसा चल रहा है सब?"
                ],
                'english': [
                    "Welcome home {name}! So good to see you.",
                    "Hello {name}! How have you been?",
                    "Hi {name}! Great to have you back."
                ]
            },
            'friend': {
                'hindi': [
                    "अरे {name}! कैसे हो यार? बहुत दिन बाद आए।",
                    "हैलो {name}! क्या हाल है? कैसी चल रही है जिंदगी?",
                    "यार {name}! मिलकर अच्छा लगा। कैसे हो?"
                ],
                'english': [
                    "Hey {name}! How's it going buddy?",
                    "Hi {name}! Long time no see!",
                    "Hello {name}! Great to see you again!"
                ]
            },
            'colleague': {
                'hindi': [
                    "नमस्कार {name} जी! कैसे हैं आप? ऑफिस कैसा चल रहा है?",
                    "गुड मॉर्निंग {name}! आज कैसा दिन है?",
                    "हैलो {name}! मिलकर खुशी हुई।"
                ],
                'english': [
                    "Good day {name}! How are things at work?",
                    "Hello {name}! Nice to see you.",
                    "Hi {name}! Hope you're doing well."
                ]
            },
            'guest': {
                'hindi': [
                    "नमस्ते! आपका स्वागत है। कृपया अंदर आइए।",
                    "आदाब! घर में आपका स्वागत है।"
                ],
                'english': [
                    "Welcome! Please come in.",
                    "Hello! Nice to meet you."
                ]
            },
            'unknown': {
                'hindi': [
                    "नमस्ते! मैं आपको पहचान नहीं पा रहा हूँ। कृपया अपना परिचय दें।",
                    "स्वागत है! आप कौन हैं? मैं AURA हूँ।"
                ],
                'english': [
                    "Hello! I don't recognize you. Could you please introduce yourself?",
                    "Welcome! I'm AURA. May I know who you are?"
                ]
            }
        }
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def detect_and_analyze_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use face_recognition for detection
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        results = []
        unknown_count = 0
        
        for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
            face_data = self._analyze_single_face(
                frame, rgb_frame, (left, top, right-left, bottom-top), face_encoding, i
            )
            results.append(face_data)
            
            if face_data['identity']['source'] == 'not_found':
                unknown_count += 1
        
        # Check for suspicious activity
        if self.security.check_suspicious_activity(unknown_count):
            self.voice.speak_async("सुरक्षा चेतावनी: अज्ञात व्यक्ति का पता चला है।", emotion='alert')
        
        return results
    
    def _analyze_single_face(self, frame: np.ndarray, rgb_frame: np.ndarray, 
                           bbox: Tuple[int, int, int, int], face_encoding: np.ndarray, 
                           face_id: int) -> Dict[str, Any]:
        x, y, w, h = bbox
        face_img = frame[y:y+h, x:x+w]
        
        # Core analysis
        identity = self._recognize_person(face_encoding, face_img)
        age, gender = self._estimate_age_gender(face_img)
        emotion = self._detect_advanced_emotion(rgb_frame, bbox)
        
        # Track face over time
        face_key = f"{identity['name']}_{face_id}"
        self.face_tracking[face_key].append({
            'timestamp': time.time(),
            'position': bbox,
            'emotion': emotion,
            'confidence': identity.get('confidence', 0)
        })
        
        # Keep tracking history manageable
        if len(self.face_tracking[face_key]) > 30:
            self.face_tracking[face_key].pop(0)
        
        # Advanced greeting logic
        if identity['name'] != 'Unknown' and identity['source'] != 'error':
            self._announce_person_advanced(identity, emotion, bbox)
        
        return {
            "identity": identity,
            "age": age,
            "gender": gender,
            "emotion": emotion,
            "position": bbox,
            "timestamp": time.time(),
            "tracking_id": face_key,
            "stability_score": self._calculate_stability_score(face_key)
        }
    
    def _recognize_person(self, face_encoding: np.ndarray, face_img: np.ndarray) -> Dict[str, Any]:
        if not self.known_members:
            return self.global_api.search_face_online(face_img)
        
        known_encodings = [member['face_encoding'] for member in self.known_members if member['face_encoding'] is not None]
        tolerance = config.getfloat('AURA', 'face_recognition_tolerance', 0.6)
        
        if not known_encodings:
            return self.global_api.search_face_online(face_img)
        
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        if matches and min(distances) < tolerance:
            best_match_idx = np.argmin(distances)
            if matches[best_match_idx]:
                # Find the corresponding member (accounting for None encodings)
                valid_member_idx = 0
                for i, member in enumerate(self.known_members):
                    if member['face_encoding'] is not None:
                        if valid_member_idx == best_match_idx:
                            # Update database
                            self.member_db.update_last_seen(
                                member['id'], 
                                emotion=None,  # Will be updated later
                                location="main_camera"
                            )
                            
                            return {
                                'name': member['name'],
                                'relationship': member['relationship'],
                                'access_level': member['access_level'],
                                'source': 'local_db',
                                'confidence': float(1 - distances[best_match_idx]),
                                'member_id': member['id'],
                                'greeting_preference': member.get('greeting_preference', 'formal'),
                                'security_clearance': member.get('security_clearance', 1)
                            }
                        valid_member_idx += 1
        
        # Fallback to global search
        return self.global_api.search_face_online(face_img)

    def _estimate_age_gender(self, face_img: np.ndarray) -> Tuple[int, str]:
        """Estimate age and gender from face image"""
        try:
            # Simple heuristic-based estimation
            # In production, you'd use a trained model
            height, width = face_img.shape[:2]
            
            # Analyze face structure (simplified)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Age estimation based on skin texture and wrinkles
            # This is a simplified approach
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            if edge_density > 0.1:
                estimated_age = random.randint(35, 65)
            elif edge_density > 0.05:
                estimated_age = random.randint(25, 45)
            else:
                estimated_age = random.randint(18, 35)
            
            # Gender estimation (simplified)
            # In reality, this would use facial feature analysis
            estimated_gender = random.choice(['Male', 'Female'])
            
            return estimated_age, estimated_gender
            
        except Exception as e:
            logger.error(f"Age/gender estimation error: {e}")
            return 25, 'Unknown'

    def _detect_advanced_emotion(self, rgb_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Advanced emotion detection using multiple methods"""
        try:
            x, y, w, h = bbox
            
            if MEDIAPIPE_AVAILABLE:
                # Use MediaPipe for facial landmark detection
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Analyze key facial points for emotion
                        emotion = self._analyze_facial_landmarks(face_landmarks)
                        return emotion
            
            # Fallback to basic emotion detection
            face_region = rgb_frame[y:y+h, x:x+w]
            return self._basic_emotion_detection(face_region)
            
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return 'neutral'
    
    def _analyze_facial_landmarks(self, landmarks) -> str:
        """Analyze facial landmarks to determine emotion"""
        try:
            # Extract key points for emotion analysis
            # Mouth corners, eyebrows, eyes
            emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
            
            # Simplified emotion detection based on landmark positions
            # In practice, this would involve complex geometric analysis
            return random.choice(emotions)
            
        except Exception as e:
            logger.error(f"Landmark analysis error: {e}")
            return 'neutral'
    
    def _basic_emotion_detection(self, face_region: np.ndarray) -> str:
        """Basic emotion detection using image analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            
            # Simple brightness and contrast analysis
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Heuristic emotion mapping
            if brightness > 120 and contrast > 40:
                return 'happy'
            elif brightness < 80:
                return 'sad'
            elif contrast > 60:
                return 'surprised'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Basic emotion detection error: {e}")
            return 'neutral'
    
    def _calculate_stability_score(self, face_key: str) -> float:
        """Calculate how stable/consistent the face tracking is"""
        if face_key not in self.face_tracking:
            return 0.0
        
        tracking_data = self.face_tracking[face_key]
        if len(tracking_data) < 3:
            return 0.5
        
        # Analyze position consistency
        positions = [data['position'] for data in tracking_data[-5:]]
        position_variance = np.var([pos[0] + pos[1] for pos in positions])
        
        # Analyze confidence consistency
        confidences = [data['confidence'] for data in tracking_data[-5:]]
        confidence_avg = np.mean(confidences)
        
        # Combined stability score
        stability = min(1.0, confidence_avg * (1.0 / (1.0 + position_variance / 1000)))
        return stability
    
    def _announce_person_advanced(self, identity: Dict[str, Any], emotion: str, bbox: Tuple[int, int, int, int]):
        """Advanced person announcement with context awareness"""
        name = identity['name']
        current_time = time.time()
        
        # Check announcement cooldown
        cooldown = config.getint('AURA', 'announcement_cooldown', 30)
        if name in self.last_announcements:
            if current_time - self.last_announcements[name] < cooldown:
                return
        
        self.last_announcements[name] = current_time
        
        # Determine greeting context
        relationship = identity.get('relationship', 'guest')
        greeting_pref = identity.get('greeting_preference', 'formal')
        language = config.get('AURA', 'greeting_language', 'mixed')
        
        # Select appropriate greeting
        greeting_category = self._map_relationship_to_category(relationship)
        greeting_lang = 'hindi' if language == 'hindi' else 'english'
        
        if language == 'mixed':
            greeting_lang = random.choice(['hindi', 'english'])
        
        greetings = self.greeting_templates.get(greeting_category, {}).get(greeting_lang, [])
        if greetings:
            greeting = random.choice(greetings).format(name=name)
        else:
            greeting = f"Hello {name}!"
        
        # Add emotional context
        if emotion in ['happy', 'surprised']:
            greeting += " आप खुश लग रहे हैं!" if greeting_lang == 'hindi' else " You look happy!"
        elif emotion in ['sad', 'angry']:
            greeting += " आप परेशान लग रहे हैं। सब ठीक तो है?" if greeting_lang == 'hindi' else " You seem troubled. Is everything okay?"
        
        # Add security level context
        security_level = identity.get('security_clearance', 1)
        if security_level > 3:
            greeting += " आपकी विशेष पहुंच सुनिश्चित की गई है।" if greeting_lang == 'hindi' else " Your special access is confirmed."
        
        self.voice.speak_async(greeting, emotion='happy')
        
        # Log the interaction
        self.member_db.log_system_event(
            'person_greeted',
            f"Greeted {name} with emotion {emotion} at position {bbox}",
            'info'
        )
    
    def _map_relationship_to_category(self, relationship: str) -> str:
        """Map relationship to greeting category"""
        relationship_lower = relationship.lower()
        
        if any(rel in relationship_lower for rel in ['family', 'parent', 'sibling', 'spouse', 'child']):
            return 'family'
        elif any(rel in relationship_lower for rel in ['friend', 'buddy', 'mate']):
            return 'friend'
        elif any(rel in relationship_lower for rel in ['colleague', 'coworker', 'boss', 'employee']):
            return 'colleague'
        elif relationship_lower in ['guest', 'visitor']:
            return 'guest'
        else:
            return 'unknown'

class GlobalFaceAPI:
    """Interface for online face recognition services"""
    
    def __init__(self):
        self.api_endpoints = {
            'face_plus_plus': 'https://api-us.faceplusplus.com/facepp/v3/search',
            'azure_face': 'https://api.cognitive.microsoft.com/face/v1.0',
            'aws_rekognition': 'https://rekognition.us-east-1.amazonaws.com'
        }
        self.api_keys = self._load_api_keys()
        self.rate_limits = defaultdict(list)
        self.cache = {}
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from config or environment"""
        return {
            'face_plus_plus': os.getenv('FACE_PLUS_PLUS_KEY', ''),
            'azure_face': os.getenv('AZURE_FACE_KEY', ''),
            'aws_access_key': os.getenv('AWS_ACCESS_KEY', ''),
            'aws_secret_key': os.getenv('AWS_SECRET_KEY', '')
        }
    
    def search_face_online(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Search for face using online APIs"""
        try:
            # Convert face image to base64
            face_b64 = self._image_to_base64(face_img)
            
            if not face_b64:
                return self._get_unknown_result('conversion_error')
            
            # Check cache first
            img_hash = hashlib.md5(face_b64.encode()).hexdigest()
            if img_hash in self.cache:
                cached_result = self.cache[img_hash]
                if time.time() - cached_result['timestamp'] < 3600:  # 1 hour cache
                    return cached_result['data']
            
            # Try different APIs with fallback
            for api_name in ['face_plus_plus', 'azure_face']:
                if self._check_rate_limit(api_name):
                    result = self._query_api(api_name, face_b64)
                    if result['name'] != 'Unknown':
                        self.cache[img_hash] = {
                            'data': result,
                            'timestamp': time.time()
                        }
                        return result
            
            return self._get_unknown_result('not_found')
            
        except Exception as e:
            logger.error(f"Online face search error: {e}")
            return self._get_unknown_result('error')
    
    def _get_unknown_result(self, source: str) -> Dict[str, Any]:
        """Return standard unknown result"""
        return {
            'name': 'Unknown',
            'relationship': 'Unknown',
            'source': source,
            'confidence': 0.0
        }
    
    def _image_to_base64(self, img: np.ndarray) -> str:
        """Convert image to base64 string"""
        try:
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            
            # Resize for API efficiency
            pil_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Image to base64 conversion error: {e}")
            return ""
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows request"""
        current_time = time.time()
        
        # Clean old entries
        self.rate_limits[api_name] = [
            t for t in self.rate_limits[api_name] 
            if current_time - t < 3600  # 1 hour window
        ]
        
        # Check limits (example: 100 requests per hour)
        if len(self.rate_limits[api_name]) >= 100:
            return False
        
        self.rate_limits[api_name].append(current_time)
        return True
    
    def _query_api(self, api_name: str, face_b64: str) -> Dict[str, Any]:
        """Query specific API for face recognition"""
        try:
            if api_name == 'face_plus_plus':
                return self._query_face_plus_plus(face_b64)
            elif api_name == 'azure_face':
                return self._query_azure_face(face_b64)
            else:
                return self._get_unknown_result('api_error')
                
        except Exception as e:
            logger.error(f"API query error for {api_name}: {e}")
            return self._get_unknown_result('api_error')
    
    def _query_face_plus_plus(self, face_b64: str) -> Dict[str, Any]:
        """Query Face++ API (placeholder implementation)"""
        # This is a placeholder - actual implementation would require API credentials
        # and proper API calls
        
        # Simulate API response
        if random.random() > 0.8:  # 20% chance of "recognition"
            fake_names = ['John Smith', 'Jane Doe', 'Alex Johnson', 'Maria Garcia']
            return {
                'name': random.choice(fake_names),
                'relationship': 'Unknown',
                'source': 'face_plus_plus',
                'confidence': random.uniform(0.6, 0.9)
            }
        
        return self._get_unknown_result('face_plus_plus')
    
    def _query_azure_face(self, face_b64: str) -> Dict[str, Any]:
        """Query Azure Face API (placeholder implementation)"""
        # Placeholder implementation
        return self._get_unknown_result('azure_face')

class AURAVisionSystem:
    """Main AURA Vision System class"""
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self.face_processor = AdvancedFaceProcessor()
        self.voice_recognition = EnhancedVoiceRecognition()
        self.voice_assistant = AdvancedVoiceAssistant()
        self.member_db = EnhancedMemberDatabase()
        
        # Camera setup
        self.camera_index = config.getint('AURA', 'camera_index', 0)
        self.cap = None
        self.running = False
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.performance_stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'recognitions_made': 0,
            'avg_processing_time': 0.0
        }
        
        # Command processing
        self.command_handlers = {
            'add member': self._handle_add_member,
            'list members': self._handle_list_members,
            'delete member': self._handle_delete_member,
            'system status': self._handle_system_status,
            'security report': self._handle_security_report,
            'statistics': self._handle_statistics,
            'help': self._handle_help,
            'shutdown': self._handle_shutdown,
            'restart': self._handle_restart
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the AURA system"""
        try:
            logger.info("Initializing AURA Vision System...")
            
            # Setup camera
            self._setup_camera()
            
            # Start voice recognition
            self.voice_recognition.start_listening()
            
            # System greeting
            greeting = "AURA Vision System activated. मैं आपकी सेवा में हूँ।"
            self.voice_assistant.speak_async(greeting, emotion='happy')
            
            logger.info("AURA Vision System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def _setup_camera(self):
        """Setup camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Optimal camera settings
            self.cap.set(cv2.CAP_PROP_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            width = int(self.cap.get(cv2.CAP_PROP_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera initialized: {width}x{height} @ {fps}fps")
            
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            raise
    
    def run(self):
        """Main system loop"""
        self.running = True
        logger.info("AURA Vision System started")
        
        try:
            while self.running:
                # Process camera frame
                self._process_frame()
                
                # Handle voice commands
                self._process_voice_commands()
                
                # Performance monitoring
                self._update_performance_stats()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self._cleanup()
    
    def _process_frame(self):
        """Process single camera frame"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                return
            
            start_time = time.time()
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Face detection and analysis
            faces_data = self.face_processor.detect_and_analyze_faces(frame)
            
            # Draw overlay information
            annotated_frame = self._draw_overlay(frame, faces_data)
            
            # Display frame
            cv2.imshow('AURA Vision System', annotated_frame)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats['frames_processed'] += 1
            self.performance_stats['faces_detected'] += len(faces_data)
            self.performance_stats['avg_processing_time'] = (
                (self.performance_stats['avg_processing_time'] * (self.performance_stats['frames_processed'] - 1) + processing_time) /
                self.performance_stats['frames_processed']
            )
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def _draw_overlay(self, frame: np.ndarray, faces_data: List[Dict[str, Any]]) -> np.ndarray:
        """Draw overlay information on frame"""
        try:
            overlay_frame = frame.copy()
            
            for face_data in faces_data:
                x, y, w, h = face_data['position']
                identity = face_data['identity']
                emotion = face_data['emotion']
                stability = face_data['stability_score']
                
                # Determine box color based on recognition status
                if identity['name'] != 'Unknown':
                    color = (0, 255, 0)  # Green for known
                    text_color = (0, 255, 0)
                else:
                    color = (0, 0, 255)  # Red for unknown
                    text_color = (0, 0, 255)
                
                # Draw face rectangle
                cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, 2)
                
                # Prepare display text
                name_text = identity['name']
                confidence = identity.get('confidence', 0)
                info_text = f"{emotion} ({stability:.2f})"
                
                if confidence > 0:
                    name_text += f" ({confidence:.2f})"
                
                # Draw text background
                text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(overlay_frame, (x, y - 30), (x + text_size[0] + 10, y), color, -1)
                
                # Draw text
                cv2.putText(overlay_frame, name_text, (x + 5, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay_frame, info_text, (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Draw system info
            self._draw_system_info(overlay_frame)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"Overlay drawing error: {e}")
            return frame
    
    def _draw_system_info(self, frame: np.ndarray):
        """Draw system information on frame"""
        try:
            height, width = frame.shape[:2]
            
            # FPS calculation
            current_time = time.time()
            self.fps_counter += 1
            
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
            
            # System info text
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(frame, fps_text, (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Status text
            status_text = "AURA ACTIVE"
            cv2.putText(frame, status_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Member count
            member_count = len(self.face_processor.known_members)
            member_text = f"Members: {member_count}"
            cv2.putText(frame, member_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"System info drawing error: {e}")
    
    def _process_voice_commands(self):
        """Process voice commands"""
        try:
            command = self.voice_recognition.get_command()
            if command:
                logger.info(f"Processing command: {command}")
                self._execute_command(command)
                
        except Exception as e:
            logger.error(f"Voice command processing error: {e}")
    
    def _execute_command(self, command: str):
        """Execute voice command"""
        try:
            command_lower = command.lower().strip()
            
            # Find matching command handler
            for cmd_key, handler in self.command_handlers.items():
                if cmd_key in command_lower:
                    handler(command)
                    return
            
            # If no specific handler found, provide general response
            self.voice_assistant.speak_async(
                "मुझे समझ नहीं आया। help कहें सभी commands जानने के लिए।"
            )
            
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            self.voice_assistant.speak_async("माफ करिए, command process नहीं हो सका।")
    
    # Command handlers
    def _handle_add_member(self, command: str):
        """Handle add member command"""
        self.voice_assistant.speak_async("नया member add करने के लिए camera के सामने खड़े हों।")
        # Implementation would capture face and add to database
        
    def _handle_list_members(self, command: str):
        """Handle list members command"""
        members = self.member_db.get_all_members()
        count = len(members)
        self.voice_assistant.speak_async(f"कुल {count} members registered हैं।")
        
    def _handle_delete_member(self, command: str):
        """Handle delete member command"""
        self.voice_assistant.speak_async("Member delete करने के लिए name बताएं।")
        
    def _handle_system_status(self, command: str):
        """Handle system status command"""
        stats = self.performance_stats
        status_msg = f"System running normally. {stats['frames_processed']} frames processed, {stats['faces_detected']} faces detected."
        self.voice_assistant.speak_async("System ठीक से चल रहा है।")
        
    def _handle_security_report(self, command: str):
        """Handle security report command"""
        report = self.face_processor.security.generate_security_report()
        status = report['system_status']
        alerts = report['total_alerts']
        
        if status == 'secure':
            self.voice_assistant.speak_async(f"Security status: Normal. {alerts} alerts till now.")
        else:
            self.voice_assistant.speak_async(f"Security attention needed. {alerts} alerts detected.")
    
    def _handle_statistics(self, command: str):
        """Handle statistics command"""
        stats = self.member_db.get_member_statistics()
        total = stats['total_members']
        recent = stats['recent_visits']
        
        self.voice_assistant.speak_async(
            f"Total {total} members registered. Recent visits: {recent}."
        )
    
    def _handle_help(self, command: str):
        """Handle help command"""
        help_text = """Available commands: add member, list members, delete member, 
                      system status, security report, statistics, help, shutdown."""
        print(help_text)
        self.voice_assistant.speak_async("Commands list console में display हो गई है।")
    
    def _handle_shutdown(self, command: str):
        """Handle shutdown command"""
        self.voice_assistant.speak_async("AURA system shutdown हो रहा है। धन्यवाद!")
        time.sleep(2)
        self.running = False
    
    def _handle_restart(self, command: str):
        """Handle restart command"""
        self.voice_assistant.speak_async("System restart हो रहा है।")
        # Implementation would restart the system
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        # This could include memory usage, CPU usage, etc.
        pass
    
    def _cleanup(self):
        """Cleanup system resources"""
        try:
            logger.info("Cleaning up system resources...")
            
            if self.cap:
                self.cap.release()
            
            self.voice_recognition.stop_listening()
            cv2.destroyAllWindows()
            
            # Final system message
            self.voice_assistant.speak("AURA Vision System बंद हो गया। नमस्ते!")
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main function to run AURA Vision System"""
    try:
        # Check system requirements
        system_info = SystemInfo.get_system_info()
        logger.info(f"System Info: {system_info}")
        
        # Create and run AURA system
        aura_system = AURAVisionSystem()
        aura_system.run()
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"Error: {e}")
    
    finally:
        # Ensure cleanup
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()