"""
AURA: Advanced Universal Responsive Assistant
Neural Voice Manager: AI-powered voice interface with continuous intelligence
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Dict, List, Optional, Union, Tuple

import langdetect
import openai
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from langdetect import DetectorFactory
from playsound import playsound

# Seed the language detector for consistent results
DetectorFactory.seed = 0

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aura_voice.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AURA.Voice')


class MessageMonitor:
    """
    Monitor and handle device messages across various communication platforms.
    Provides unified interface for message handling regardless of source application.
    """
    
    def __init__(self):
        """Initialize the message monitoring system."""
        self.message_queue = []
        self.app_connections = {}  # Would store API connections to messaging apps
        logger.info("Message Monitor initialized")
        
    def check_messages(self) -> Optional[Dict]:
        """
        Check for new messages across all connected platforms.
        
        Returns:
            Optional[Dict]: Message data if available, None otherwise
        """
        # This would integrate with actual messaging APIs in production
        if self.message_queue:
            return self.message_queue.pop(0)
        return None
        
    def send_response(self, app: str, message: str) -> bool:
        """
        Send response through the specified messaging app.
        
        Args:
            app: Name of the messaging application
            message: Content to send
            
        Returns:
            bool: Success status of the sending operation
        """
        logger.info(f"Responding via {app}: {message}")
        # Actual implementation would use app-specific APIs
        return True
    
    def register_app(self, app_name: str, credentials: Dict) -> bool:
        """
        Register and authenticate with a messaging application.
        
        Args:
            app_name: Name of the messaging application
            credentials: Authentication details for the app
            
        Returns:
            bool: Success status of the registration
        """
        # In production, this would connect to the app's API
        logger.info(f"Registered app: {app_name}")
        return True


class LanguageManager:
    """
    Handles multilingual support for voice interactions.
    Provides language detection, switching, and appropriate speech processing.
    """
    
    # Supported languages and their configurations
    SUPPORTED_LANGUAGES = {
        'en': {'name': 'English', 'tts': 'pyttsx3', 'sr': 'en-US'},
        'hi': {'name': 'Hindi', 'tts': 'gTTS', 'sr': 'hi-IN'},
        'hinglish': {'name': 'Hinglish', 'tts': 'gTTS', 'sr': 'en-IN'}
    }
    
    def __init__(self, default_language: str = 'en'):
        """
        Initialize language manager with default language.
        
        Args:
            default_language: Initial language code to use
        """
        if default_language not in self.SUPPORTED_LANGUAGES:
            default_language = 'en'
            
        self.current_language = default_language
        self._load_language_mappings()
        logger.info(f"Language Manager initialized with {self.SUPPORTED_LANGUAGES[default_language]['name']}")
        
    def _load_language_mappings(self):
        """Load language-specific word mappings for translations."""
        # This would ideally load from a configuration file
        self.hinglish_map = {
            'hello': 'namaste',
            'thank you': 'dhanyavaad',
            'good': 'achha',
            'yes': 'haan',
            'no': 'nahi',
            'please': 'kripya',
            'welcome': 'swagat hai'
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Detected language code
        """
        try:
            # Check for Devanagari characters (Hindi)
            if any(char in '\u0900-\u097F' for char in text):
                return 'hi'
                
            # Check for Hinglish (mix of English and Hindi structures)
            try:
                lang = langdetect.detect(text)
                confidence = langdetect.detect_langs(text)[0].prob
                
                # If detected as English but with low confidence, might be Hinglish
                if lang == 'en' and confidence < 0.6:
                    # Check for common Hinglish words
                    for word in text.lower().split():
                        if word in self.hinglish_map.values():
                            return 'hinglish'
                    
                return lang if lang in self.SUPPORTED_LANGUAGES else 'en'
            except:
                return 'en'
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return self.current_language
    
    def switch_language(self, lang_code: str) -> bool:
        """
        Change the current interaction language.
        
        Args:
            lang_code: Language code to switch to
            
        Returns:
            bool: Success status of the language change
        """
        if lang_code in self.SUPPORTED_LANGUAGES:
            self.current_language = lang_code
            logger.info(f"Language switched to {self.SUPPORTED_LANGUAGES[lang_code]['name']}")
            return True
        
        logger.warning(f"Unsupported language requested: {lang_code}")
        return False
    
    def convert_to_hinglish(self, text: str) -> str:
        """
        Convert English text to Hinglish using language transfer rules.
        
        Args:
            text: English text to convert
            
        Returns:
            str: Converted Hinglish text
        """
        for eng, hing in self.hinglish_map.items():
            text = text.replace(eng, hing)
        return text
    
    def get_speech_recognition_language(self) -> str:
        """
        Get the appropriate language code for speech recognition.
        
        Returns:
            str: Language code for speech recognition
        """
        return self.SUPPORTED_LANGUAGES[self.current_language]['sr']
    
    def get_tts_engine(self) -> str:
        """
        Get the appropriate text-to-speech engine for current language.
        
        Returns:
            str: TTS engine identifier
        """
        return self.SUPPORTED_LANGUAGES[self.current_language]['tts']


class TTSManager:
    """
    Manages text-to-speech operations across multiple engines.
    Provides unified interface for speech output regardless of engine.
    """
    
    def __init__(self, language_manager: LanguageManager):
        """
        Initialize TTS manager with appropriate engines.
        
        Args:
            language_manager: Reference to the language manager
        """
        self.language_manager = language_manager
        self.engine = pyttsx3.init()
        self._configure_pyttsx()
        logger.info("TTS Manager initialized")
        
    def _configure_pyttsx(self):
        """Configure the pyttsx3 TTS engine properties."""
        self.engine.setProperty('rate', 170)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level
        
        # Set voice to female if available
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
    
    def speak(self, text: str, interrupt: bool = False) -> None:
        """
        Convert text to speech using appropriate engine for current language.
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        if not text:
            return
            
        if interrupt:
            self.stop()
        
        engine_type = self.language_manager.get_tts_engine()
        
        if engine_type == 'pyttsx3':
            self._speak_pyttsx(text)
        else:  # gTTS for other languages
            lang_code = 'hi' if self.language_manager.current_language in ['hi', 'hinglish'] else 'en'
            self._speak_gtts(text, lang_code)
    
    def _speak_pyttsx(self, text: str) -> None:
        """Use pyttsx3 for speech output."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 speech error: {e}")
    
    def _speak_gtts(self, text: str, lang_code: str) -> None:
        """Use Google Text-to-Speech for speech output."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                temp_filename = fp.name
                
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_filename)
            playsound(temp_filename)
            
            # Clean up temp file after playing
            try:
                os.unlink(temp_filename)
            except:
                pass
        except Exception as e:
            logger.error(f"gTTS speech error: {e}")
    
    def stop(self) -> None:
        """Stop any current speech output."""
        try:
            self.engine.stop()
        except:
            pass


class SpeechRecognitionManager:
    """
    Manages speech recognition operations with error handling and adaptability.
    Provides uniform interface for capturing and processing speech input.
    """
    
    def __init__(self, language_manager: LanguageManager):
        """
        Initialize speech recognition components.
        
        Args:
            language_manager: Reference to the language manager
        """
        self.language_manager = language_manager
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # Adjust microphone for ambient noise
        self._calibrate()
        logger.info("Speech Recognition Manager initialized")
    
    def _calibrate(self) -> None:
        """Calibrate the microphone for ambient noise levels."""
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.debug("Microphone calibrated for ambient noise")
        except Exception as e:
            logger.error(f"Microphone calibration error: {e}")
    
    async def listen_async(self, timeout: int = 3, phrase_time_limit: int = 7) -> str:
        """
        Listen for speech input asynchronously.
        
        Args:
            timeout: Maximum seconds to wait for speech start
            phrase_time_limit: Maximum seconds to capture a single phrase
            
        Returns:
            str: Recognized speech text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.listen(timeout, phrase_time_limit)
        )
    
    def listen(self, timeout: int = 3, phrase_time_limit: int = 7) -> str:
        """
        Blocking listen for speech input.
        
        Args:
            timeout: Maximum seconds to wait for speech start
            phrase_time_limit: Maximum seconds to capture a single phrase
            
        Returns:
            str: Recognized speech text
        """
        try:
            with self.mic as source:
                logger.debug("Active listening engaged")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                # Get appropriate language for recognition
                language = self.language_manager.get_speech_recognition_language()
                
                text = self.recognizer.recognize_google(audio, language=language)
                logger.info(f"Heard ({language}): {text}")
                return text.lower()
        except sr.UnknownValueError:
            logger.debug("No intelligible speech detected")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Listening error: {e}")
            return ""


class AIResponseGenerator:
    """
    Generates AI-powered responses to user queries.
    Handles API communication and response formatting.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI response generator with API credentials.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
            logger.info("AI Response Generator initialized")
        else:
            logger.warning("No API key provided for AI Response Generator")
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Generate AI response to user query with context awareness.
        
        Args:
            query: User's question or command
            context: Additional context information
            
        Returns:
            str: AI-generated response
        """
        if not self.api_key:
            return "I need to check my knowledge sources to answer that properly."
            
        try:
            system_content = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if context:
                system_content += f"\nContext: {json.dumps(context)}"
                
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"{query} Provide concise answer with verified sources."}
                ],
                temperature=0.4,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "I need to check my knowledge sources to answer that properly."


class EmergencyManager:
    """
    Handles emergency detection and appropriate response protocols.
    Provides critical response capabilities for urgent situations.
    """
    
    def __init__(self):
        """Initialize emergency response systems."""
        self.emergency_phrases = [
            "help", "emergency", "stop", "danger", "fire", 
            "accident", "urgent", "critical", "emergency"
        ]
        self.protocol_active = False
        logger.info("Emergency Manager initialized")
    
    def is_emergency(self, text: str) -> bool:
        """
        Detect if text contains emergency indicators.
        
        Args:
            text: Text to analyze for emergency content
            
        Returns:
            bool: True if emergency detected
        """
        return any(phrase in text.lower() for phrase in self.emergency_phrases)
    
    def handle_emergency(self, text: str) -> str:
        """
        Implement emergency response protocol.
        
        Args:
            text: Emergency text that triggered the response
            
        Returns:
            str: Emergency response message
        """
        self.protocol_active = True
        logger.critical(f"EMERGENCY DETECTED: {text}")
        
        # In a real implementation, this would trigger appropriate emergency actions
        response = "Emergency protocol activated. What's the nature of the emergency?"
        
        return response
    
    def deactivate_protocol(self) -> None:
        """Deactivate emergency protocol when situation resolved."""
        if self.protocol_active:
            self.protocol_active = False
            logger.info("Emergency protocol deactivated")


class VisionServiceConnector:
    """
    Interface for connecting to external vision systems.
    Provides methods to control vision service operations.
    """
    
    def __init__(self):
        """Initialize vision service connector."""
        self.vision_active = False
        logger.info("Vision Service Connector initialized")
    
    def start_vision_service(self) -> bool:
        """
        Activate the vision service.
        
        Returns:
            bool: Success status
        """
        self.vision_active = True
        logger.info("Vision service activated")
        return True
    
    def stop_vision_service(self) -> bool:
        """
        Deactivate the vision service.
        
        Returns:
            bool: Success status
        """
        self.vision_active = False
        logger.info("Vision service deactivated")
        return True
    
    def is_active(self) -> bool:
        """
        Check if vision service is currently active.
        
        Returns:
            bool: Active status
        """
        return self.vision_active


class CommandProcessor:
    """
    Processes and routes voice commands to appropriate handlers.
    Provides centralized command handling with priority management.
    """
    
    def __init__(self, 
                 tts_manager: TTSManager,
                 emergency_manager: EmergencyManager,
                 vision_connector: Optional[VisionServiceConnector] = None,
                 ai_generator: Optional[AIResponseGenerator] = None,
                 message_monitor: Optional[MessageMonitor] = None,
                 language_manager: Optional[LanguageManager] = None):
        """
        Initialize command processor with required service references.
        
        Args:
            tts_manager: Text-to-speech manager
            emergency_manager: Emergency response manager
            vision_connector: Vision service connector
            ai_generator: AI response generator
            message_monitor: Message monitoring service
            language_manager: Language management service
        """
        self.tts_manager = tts_manager
        self.emergency_manager = emergency_manager
        self.vision_connector = vision_connector
        self.ai_generator = ai_generator
        self.message_monitor = message_monitor
        self.language_manager = language_manager
        
        logger.info("Command Processor initialized")
    
    def process_command(self, text: str) -> None:
        """
        Process and route commands based on content and priority.
        
        Args:
            text: Command text to process
        """
        if not text:
            return
            
        # Handle language switching commands first
        if self.language_manager and self._is_language_command(text):
            return
            
        # Emergency commands take highest priority
        if self.emergency_manager.is_emergency(text):
            response = self.emergency_manager.handle_emergency(text)
            self.tts_manager.speak(response, interrupt=True)
            return
            
        # Vision system commands
        if self.vision_connector and self._is_vision_command(text):
            self._handle_vision_command(text)
            return
            
        # Message handling commands
        if self.message_monitor and self._is_message_command(text):
            self._handle_message_command(text)
            return
            
        # AI-powered question answering (if none of the above)
        if self.ai_generator and self._is_question(text):
            response = self.ai_generator.generate_response(text)
            self.tts_manager.speak(response)
    
    def _is_language_command(self, text: str) -> bool:
        """
        Check if command is related to language switching.
        
        Args:
            text: Command text
            
        Returns:
            bool: True if language command detected
        """
        if not self.language_manager:
            return False
            
        lang_cmd_patterns = {
            'en': ["switch to english", "speak english", "english please", "अंग्रेज़ी"],
            'hi': ["switch to hindi", "speak hindi", "hindi please", "हिंदी"],
            'hinglish': ["switch to hinglish", "speak hinglish", "hinglish please", "हिंग्लिश"]
        }
        
        for lang, patterns in lang_cmd_patterns.items():
            if any(pattern in text.lower() for pattern in patterns):
                self.language_manager.switch_language(lang)
                self.tts_manager.speak(f"Language switched to {self.language_manager.SUPPORTED_LANGUAGES[lang]['name']}")
                return True
                
        return False
    
    def _is_vision_command(self, text: str) -> bool:
        """
        Check if command is related to vision system.
        
        Args:
            text: Command text
            
        Returns:
            bool: True if vision command detected
        """
        vision_patterns = [
            "camera", "vision", "look", "see", "watch", 
            "visual", "display", "show", "view"
        ]
        return any(pattern in text.lower() for pattern in vision_patterns)
    
    def _handle_vision_command(self, text: str) -> None:
        """
        Process vision-related commands.
        
        Args:
            text: Vision command text
        """
        if any(cmd in text.lower() for cmd in ["start", "enable", "activate", "turn on"]):
            self.vision_connector.start_vision_service()
            self.tts_manager.speak("Vision system activated")
        elif any(cmd in text.lower() for cmd in ["stop", "disable", "deactivate", "turn off"]):
            self.vision_connector.stop_vision_service()
            self.tts_manager.speak("Vision system deactivated")
        else:
            self.tts_manager.speak("Vision command not recognized")
    
    def _is_message_command(self, text: str) -> bool:
        """
        Check if command is related to messaging.
        
        Args:
            text: Command text
            
        Returns:
            bool: True if message command detected
        """
        message_patterns = ["message", "notification", "text", "chat", "reply", "respond"]
        return any(pattern in text.lower() for pattern in message_patterns)
    
    def _handle_message_command(self, text: str) -> None:
        """
        Process message-related commands.
        
        Args:
            text: Message command text
        """
        new_msg = self.message_monitor.check_messages()
        if not new_msg:
            self.tts_manager.speak("No new messages found")
            return
            
        self.tts_manager.speak(f"New message from {new_msg['sender']}: {new_msg['content']}")
    
    def _is_question(self, text: str) -> bool:
        """
        Determine if text is a question or query.
        
        Args:
            text: Text to analyze
            
        Returns:
            bool: True if likely a question
        """
        question_indicators = ["?", "what", "how", "why", "when", "where", "who", "which", "can", "could", "tell me"]
        return any(indicator in text.lower() for indicator in question_indicators)


class NeuroVoiceManager:
    """
    Advanced AI-powered voice interface with continuous intelligence.
    Central coordinator managing all voice interaction components.
    """
    
    def __init__(self):
        """Initialize the Neural Voice Management system."""
        # Load environment configuration
        load_dotenv()
        
        # Initialize core components
        self.language_manager = LanguageManager(default_language='en')
        self.tts_manager = TTSManager(self.language_manager)
        self.sr_manager = SpeechRecognitionManager(self.language_manager)
        self.message_monitor = MessageMonitor()
        self.emergency_manager = EmergencyManager()
        self.vision_connector = VisionServiceConnector()
        self.ai_generator = AIResponseGenerator()
        
        # Initialize command processor with dependencies
        self.command_processor = CommandProcessor(
            tts_manager=self.tts_manager,
            emergency_manager=self.emergency_manager,
            vision_connector=self.vision_connector,
            ai_generator=self.ai_generator,
            message_monitor=self.message_monitor,
            language_manager=self.language_manager
        )
        
        # Continuous listening setup
        self.listening_active = True
        self.command_queue = Queue()
        self.listening_thread = threading.Thread(target=self._continuous_listen)
        self.listening_thread.daemon = True
        
        logger.info("Neural Voice Manager initialized successfully")
    
    def start(self) -> None:
        """Start the voice management system."""
        logger.info("Starting Neural Voice Manager")
        self.tts_manager.speak("Voice interface activated")
        self.start_continuous_listening()
    
    def stop(self) -> None:
        """Gracefully shut down the voice management system."""
        logger.info("Stopping Neural Voice Manager")
        self.listening_active = False
        self.tts_manager.speak("Voice interface deactivated")
        self.tts_manager.stop()
        
        # Wait for listening thread to complete (max 2 seconds)
        if self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2.0)
            
        logger.info("Neural Voice Manager shutdown complete")
    
    def start_continuous_listening(self) -> None:
        """Enable always-on background listening."""
        self.listening_active = True
        if not self.listening_thread.is_alive():
            self.listening_thread.start()
        logger.info("Continuous listening activated")
    
    def _continuous_listen(self) -> None:
        """Background listening loop for continuous operation."""
        while self.listening_active:
            try:
                text = self.sr_manager.listen()
                if text:
                    self.command_processor.process_command(text)
                    self._check_messages()
                # Small sleep to prevent CPU overuse
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Continuous listen error: {e}")
                time.sleep(1)
    
    def _check_messages(self) -> None:
        """Periodically check for new messages."""
        try:
            new_msg = self.message_monitor.check_messages()
            if new_msg:
                self.tts_manager.speak(f"New message from {new_msg['sender']}: {new_msg['content']}")
                
                # Wait for response
                self.tts_manager.speak("Would you like to respond?")
                response = self.sr_manager.listen(timeout=15)
                
                if response and any(word in response.lower() for word in ["yes", "yeah", "sure", "okay"]):
                    self.tts_manager.speak("What's your response?")
                    reply = self.sr_manager.listen(timeout=15)
                    
                    if reply:
                        self.message_monitor.send_response(new_msg['app'], reply)
                        self.tts_manager.speak("Response sent successfully")
        except Exception as e:
            logger.error(f"Message check error: {e}")
    
    async def listen_async(self) -> str:
        """
        Asynchronous high-priority listening for immediate commands.
        
        Returns:
            str: Recognized text from speech
        """
        return await self.sr_manager.listen_async()
    
    def speak(self, text: str, interrupt: bool = False) -> None:
        """
        Speak text using appropriate TTS engine.
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        # Detect language if it might be different from current setting
        detected_lang = self.language_manager.detect_language(text)
        if detected_lang != self.language_manager.current_language:
            # If language changed, apply special handling
            orig_lang = self.language_manager.current_language
            self.language_manager.switch_language(detected_lang)
            self.tts_manager.speak(text, interrupt)
            # Restore original language
            self.language_manager.switch_language(orig_lang)
        else:
            # Same language, normal speech
            self.tts_manager.speak(text, interrupt)


# Example usage
if __name__ == "__main__":
    try:
        # Initialize and start the voice manager
        voice_manager = NeuroVoiceManager()
        voice_manager.start()
        
        # Keep main thread alive to allow background processing
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Graceful shutdown on Ctrl+C
            pass
        finally:
            voice_manager.stop()
            
    except Exception as e:
        logger.critical(f"Critical error in voice manager: {e}")