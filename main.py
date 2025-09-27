# main.py
from dotenv import load_dotenv
load_dotenv()  # 👈 Before any other imports!
from pydub import AudioSegment
AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"
# Now access variables like:
import os
openai_key = os.getenv("OPENAI_API_KEY")
import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional

# Core components
from access import AuraAssistant
from ethical import CyberOperationsController, AuthorizationSystem
from map_service import EthicalMapService
from memory_manager import KnowledgeGraph
from neuro_web_controller import NeuroWebController
from nlp_processor import EnhancedNLPEngine, ConversationContext, NLPAnalysisResult, AIProvider, AnalysisMetrics
from security import SecurityHandler
from system_monitor import NeuroSystemMonitor
from vision_processor import AdvancedVisionSystem
from voice_interface import NeuroVoiceManager

class AURACore:
    def __init__(self):
        """Central controller for AURA AI system"""
        self._configure_system()
        self._initialize_components()
        self._establish_cross_links()
        self._start_background_services()
        
        logger.info("AURA AI System Initialized")

    def _configure_system(self):
        """Initialize system-wide configurations"""
        self.running = True
        self.command_queue = []
        self.security_status = "SECURE"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s',
            handlers=[
                logging.FileHandler("aura_core.log"),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger('AURA.Core')

    def _initialize_components(self):
        """Initialize all system components"""
        # Security First
        self.security = SecurityHandler(security_level="maximum")
        
        # Core modules
        self.voice = NeuroVoiceManager()
        self.vision = AdvancedVisionSystem()
        self.nlp = EnhancedNLPEngine(),ConversationContext(), NLPAnalysisResult(), AIProvider(), AnalysisMetrics()
        self.memory = KnowledgeGraph(encryption_key=self.security.key)
        self.access = AuraAssistant()
        self.cyber = CyberOperationsController(AuthorizationSystem())
        self.maps = EthicalMapService(self.cyber)
        self.web = NeuroWebController()
        self.monitor = NeuroSystemMonitor()
        
        # System integrations
        self.auth = AuthorizationSystem()

    def _establish_cross_links(self):
        """Connect components into neural network"""
        # Voice integrations
        self.voice.link_services(self.vision, self.nlp)
        
        # Security integrations
        self.cyber.security = self.security
        self.web.security = self.security
        self.memory.security = self.security
        
        # Knowledge sharing
        self.nlp.context_memory = self.memory
        self.vision.face_db = self.memory
        
        # Hardware access
        self.cyber.link_hardware_controller(self.access)
        
        logger.info("Neural Component Network Established")

    def _start_background_services(self):
        """Start essential background services"""
        # Security monitor
        self.security_thread = threading.Thread(target=self._security_daemon)
        self.security_thread.daemon = True
        self.security_thread.start()
        
        # System health monitor
        self.health_thread = threading.Thread(target=self.monitor._monitoring_loop)
        self.health_thread.daemon = True
        self.health_thread.start()
        
        # Continuous voice listening
        self.voice.start_continuous_listening()

    async def main_loop(self):
        """Primary async control loop"""
        try:
            while self.running:
                await self._process_commands()
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            self.graceful_shutdown()

    def _security_daemon(self):
        """Real-time security monitoring"""
        while self.running:
            try:
                audit = self.security.get_security_audit(1)
                if audit['suspicious_activity_count'] > 0:
                    self._handle_security_alert()
                
                if self.monitor.get_recommendations():
                    self._execute_system_actions()
                    
                time.sleep(10)
            except Exception as e:
                logger.error(f"Security daemon error: {e}")

    def _process_commands(self):
        """Handle incoming voice commands"""
        if not self.voice.command_queue.empty():
            command = self.voice.command_queue.get()
            logger.info(f"Processing command: {command}")
            
            # Security verification
            if not self.security.validate_command(command):
                logger.warning("Command security check failed")
                return
                
            response = self._route_command(command)
            self._generate_response(response)

    def _route_command(self, command: str) -> Dict:
        """Intelligent command routing"""
        try:
            # Security emergency protocol
            if self._is_emergency(command):
                return self.cyber.emergency_protocol(command)
            
            # Hardware control
            if any(keyword in command for keyword in ["wifi", "data", "torch"]):
                return self.access.execute_hardware_command(command)
            
            # Navigation commands
            if any(keyword in command for keyword in ["navigate", "route", "location"]):
                return self.maps.handle_voice_command(command)
            
            # Web operations
            if any(keyword in command for keyword in ["search", "download", "browser"]):
                return self.web.handle_voice_command(command)
            
            # Vision system
            if any(keyword in command for keyword in ["camera", "recognize", "vision"]):
                return self.vision.handle_voice_command(command)
            
            # Default NLP processing
            return self._handle_conversation(command)
            
        except Exception as e:
            logger.error(f"Command routing failed: {e}")
            return {"error": "Command processing failed"}

    def _handle_conversation(self, command: str) -> Dict:
        """Handle natural language interaction"""
        analysis = self.nlp.analyze(command)
        response = self.nlp.generate_response(analysis)
        return {
            "type": "conversation",
            "response": response,
            "analysis": analysis
        }

    def _generate_response(self, response: Dict):
        """Handle system responses"""
        if 'response' in response:
            self.voice.speak(response['response'])
        elif 'message' in response:
            self.voice.speak(response['message'])
            
        # Log interaction in knowledge graph
        self.memory.store_context({
            "command": response.get('original_command'),
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

    def graceful_shutdown(self):
        """Safe system shutdown procedure"""
        logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Shutdown sequence
        components = [
            self.voice,
            self.vision,
            self.web,
            self.cyber,
            self.memory,
            self.monitor
        ]
        
        for component in components:
            try:
                if hasattr(component, "shutdown"):
                    component.shutdown()
                elif hasattr(component, "close"):
                    component.close()
                logger.info(f"Shut down {type(component).__name__}")
            except Exception as e:
                logger.error(f"Error shutting down {type(component).__name__}: {e}")
        
        logger.info("AURA shutdown complete")

if __name__ == "__main__":
    aura = AURACore()
    
    try:
        # Start async main loop
        asyncio.run(aura.main_loop())
    except KeyboardInterrupt:
        aura.graceful_shutdown()