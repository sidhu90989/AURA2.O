
"""
AuraAssistant: A cross-platform system control and automation utility.

This module provides a comprehensive interface for controlling various system functions
across different operating systems including Windows, macOS, Linux, and Android.
"""

import os
import sys
import platform
import subprocess
import time
import ctypes
import shutil
import logging
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AuraAssistant")

class OSType(Enum):
    """Enumeration of supported operating systems."""
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    ANDROID = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, os_name: str) -> 'OSType':
        """Convert string OS name to OSType enum value."""
        os_map = {
            'windows': cls.WINDOWS,
            'linux': cls.LINUX,
            'darwin': cls.MACOS,
            'android': cls.ANDROID
        }
        return os_map.get(os_name.lower(), cls.UNKNOWN)


class MediaPlayer:
    """Handles media playback functionality."""
    
    def __init__(self):
        """Initialize the MediaPlayer with proper configurations."""
        # Configure audio libraries
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            self.AudioSegment = AudioSegment
            self.play = play
            
            # Set custom ffmpeg path for Windows if available
            if platform.system().lower() == 'windows' and os.path.exists("C:/ffmpeg/bin/ffmpeg.exe"):
                self.AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"
                logger.info("Custom ffmpeg path set successfully.")
            
            self.initialized = True
            logger.info("MediaPlayer initialized successfully.")
        except ImportError:
            logger.warning("pydub module not found. Audio playback functionality will be limited.")
            self.initialized = False

    def play_sound(self, file_path: str) -> bool:
        """
        Play an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("MediaPlayer not properly initialized.")
            return False
            
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            if file_path.suffix.lower() == '.mp3':
                audio = self.AudioSegment.from_mp3(str(file_path))
                self.play(audio)
                return True
            elif file_path.suffix.lower() == '.wav':
                audio = self.AudioSegment.from_wav(str(file_path))
                self.play(audio)
                return True
            else:
                logger.error(f"Unsupported audio format: {file_path.suffix}")
                return False
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            return False


class AuraAssistant:
    """
    A cross-platform assistant that provides system control and automation capabilities.
    
    Features:
    - OS detection and privilege management
    - Network control (WiFi, mobile data)
    - Device control (torch, screen)
    - Communication tools (calls, SMS)
    - File management
    - Application control
    - System commands execution
    - Media playback
    - Text-to-speech functionality
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the AuraAssistant.
        
        Args:
            verbose: Whether to enable verbose logging
        """
        # Set up logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Detect system information
        self._os_name = platform.system().lower()
        self._os_type = self._detect_os()
        self._admin_privileges = self._check_admin()
        
        # Initialize media player
        self.media_player = MediaPlayer()
        
        logger.info(f"AuraAssistant initialized on {self._os_name} with admin privileges: {self._admin_privileges}")

    def _detect_os(self) -> OSType:
        """
        Detect the operating system.
        
        Returns:
            OSType: Enum representing the detected operating system
        """
        if 'android' in self._os_name:
            return OSType.ANDROID
        
        os_mapping = {
            'windows': OSType.WINDOWS,
            'linux': OSType.LINUX,
            'darwin': OSType.MACOS
        }
        
        return os_mapping.get(self._os_name, OSType.UNKNOWN)

    def _check_admin(self) -> bool:
        """
        Check if the application is running with administrative privileges.
        
        Returns:
            bool: True if running with admin privileges, False otherwise
        """
        try:
            if self._os_type == OSType.WINDOWS:
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            elif self._os_type in [OSType.LINUX, OSType.MACOS]:
                return os.geteuid() == 0 if hasattr(os, "geteuid") else False
            return False
        except Exception as e:
            logger.error(f"Error checking admin privileges: {str(e)}")
            return False
            
    @property
    def os_type(self) -> OSType:
        """Get the detected operating system type."""
        return self._os_type
        
    @property
    def admin_privileges(self) -> bool:
        """Check if the application has administrative privileges."""
        return self._admin_privileges

    def _run_command(self, cmd: Union[str, List[str]], shell: bool = False) -> Dict[str, Any]:
        """
        Run a system command and capture its output.
        
        Args:
            cmd: Command to execute (string or list of strings)
            shell: Whether to use shell execution
            
        Returns:
            dict: Dictionary containing command results
        """
        try:
            result = subprocess.run(
                cmd, 
                shell=shell, 
                capture_output=True, 
                text=True,
                timeout=30  # Add timeout for safety
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {cmd}")
            return {'success': False, 'error': 'Command timed out'}
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {'success': False, 'error': str(e)}

    # ------------------ Network Control ------------------
    def toggle_wifi(self, state: bool) -> bool:
        """
        Toggle WiFi state.
        
        Args:
            state: True to enable, False to disable
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Attempting to {'enable' if state else 'disable'} WiFi")
        
        if not self._admin_privileges:
            logger.warning("Admin privileges required to toggle WiFi")
            return False
            
        try:
            if self._os_type == OSType.WINDOWS:
                cmd = f'netsh interface set interface "Wi-Fi" {"admin=enabled" if state else "admin=disabled"}'
                result = self._run_command(cmd, shell=True)
            elif self._os_type == OSType.LINUX:
                state_str = 'on' if state else 'off'
                result = self._run_command(['nmcli', 'radio', 'wifi', state_str])
            elif self._os_type == OSType.MACOS:
                state_str = 'on' if state else 'off'
                result = self._run_command(['networksetup', '-setairportpower', 'en0', state_str])
            else:
                logger.error(f"WiFi toggling not supported on {self._os_name}")
                return False
                
            return result['success']
        except Exception as e:
            logger.error(f"Error toggling WiFi: {str(e)}")
            return False

    def toggle_mobile_data(self, state: bool) -> bool:
        """
        Toggle mobile data state (Android only).
        
        Args:
            state: True to enable, False to disable
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self._os_type != OSType.ANDROID:
            logger.error("Mobile data toggling is only supported on Android")
            return False
            
        try:
            cmd = ['adb', 'shell', 'svc', 'data', 'enable' if state else 'disable']
            result = self._run_command(cmd)
            return result['success']
        except Exception as e:
            logger.error(f"Error toggling mobile data: {str(e)}")
            return False
            
    # ------------------ Device Control ------------------
    def toggle_torch(self, state: bool) -> bool:
        """
        Toggle device torch/flashlight (Android only).
        
        Args:
            state: True to enable, False to disable
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self._os_type != OSType.ANDROID:
            logger.error("Torch control is only supported on Android")
            return False
            
        try:
            # This is a workaround since direct torch control requires root
            result1 = self._run_command([
                'adb', 'shell', 'am', 'start', '-a', 
                'android.media.action.IMAGE_CAPTURE'
            ])
            
            time.sleep(1)
            
            result2 = self._run_command([
                'adb', 'shell', 'input', 'keyevent', '26'
            ])
            
            return result1['success'] and result2['success']
        except Exception as e:
            logger.error(f"Error controlling torch: {str(e)}")
            return False

    # ------------------ Communication ------------------
    def make_call(self, number: str) -> bool:
        """
        Initiate a phone call (Android only).
        
        Args:
            number: Phone number to call
            
        Returns:
            bool: True if call initiation was successful, False otherwise
        """
        if self._os_type != OSType.ANDROID:
            logger.error("Call functionality is only supported on Android")
            return False
            
        if not number or not isinstance(number, str):
            logger.error("Invalid phone number provided")
            return False
            
        try:
            cmd = [
                'adb', 'shell', 'am', 'start', '-a', 
                'android.intent.action.CALL', '-d', f'tel:{number}'
            ]
            result = self._run_command(cmd)
            return result['success']
        except Exception as e:
            logger.error(f"Error making call: {str(e)}")
            return False
            
    def send_sms(self, number: str, message: str) -> bool:
        """
        Send SMS message (Android only).
        
        Args:
            number: Recipient phone number
            message: Message content
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self._os_type != OSType.ANDROID:
            logger.error("SMS functionality is only supported on Android")
            return False
            
        if not number or not message:
            logger.error("Both number and message must be provided")
            return False
            
        try:
            cmd = [
                'adb', 'shell', 'service', 'call', 'isms', '5', 
                's16', 'com.android.mms.service', 's16', 'null', 
                's16', number, 's16', 'null', 's16', message
            ]
            result = self._run_command(cmd)
            return result['success']
        except Exception as e:
            logger.error(f"Error sending SMS: {str(e)}")
            return False

    # ------------------ File Management ------------------
    def find_files(self, extensions: List[str], search_paths: Optional[List[str]] = None, 
                   max_files: int = 100, max_depth: int = 5) -> List[str]:
        """
        Find files with specific extensions.
        
        Args:
            extensions: List of file extensions to find (without dots)
            search_paths: List of paths to search (defaults to standard locations)
            max_files: Maximum number of files to return
            max_depth: Maximum directory depth to search
            
        Returns:
            List[str]: List of found file paths
        """
        if not extensions:
            logger.error("No file extensions provided")
            return []
            
        # Clean extensions format
        clean_extensions = [ext.lower().lstrip('.') for ext in extensions]
        
        # Set default search paths based on OS
        if not search_paths:
            if self._os_type == OSType.WINDOWS:
                search_paths = [os.path.expanduser('~')]
            elif self._os_type in [OSType.LINUX, OSType.MACOS]:
                search_paths = [os.path.expanduser('~'), '/usr/local']
            else:
                search_paths = [os.path.expanduser('~')]
                
        found_files = []
        
        for base_path in search_paths:
            if not os.path.exists(base_path):
                logger.warning(f"Search path does not exist: {base_path}")
                continue
                
            try:
                # Convert to Path object for safer path manipulation
                base = Path(base_path)
                # Use os.walk with topdown=True to allow modifying dirs list in-place
                for current_depth, (root, dirs, files) in enumerate(os.walk(base_path)):
                    # Stop if we've reached max depth
                    if current_depth >= max_depth:
                        del dirs[:]  # Clear dirs to prevent deeper recursion
                        continue
                        
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    # Find matching files
                    for file in files:
                        if len(found_files) >= max_files:
                            return found_files
                            
                        file_ext = Path(file).suffix.lstrip('.').lower()
                        if file_ext in clean_extensions:
                            full_path = os.path.join(root, file)
                            found_files.append(full_path)
            except Exception as e:
                logger.error(f"Error searching in {base_path}: {str(e)}")
                
        return found_files

    # ------------------ Application Control ------------------
    def open_app(self, app_name: str) -> bool:
        """
        Open an application.
        
        Args:
            app_name: Name of the application to open
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not app_name:
            logger.error("No application name provided")
            return False
            
        try:
            if self._os_type == OSType.WINDOWS:
                result = self._run_command(['start', '', app_name], shell=True)
            elif self._os_type == OSType.LINUX:
                result = self._run_command(['xdg-open', app_name])
            elif self._os_type == OSType.MACOS:
                result = self._run_command(['open', '-a', app_name])
            elif self._os_type == OSType.ANDROID:
                result = self._run_command(['adb', 'shell', 'monkey', '-p', app_name, '1'])
            else:
                logger.error(f"Application opening not supported on {self._os_name}")
                return False
                
            return result['success']
        except Exception as e:
            logger.error(f"Error opening application {app_name}: {str(e)}")
            return False

    # ------------------ System Control ------------------
    def system_command(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a system command.
        
        Args:
            cmd: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Dict: Result dictionary with keys 'success', 'stdout', 'stderr', 'returncode'
        """
        if not cmd:
            return {'success': False, 'error': 'No command provided'}
            
        # Security check - prevent dangerous commands
        dangerous_commands = ['rm -rf', 'format', 'deltree', ':(){:|:&};:']
        if any(dangerous in cmd.lower() for dangerous in dangerous_commands):
            logger.error(f"Potentially dangerous command rejected: {cmd}")
            return {'success': False, 'error': 'Command rejected for security reasons'}
            
        try:
            return self._run_command(cmd, shell=True)
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def lock_screen(self) -> bool:
        """
        Lock the system screen.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self._os_type == OSType.WINDOWS:
                ctypes.windll.user32.LockWorkStation()
                return True
            elif self._os_type == OSType.LINUX:
                result = self._run_command(['gnome-screensaver-command', '-l'])
                if not result['success']:
                    # Try alternative methods for different desktop environments
                    alt_cmds = [
                        ['xdg-screensaver', 'lock'],
                        ['loginctl', 'lock-session'],
                        ['dm-tool', 'lock']
                    ]
                    for cmd in alt_cmds:
                        result = self._run_command(cmd)
                        if result['success']:
                            return True
                    return False
                return True
            elif self._os_type == OSType.MACOS:
                # Try multiple methods for macOS
                cmds = [
                    ['pmset', 'displaysleepnow'],
                    ['osascript', '-e', 'tell application "System Events" to keystroke "q" using {command down, control down}']
                ]
                for cmd in cmds:
                    result = self._run_command(cmd)
                    if result['success']:
                        return True
                return False
            else:
                logger.error(f"Screen locking not supported on {self._os_name}")
                return False
        except Exception as e:
            logger.error(f"Error locking screen: {str(e)}")
            return False

    # ------------------ Media Control ------------------
    def play_sound(self, file_path: str) -> bool:
        """
        Play an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.media_player.play_sound(file_path)

    # ------------------ Utility Functions ------------------
    def speak(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.error("No text provided for speech")
            return False
            
        try:
            if self._os_type == OSType.WINDOWS:
                try:
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(text)
                    return True
                except ImportError:
                    logger.error("win32com.client module not available")
                    return False
            elif self._os_type == OSType.MACOS:
                result = self._run_command(['say', text])
                return result['success']
            elif self._os_type == OSType.LINUX:
                # Try multiple TTS engines
                tts_commands = [
                    ['espeak', text],
                    ['festival', '--tts'],
                    ['pico2wave', '-w', '/tmp/temp.wav', text]
                ]
                
                for cmd in tts_commands:
                    if cmd[0] == 'festival':
                        # Festival needs text via stdin
                        try:
                            process = subprocess.Popen(
                                cmd, 
                                stdin=subprocess.PIPE, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True
                            )
                            stdout, stderr = process.communicate(input=text, timeout=10)
                            if process.returncode == 0:
                                return True
                        except:
                            continue
                    elif cmd[0] == 'pico2wave':
                        # pico2wave writes to file and needs a separate play command
                        try:
                            result1 = self._run_command(cmd)
                            if result1['success']:
                                result2 = self._run_command(['aplay', '/tmp/temp.wav'])
                                return result2['success']
                        except:
                            continue
                    else:
                        # Standard command
                        result = self._run_command(cmd)
                        if result['success']:
                            return True
                
                logger.error("No working text-to-speech engine found")
                return False
            else:
                logger.error(f"Text-to-speech not supported on {self._os_name}")
                return False
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return False


def main():
    """Example usage of AuraAssistant."""
    aura = AuraAssistant(verbose=True)
    
    # System information
    print(f"OS Type: {aura.os_type}")
    print(f"Admin Privileges: {aura.admin_privileges}")
    
    # Example: Text-to-speech
    aura.speak("Aura Assistant is now active")
    
    # Example: Finding files
    pdf_files = aura.find_files(['pdf'], max_files=5)
    print(f"Found {len(pdf_files)} PDF files")
    
    # More examples could be added here


if __name__ == "__main__":
    main()
