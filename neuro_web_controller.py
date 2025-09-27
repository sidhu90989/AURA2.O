import os
import re
import time
import logging
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sentinelhub import WmsRequest, SHConfig, BBox, CRS
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from urllib.parse import urlparse
import hashlib
from cryptography.fernet import Fernet
import magic
import tempfile
import shutil
import json



# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aura_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AURA.Web')

class NeuroWebController:
    def __init__(self, voice_interface=None):
        """AI-powered web services with voice integration"""
        # Browser configuration
        self.setup_browser()
        
        # Satellite configuration
        self.setup_satellite_config()
        
        # Document management
        self.setup_document_management()
        
        # Voice integration
        self.voice = voice_interface
        self.command_patterns = {
            'download_doc': r"(download|find) (?:the )?(.+?) (pdf|doc|docx)",
            'satellite_data': r"show (?:me )?(satellite|aerial) data for (.+)",
            'web_search': r"(search|look up|find information about) (.+)"
        }
        
        logger.info("Neural Web Controller initialized")

    def setup_browser(self):
        """Set up browser with proper configuration for Python 3.8.10"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            # Using webdriver-manager for automatic ChromeDriver management
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            logger.error(f"Failed to initialize Chrome: {e}")
            # Fallback to direct ChromeDriver path if webdriver-manager fails
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
            except Exception as e2:
                logger.critical(f"Chrome initialization completely failed: {e2}")
                raise RuntimeError(f"Could not initialize Chrome browser: {e2}")

    def setup_satellite_config(self):
        """Set up Sentinel Hub configuration"""
        self.sh_config = SHConfig()
        client_id = os.getenv("SENTINEL_CLIENT_ID")
        client_secret = os.getenv("SENTINEL_CLIENT_SECRET")
        
        if client_id and client_secret:
            self.sh_config.sh_client_id = client_id
            self.sh_config.sh_client_secret = client_secret
            self.satellite_available = True
        else:
            logger.warning("Sentinel Hub credentials not found. Satellite functions will be limited.")
            self.satellite_available = False

    def setup_document_management(self):
        """Set up document management with proper error handling"""
        try:
            # Create download directory with proper permissions
            self.download_dir = Path("~/AURA_Documents").expanduser()
            self.download_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a secure temporary directory for processing files
            self.temp_dir = Path(tempfile.mkdtemp(prefix="aura_"))
            
            # Generate or load encryption key
            key_file = self.download_dir / ".key"
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                # Secure the key file
                os.chmod(key_file, 0o600)
                
            logger.info(f"Document management initialized at {self.download_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize document management: {e}")
            # Fallback to temporary directory
            self.download_dir = Path(tempfile.mkdtemp(prefix="aura_docs_"))
            self.temp_dir = Path(tempfile.mkdtemp(prefix="aura_temp_"))
            self.encryption_key = Fernet.generate_key()
            logger.warning(f"Using fallback temporary directory: {self.download_dir}")

    def _secure_download(self, url: str) -> Optional[Path]:
        """Safely download and verify documents"""
        try:
            # Validate URL
            if not re.match(r"^https?://", url):
                raise ValueError("Invalid URL format")

            # Create a temporary file for downloading
            fd, temp_path = tempfile.mkstemp(dir=self.temp_dir)
            os.close(fd)
            temp_path = Path(temp_path)
            
            # Download file with progress tracking
            response = requests.get(url, stream=True, timeout=15)
            
            # Security checks
            if response.status_code != 200:
                raise ConnectionError(f"Download failed with status code {response.status_code}")
            
            # Save content to temporary file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file type
            mime_type = magic.Magic(mime=True).from_buffer(open(temp_path, 'rb').read(2048))
            allowed_types = [
                'application/pdf', 
                'application/msword', 
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ]
            
            if mime_type not in allowed_types:
                raise ValueError(f"Unsupported file type: {mime_type}")

            # Move to final destination with secure name
            file_hash = self._calculate_file_hash(temp_path)
            final_name = f"{file_hash[:8]}_{os.path.basename(urlparse(url).path)}"
            final_path = self.download_dir / final_name
            
            # Copy the file to final destination
            shutil.copy2(temp_path, final_path)
            
            # Encrypt the final file
            self._encrypt_file(final_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            logger.info(f"Successfully downloaded and secured file: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _encrypt_file(self, path: Path) -> bool:
        """Encrypt documents using Fernet with proper error handling"""
        try:
            cipher = Fernet(self.encryption_key)
            # Create a temporary file for the encrypted content
            temp_encrypted = path.with_suffix(path.suffix + ".tmp")
            
            with open(path, 'rb') as f_in:
                file_data = f_in.read()
                encrypted_data = cipher.encrypt(file_data)
                
            with open(temp_encrypted, 'wb') as f_out:
                f_out.write(encrypted_data)
            
            # Replace the original file with the encrypted version
            temp_encrypted.replace(path)
            return True
        except Exception as e:
            logger.error(f"Encryption failed for {path}: {str(e)}")
            return False

    def _decrypt_file(self, path: Path) -> Optional[bytes]:
        """Decrypt a file and return its contents"""
        try:
            cipher = Fernet(self.encryption_key)
            with open(path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = cipher.decrypt(encrypted_data)
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed for {path}: {str(e)}")
            return None

    def find_document(self, doc_type: str, keywords: str) -> Optional[Path]:
        """AI-powered document search and retrieval"""
        try:
            # Web search using Google Custom Search API
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found in environment")
                
            search_query = f"{keywords} filetype:{doc_type}"
            search_url = f"https://www.googleapis.com/customsearch/v1"
            
            params = {
                'q': search_query,
                'key': api_key,
                'cx': os.getenv('GOOGLE_CSE_ID'),  # Custom Search Engine ID
                'num': 5  # Get top 5 results
            }
            
            logger.info(f"Searching for documents: {search_query}")
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise ConnectionError(f"API request failed with status {response.status_code}")
                
            results = response.json()
            
            if 'items' in results:
                for item in results['items']:
                    link = item.get('link', '')
                    if doc_type.lower() in link.lower():
                        logger.info(f"Found document: {link}")
                        downloaded = self._secure_download(link)
                        if downloaded:
                            return downloaded
                            
            logger.warning(f"No documents found for: {search_query}")
            return None
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return None

    def get_satellite_data(self, location: str, layer: str = 'TRUE-COLOR', 
                          time_range: str = 'latest') -> Dict:
        """Advanced Earth observation data retrieval"""
        try:
            if not self.satellite_available:
                return {"error": "Satellite services not configured"}
                
            # Geocode location
            coords = self._geocode_location(location)
            if not coords:
                return {"error": "Location not found or geocoding failed"}
                
            bbox = BBox(bbox=coords, crs=CRS.WGS84)
            
            # Get multiple data layers with error handling
            requests_config = {
                'visual': {
                    'layer': layer,
                    'bbox': bbox,
                    'time': time_range,
                    'width': 1024,
                    'config': self.sh_config
                },
                'vegetation': {
                    'layer': 'NDVI',
                    'bbox': bbox,
                    'time': time_range,
                    'width': 1024,
                    'config': self.sh_config
                },
                'thermal': {
                    'layer': 'THERMAL',
                    'bbox': bbox,
                    'time': time_range,
                    'width': 1024,
                    'config': self.sh_config
                }
            }
            
            results = {}
            for key, config in requests_config.items():
                try:
                    wms_request = WmsRequest(**config)
                    results[key] = {
                        'data': wms_request.get_data(),
                        'metadata': {
                            'bbox': coords,
                            'time': time_range,
                            'location': location
                        }
                    }
                except Exception as e:
                    logger.error(f"Failed to get {key} data: {str(e)}")
                    results[key] = {'error': str(e)}
            
            # Save results to disk with timestamp
            timestamp = int(time.time())
            result_file = self.download_dir / f"satellite_{location.replace(' ', '_')}_{timestamp}.json"
            
            with open(result_file, 'w') as f:
                json.dump({
                    'location': location,
                    'timestamp': timestamp,
                    'available_layers': list(results.keys())
                }, f)
                
            return results
            
        except Exception as e:
            logger.error(f"Satellite data error: {str(e)}")
            return {"error": str(e)}

    def _geocode_location(self, location: str) -> Optional[List[float]]:
        """Convert location name to coordinates with improved error handling"""
        try:
            # Try with multiple geocoding services for redundancy
            services = [
                f"https://geocode.maps.co/search?q={location}",
                f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
            ]
            
            for service_url in services:
                try:
                    headers = {'User-Agent': 'AURA-Research-Bot/1.0'}
                    response = requests.get(service_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            # Extract coordinates
                            lon = float(data[0].get('lon', data[0].get('longitude', 0)))
                            lat = float(data[0].get('lat', data[0].get('latitude', 0)))
                            
                            # Validate coordinates
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                # Create bounding box (0.1 degree buffer)
                                return [
                                    lon - 0.1,
                                    lat - 0.1,
                                    lon + 0.1,
                                    lat + 0.1
                                ]
                except Exception as service_error:
                    logger.warning(f"Geocoding service error: {service_error}")
                    continue
                    
            logger.error(f"All geocoding services failed for location: {location}")
            return None
            
        except Exception as e:
            logger.error(f"Geocoding failed: {str(e)}")
            return None

    def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform a general web search and return structured results"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found in environment")
                
            search_url = f"https://www.googleapis.com/customsearch/v1"
            
            params = {
                'q': query,
                'key': api_key,
                'cx': os.getenv('GOOGLE_CSE_ID'),
                'num': max_results
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise ConnectionError(f"API request failed with status {response.status_code}")
                
            results = response.json()
            
            if 'items' not in results:
                return []
                
            structured_results = []
            for item in results['items']:
                structured_results.append({
                    'title': item.get('title', 'No title'),
                    'link': item.get('link', 'No link'),
                    'snippet': item.get('snippet', 'No description')
                })
                
            return structured_results
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []

    def handle_voice_command(self, text: str) -> Dict:
        """Process voice commands for web services with enhanced feedback"""
        response = {
            'success': False,
            'message': '',
            'data': None
        }
        
        try:
            # Document download commands
            if match := re.search(self.command_patterns['download_doc'], text, re.IGNORECASE):
                action, keywords, doc_type = match.groups()
                logger.info(f"Document request: {action} {keywords} {doc_type}")
                
                result = self.find_document(doc_type, keywords)
                if result:
                    response.update({
                        'success': True,
                        'message': f"Downloaded {keywords} {doc_type} document",
                        'data': {'path': str(result)}
                    })
                    if self.voice:
                        self.voice.speak(f"Downloaded {keywords} {doc_type} to your documents")
                else:
                    response.update({
                        'success': False,
                        'message': f"Could not find {keywords} {doc_type} document"
                    })
                    if self.voice:
                        self.voice.speak("Could not find the requested document")
                    
            # Satellite data commands
            elif match := re.search(self.command_patterns['satellite_data'], text, re.IGNORECASE):
                data_type, location = match.groups()
                logger.info(f"Satellite request: {data_type} for {location}")
                
                satellite_data = self.get_satellite_data(location)
                if 'error' not in satellite_data:
                    response.update({
                        'success': True,
                        'message': f"Retrieved {data_type} data for {location}",
                        'data': {'satellite_data': satellite_data}
                    })
                    if self.voice:
                        self.voice.speak(f"Showing {data_type} data for {location}")
                else:
                    response.update({
                        'success': False,
                        'message': f"Failed to retrieve {data_type} data for {location}: {satellite_data['error']}"
                    })
                    if self.voice:
                        self.voice.speak("Could not retrieve satellite data")
            
            # Web search commands
            elif match := re.search(self.command_patterns['web_search'], text, re.IGNORECASE):
                action, query = match.groups()
                logger.info(f"Web search request: {action} {query}")
                
                search_results = self.web_search(query)
                if search_results:
                    response.update({
                        'success': True,
                        'message': f"Found {len(search_results)} results for {query}",
                        'data': {'search_results': search_results}
                    })
                    if self.voice:
                        top_result = search_results[0]
                        self.voice.speak(f"Top result for {query}: {top_result['title']}. {top_result['snippet']}")
                else:
                    response.update({
                        'success': False,
                        'message': f"No search results found for {query}"
                    })
                    if self.voice:
                        self.voice.speak(f"I couldn't find any information about {query}")
            
            else:
                response.update({
                    'success': False,
                    'message': "Command not recognized"
                })
                if self.voice:
                    self.voice.speak("I didn't understand that web command")
                    
        except Exception as e:
            error_msg = f"Voice command handling failed: {str(e)}"
            logger.error(error_msg)
            response.update({
                'success': False,
                'message': error_msg
            })
            if self.voice:
                self.voice.speak("Error processing web request")
        
        return response

    def cleanup(self):
        """Secure cleanup of web resources with proper error handling"""
        try:
            # Close browser
            if hasattr(self, 'driver'):
                self.driver.quit()
                
            # Clean up temporary directory
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
            logger.info("Web controller shutdown complete")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup happens"""
        try:
            self.cleanup()
        except:
            pass


