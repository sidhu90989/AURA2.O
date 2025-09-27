from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
import hmac
import os
import base64
import json
import logging
import time
import uuid
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SecurityHandler")

class SecurityHandler:
    """Enhanced security handling for AURA AI Assistant"""
    
    def __init__(self, security_level="high", encryption_type="hybrid", key_path=None):
        """
        Initialize security systems with configurable parameters
        
        Args:
            security_level: low, medium, high (default)
            encryption_type: symmetric, asymmetric, hybrid (default)
            key_path: Path to store/load keys (optional)
        """
        self.security_level = security_level
        self.encryption_type = encryption_type
        self.key_path = key_path or os.path.join(os.path.expanduser("~"), ".aura", "security")
        
        # Create directories if they don't exist
        if not os.path.exists(self.key_path):
            os.makedirs(self.key_path, exist_ok=True)
        
        # Initialize key management
        self._initialize_encryption()
        
        # Session management
        self.active_sessions = {}
        self.access_log = []
        
        # Authentication settings
        self.auth_settings = {
            "failed_attempts_limit": 5,
            "lockout_duration_minutes": 15,
            "mfa_required": True,
            "biometric_matching_threshold": 0.85,
            "password_min_length": 12,
            "password_complexity": True,
            "session_timeout_minutes": 30
        }
        
        # Threat detection
        self.suspicious_activity = []
        self.last_security_scan = datetime.now()
        
        logger.info(f"Security handler initialized with {security_level} security level")
        
    def _initialize_encryption(self):
        """Set up encryption systems based on configured type"""
        # Base salt for key derivation
        self.salt = os.urandom(16)
        
        # Symmetric encryption (Fernet)
        key_file = os.path.join(self.key_path, "symmetric.key")
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.key)
        
        self.cipher = Fernet(self.key)
        
        # Asymmetric encryption if needed
        if self.encryption_type in ["asymmetric", "hybrid"]:
            private_key_file = os.path.join(self.key_path, "private.key")
            public_key_file = os.path.join(self.key_path, "public.key")
            
            if os.path.exists(private_key_file) and os.path.exists(public_key_file):
                # Load existing keys
                with open(private_key_file, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                
                with open(public_key_file, "rb") as f:
                    pem_data = f.read()
                    self.public_key = serialization.load_pem_public_key(
                        pem_data,
                        backend=default_backend())
            else:
                # Generate new keypair
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096,
                    backend=default_backend()
                )
                self.public_key = self.private_key.public_key()
                
                # Save keys
                with open(private_key_file, "wb") as f:
                    f.write(self.private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                
                with open(public_key_file, "wb") as f:
                    f.write(self.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ))
    
    def encrypt(self, data, recipient_id=None):
        """
        Encrypt data using the appropriate encryption method
        
        Args:
            data: String data to encrypt
            recipient_id: Optional recipient ID for personalized encryption
            
        Returns:
            Encrypted data as bytes
        """
        if not isinstance(data, str):
            data = json.dumps(data)
            
        # Add metadata for security tracking
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "encryption_type": self.encryption_type,
            "security_level": self.security_level
        }
        
        if recipient_id:
            metadata["recipient_id"] = recipient_id
            
        # Combine data with metadata
        payload = json.dumps({
            "data": data,
            "metadata": metadata
        })
        
        # Encrypt based on encryption type
        if self.encryption_type == "symmetric":
            return self.cipher.encrypt(payload.encode())
        elif self.encryption_type == "asymmetric":
            # Ensure the public key is RSA
            from cryptography.hazmat.primitives.asymmetric import rsa
            if not isinstance(self.public_key, rsa.RSAPublicKey):
                raise TypeError("Asymmetric encryption requires an RSA public key.")
            return self.public_key.encrypt(
                payload.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:  # hybrid encryption
            # Ensure the public key is RSA
            from cryptography.hazmat.primitives.asymmetric import rsa
            if not isinstance(self.public_key, rsa.RSAPublicKey):
                raise TypeError("Hybrid encryption requires an RSA public key.")
            # Generate a random session key for AES
            session_key = os.urandom(32)
            encrypted_session_key = self.public_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Use the session key to encrypt the data with AES
            iv = os.urandom(16)
            cipher = Cipher(
                algorithms.AES(session_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(payload.encode()) + encryptor.finalize()
            
            # Combine everything into a package
            return base64.b64encode(json.dumps({
                "encrypted_session_key": base64.b64encode(encrypted_session_key).decode(),
                "iv": base64.b64encode(iv).decode(),
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "tag": base64.b64encode(encryptor.tag).decode()
            }).encode())
            
    def decrypt(self, encrypted_data, sender_id=None):
        """
        Decrypt data using the appropriate method
        
        Args:
            encrypted_data: Encrypted data bytes
            sender_id: Optional sender ID for verification
            
        Returns:
            Decrypted data as string
        """
        try:
            # Log decryption attempt for security auditing
            self._log_security_event("decrypt_attempt", {"sender_id": sender_id})
            
            if self.encryption_type == "symmetric":
                decrypted = self.cipher.decrypt(encrypted_data)
                payload = json.loads(decrypted.decode())
            elif self.encryption_type == "asymmetric":
                decrypted = self.private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                payload = json.loads(decrypted.decode())
            else:  # hybrid
                # Parse the package
                package = json.loads(base64.b64decode(encrypted_data).decode())
                encrypted_session_key = base64.b64decode(package["encrypted_session_key"])
                iv = base64.b64decode(package["iv"])
                encrypted_data = base64.b64decode(package["encrypted_data"])
                tag = base64.b64decode(package["tag"])
                
                # Decrypt the session key
                session_key = self.private_key.decrypt(
                    encrypted_session_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Use the session key to decrypt the data
                cipher = Cipher(
                    algorithms.AES(session_key),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                decryptor = cipher.decryptor()
                decrypted = decryptor.update(encrypted_data) + decryptor.finalize()
                payload = json.loads(decrypted.decode())
            
            # Verify sender if specified
            if sender_id and payload["metadata"].get("recipient_id") != sender_id:
                self._log_security_event("decrypt_mismatch", {
                    "expected": sender_id,
                    "found": payload["metadata"].get("recipient_id")
                })
                logger.warning(f"Sender ID mismatch in decryption")
                
            return payload["data"]
        except Exception as e:
            self._log_security_event("decrypt_error", {"error": str(e)})
            logger.error(f"Decryption failed: {e}")
            return None
    
    def hash_biometric(self, biometric_data, salt=None):
        """
        Create secure hash of biometric data with improved security
        
        Args:
            biometric_data: Biometric vector/bytes to hash
            salt: Optional salt to use
            
        Returns:
            Secure hash of the biometric data
        """
        if salt is None:
            salt = os.urandom(16)
            
        # Convert array to bytes if needed
        if hasattr(biometric_data, 'tobytes'):
            bio_bytes = biometric_data.tobytes()
        else:
            bio_bytes = bytes(biometric_data)
            
        # Create a stronger hash with key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(bio_bytes)
        
        # Final hash for storage
        hash_obj = hashlib.sha3_512()
        hash_obj.update(key)
        
        return {
            "hash": hash_obj.hexdigest(),
            "salt": base64.b64encode(salt).decode()
        }
    
    def verify_biometric(self, stored_hash_data, new_biometric_data):
        """
        Verify biometric data against stored hash
        
        Args:
            stored_hash_data: Dictionary with hash and salt values
            new_biometric_data: New biometric data to verify
            
        Returns:
            Boolean indicating match status
        """
        # Extract values from stored data
        stored_hash = stored_hash_data["hash"]
        salt = base64.b64decode(stored_hash_data["salt"])
        
        # Generate hash for comparison
        verification_result = self.hash_biometric(new_biometric_data, salt)
        
        # Secure comparison (constant time)
        return hmac.compare_digest(stored_hash, verification_result["hash"])
    
    def create_session(self, user_id, auth_level="standard", expiry_minutes=None):
        """
        Create a new authenticated session
        
        Args:
            user_id: User identifier
            auth_level: Authentication level (standard, elevated, admin)
            expiry_minutes: Session validity period
            
        Returns:
            Session token
        """
        # Generate session token
        session_id = str(uuid.uuid4())
        
        # Set expiry time
        if expiry_minutes is None:
            expiry_minutes = self.auth_settings["session_timeout_minutes"]
            
        expiry = datetime.now() + timedelta(minutes=expiry_minutes)
        
        # Create session data
        session = {
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "expires": expiry.isoformat(),
            "auth_level": auth_level,
            "ip_address": None,  # Would be set from request in real implementation
            "device_info": None  # Would be set from request in real implementation
        }
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Log session creation
        self._log_security_event("session_created", {
            "session_id": session_id,
            "user_id": user_id,
            "auth_level": auth_level
        })
        
        return session_id
        
    def validate_session(self, session_id, required_auth_level="standard"):
        """
        Validate a session token
        
        Args:
            session_id: Session identifier token
            required_auth_level: Minimum authentication level required
            
        Returns:
            User ID if valid, None otherwise
        """
        # Check if session exists
        if session_id not in self.active_sessions:
            logger.warning(f"Invalid session ID: {session_id}")
            return None
            
        session = self.active_sessions[session_id]
        
        # Check expiry
        if datetime.now() > datetime.fromisoformat(session["expires"]):
            # Session expired
            self._log_security_event("session_expired", {"session_id": session_id})
            del self.active_sessions[session_id]
            return None
            
        # Check auth level
        auth_levels = {"standard": 1, "elevated": 2, "admin": 3}
        if auth_levels.get(session["auth_level"], 0) < auth_levels.get(required_auth_level, 1):
            self._log_security_event("insufficient_auth", {
                "session_id": session_id, 
                "provided": session["auth_level"],
                "required": required_auth_level
            })
            return None
            
        # Session valid, extend expiry
        session["expires"] = (datetime.now() + timedelta(minutes=self.auth_settings["session_timeout_minutes"])).isoformat()
        
        return session["user_id"]
    
    def _log_security_event(self, event_type, details=None):
        """Log security events for auditing"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details or {}
        }
        self.access_log.append(log_entry)
        
        # Keep log at reasonable size
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]
        
        # Check for suspicious patterns
        if event_type in ["decrypt_error", "decrypt_mismatch", "insufficient_auth"]:
            self.suspicious_activity.append(log_entry)
            
    def check_password_strength(self, password):
        """
        Verify password meets security requirements
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if len(password) < self.auth_settings["password_min_length"]:
            return False, f"Password must be at least {self.auth_settings['password_min_length']} characters"
            
        if self.auth_settings["password_complexity"]:
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(not c.isalnum() for c in password)
            
            if not (has_upper and has_lower and has_digit and has_special):
                return False, "Password must include uppercase, lowercase, digits, and special characters"
                
        return True, "Password meets requirements"
        
    def hash_password(self, password, salt=None):
        """Create secure password hash"""
        if salt is None:
            salt = os.urandom(16)
            
        # Use key derivation for secure password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        
        return {
            "hash": base64.b64encode(key).decode(),
            "salt": base64.b64encode(salt).decode()
        }
        
    def verify_password(self, stored_password_data, provided_password):
        """Verify a password against stored hash"""
        salt = base64.b64decode(stored_password_data["salt"])
        
        # Hash the provided password with the same salt
        provided_hash = self.hash_password(provided_password, salt)
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_password_data["hash"], provided_hash["hash"])
        
    def encrypt_file(self, file_path, output_path=None):
        """Encrypt a file"""
        if output_path is None:
            output_path = file_path + ".encrypted"
            
        with open(file_path, "rb") as f:
            file_data = f.read()
            
        encrypted_data = self.encrypt(file_data)
        
        with open(output_path, "wb") as f:
            f.write(encrypted_data)
            
        return output_path
        
    def decrypt_file(self, file_path, output_path=None):
        """Decrypt a file"""
        if output_path is None:
            if file_path.endswith(".encrypted"):
                output_path = file_path[:-10]
            else:
                output_path = file_path + ".decrypted"
                
        with open(file_path, "rb") as f:
            encrypted_data = f.read()
            
        decrypted_data = self.decrypt(encrypted_data)
        
        with open(output_path, "wb") as f:
            f.write(decrypted_data)
            
        return output_path
        
    def generate_api_key(self, user_id, permissions=None, expiry_days=30):
        """Generate an API key for external service usage"""
        if permissions is None:
            permissions = ["read"]
            
        # Create a structured API key with metadata
        api_key_data = {
            "key_id": str(uuid.uuid4()),
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "expires": (datetime.now() + timedelta(days=expiry_days)).isoformat(),
            "permissions": permissions
        }
        
        # Generate random component
        random_component = base64.b64encode(os.urandom(24)).decode()
        
        # Combine structured and random data
        api_key = f"AURA_{api_key_data['key_id']}_{random_component}"
        
        # Store API key data (hash only)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        api_key_data["hash"] = key_hash
        
        # In a real implementation, store this in a database
        # For now, we'll just log it
        self._log_security_event("api_key_generated", {
            "key_id": api_key_data["key_id"],
            "user_id": user_id
        })
        
        return api_key, api_key_data
        
    def get_security_audit(self, days=7):
        """Get security audit information"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Filter events within the time range
        recent_events = [
            event for event in self.access_log 
            if datetime.fromisoformat(event["timestamp"]) > cutoff
        ]
        
        # Count event types
        event_counts = {}
        for event in recent_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        # Get suspicious activity
        recent_suspicious = [
            event for event in self.suspicious_activity
            if datetime.fromisoformat(event["timestamp"]) > cutoff
        ]
        
        return {
            "period_days": days,
            "total_events": len(recent_events),
            "event_breakdown": event_counts,
            "suspicious_activity_count": len(recent_suspicious),
            "active_sessions": len(self.active_sessions)
        }