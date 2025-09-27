from neo4j import GraphDatabase
import json
import logging
from datetime import datetime
import hashlib
from cryptography.fernet import Fernet

class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None, encryption_key=None):
        """Initialize enhanced knowledge graph with security features
        
        Args:
            uri: Neo4j connection URI (default: localhost)
            user: Neo4j username (default: neo4j)
            password: Neo4j password
            encryption_key: Optional encryption key for sensitive data
        """
        # Set up logging
        self.logger = logging.getLogger("aura.knowledge")
        
        # Use provided credentials or defaults
        self.uri = uri or "bolt://localhost:7687"
        self.user = user or "neo4j"
        self.password = password or "password"
        
        # Initialize connection
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                encrypted=True
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("Knowledge graph connected successfully")
        except Exception as e:
            self.logger.error(f"Knowledge graph connection error: {str(e)}")
            raise
            
        # Set up encryption for sensitive data
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        else:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for performance optimization"""
        try:
            with self.driver.session() as session:
                # Create index on text property for fast lookups
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Interaction) ON (n.text)")
                # Create index on timestamp for temporal queries
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Interaction) ON (n.timestamp)")
                # Create index on emotion for emotion-based queries
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Interaction) ON (n.emotion)")
        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {str(e)}")
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data before storage"""
        try:
            return self.cipher_suite.encrypt(data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data after retrieval"""
        try:
            return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            return encrypted_data

    def store_context(self, data):
        """Store interaction context in Neo4j with enhanced metadata
        
        Args:
            data: Dictionary containing interaction data
                Required keys: 'text', 'emotion'
                Optional keys: 'user', 'context', 'intent', 'confidence'
        """
        try:
            # Generate unique ID for the interaction
            interaction_id = hashlib.sha256(f"{data['text']}:{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Handle sensitive data if present
            if 'sensitive' in data and data['sensitive']:
                data['text'] = self.encrypt_sensitive_data(data['text'])
            
            with self.driver.session() as session:
                session.execute_write(
                    self._create_enhanced_node,
                    interaction_id,
                    data
                )
            self.logger.info(f"Stored interaction: {interaction_id[:8]}...")
            return interaction_id
        except Exception as e:
            self.logger.error(f"Failed to store context: {str(e)}")
            return None

    @staticmethod
    def _create_enhanced_node(tx, interaction_id, data):
        """Create node with all available properties from data"""
        # Base properties that must exist
        properties = {
            "id": interaction_id,
            "text": data["text"],
            "emotion": data["emotion"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add optional properties if they exist
        for key in ["user", "context", "intent", "confidence", "sentiment", "language"]:
            if key in data:
                properties[key] = data[key]
        
        # Create parameterized query dynamically based on available properties
        property_strings = [f"{k}: ${k}" for k in properties.keys()]
        query = f"CREATE (a:Interaction {{{', '.join(property_strings)}}})"
        
        # Execute query with all parameters
        tx.run(query, **properties)
    
    def query_by_emotion(self, emotion):
        """Query interactions by emotion"""
        with self.driver.session() as session:
            result = session.read_transaction(
                self._match_by_emotion,
                emotion
            )
            return result
    
    @staticmethod
    def _match_by_emotion(tx, emotion):
        query = (
            "MATCH (a:Interaction) "
            "WHERE a.emotion = $emotion "
            "RETURN a "
            "ORDER BY a.timestamp DESC "
            "LIMIT 10"
        )
        records = tx.run(query, emotion=emotion)
        return [record["a"] for record in records]
    
    def create_relationship(self, source_id, target_id, relationship_type, properties=None):
        """Create relationship between two interaction nodes"""
        if properties is None:
            properties = {}
            
        with self.driver.session() as session:
            session.write_transaction(
                self._create_relationship,
                source_id,
                target_id,
                relationship_type,
                properties
            )
    
    @staticmethod
    def _create_relationship(tx, source_id, target_id, relationship_type, properties):
        # Create parameterized query for relationship properties
        property_strings = [f"{k}: ${k}" for k in properties.keys()]
        property_clause = ""
        if property_strings:
            property_clause = f" {{{', '.join(property_strings)}}}"
            
        query = (
            f"MATCH (a:Interaction), (b:Interaction) "
            f"WHERE a.id = $source_id AND b.id = $target_id "
            f"CREATE (a)-[r:{relationship_type}{property_clause}]->(b) "
            f"RETURN r"
        )
        
        # Execute with all parameters
        params = {"source_id": source_id, "target_id": target_id}
        params.update(properties)
        return tx.run(query, **params)
        
    def get_related_contexts(self, text, limit=5):
        """Find contextually related interactions using text similarity"""
        with self.driver.session() as session:
            result = session.read_transaction(
                self._find_similar_contexts,
                text,
                limit
            )
            return result
    
    @staticmethod
    def _find_similar_contexts(tx, text, limit):
        # This uses a simplified text similarity approach
        query = (
            "MATCH (a:Interaction) "
            "WHERE a.text CONTAINS $text_fragment "
            "RETURN a "
            "ORDER BY a.timestamp DESC "
            "LIMIT $limit"
        )
        # Extract keywords for matching (simplified)
        text_fragment = text.split()[:3]
        text_fragment = " ".join(text_fragment) if text_fragment else text
        
        records = tx.run(query, text_fragment=text_fragment, limit=limit)
        return [record["a"] for record in records]
    
    def refresh_knowledge_graph(self):
        """Update timestamps or metadata"""
        try:
            with self.driver.session() as session:
                session.run(
                    "MATCH (n:Interaction) "
                    "SET n.lastAccessed = $time",
                    time=datetime.now().isoformat()
                )
            self.logger.info("Knowledge graph refreshed")
            return True
        except Exception as e:
            self.logger.error(f"Knowledge graph refresh error: {str(e)}")
            return False
    
    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Knowledge graph connection closed")