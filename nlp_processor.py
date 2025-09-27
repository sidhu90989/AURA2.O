
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import Counter, deque
from functools import wraps
import random
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
import openai
import anthropic
from dotenv import load_dotenv
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel
)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob
import requests
from cachetools import TTLCache
import threading
import torch


class AIProvider(Enum):
    """Enumeration of supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    GOOGLE = "google"


@dataclass
class AnalysisMetrics:
    """Enhanced metrics for analysis quality tracking"""
    confidence_score: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    api_provider: str = ""
    tokens_used: int = 0
    cost_estimate: float = 0.0


@dataclass
class ConversationContext:
    """Enhanced conversation context with threading support"""
    user_id: str
    session_id: str = ""
    turn_count: int = 0
    last_interaction: datetime = None
    conversation_thread: List[Dict[str, Any]] = None
    user_profile: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_thread is None:
            self.conversation_thread = []
        if self.user_profile is None:
            self.user_profile = {}
        if self.last_interaction is None:
            self.last_interaction = datetime.now()


class NLPAnalysisResult:
    """Enhanced container for NLP analysis results with comprehensive metrics"""
    
    def __init__(self, text: str, timestamp: str = None, **kwargs):
        self.text = text
        self.timestamp = timestamp or datetime.now().isoformat()
        self.sentiment = kwargs.get('sentiment', {'label': 'NEUTRAL', 'score': 0.5})
        self.entities = kwargs.get('entities', [])
        self.emotion = kwargs.get('emotion', 'neutral')
        self.intent = kwargs.get('intent', 'unknown')
        self.topics = kwargs.get('topics', [])
        self.language = kwargs.get('language', 'en')
        self.toxicity_score = kwargs.get('toxicity_score', 0.0)
        self.readability_score = kwargs.get('readability_score', 0.0)
        self.keywords = kwargs.get('keywords', [])
        self.semantic_embedding = kwargs.get('semantic_embedding', None)
        self.additional_context = kwargs.get('additional_context', {})
        self.metrics = kwargs.get('metrics', AnalysisMetrics())
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize analysis results to dictionary"""
        result = {
            'text': self.text,
            'timestamp': self.timestamp,
            'sentiment': self.sentiment,
            'entities': self.entities,
            'emotion': self.emotion,
            'intent': self.intent,
            'topics': self.topics,
            'language': self.language,
            'toxicity_score': self.toxicity_score,
            'readability_score': self.readability_score,
            'keywords': self.keywords,
            'metrics': asdict(self.metrics) if isinstance(self.metrics, AnalysisMetrics) else self.metrics,
            **self.additional_context
        }
        if self.semantic_embedding is not None:
            result['semantic_embedding'] = self.semantic_embedding.tolist() if hasattr(self.semantic_embedding, 'tolist') else self.semantic_embedding
        return result


def async_handle_errors(max_retries: int = 3, backoff_factor: float = 1.5):
    """Enhanced async decorator for error handling with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    self.logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        await asyncio.sleep(wait_time)
                    
            self.logger.error(f"Operation failed after {max_retries} attempts")
            if kwargs.get('raise_error', True):
                raise last_exception
            return None
        return wrapper
    return decorator


def handle_errors(max_retries: int = 3, backoff_factor: float = 1.5):
    """Enhanced synchronous decorator for error handling with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    self.logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        time.sleep(wait_time)
                    
            self.logger.error(f"Operation failed after {max_retries} attempts")
            if kwargs.get('raise_error', True):
                raise last_exception
            return None
        return wrapper
    return decorator


class EnhancedNLPEngine:
    """
    Next-generation NLP Engine with multi-AI provider support, 
    advanced analytics, and enterprise-grade features.
    """
    
    # Enhanced emotion mapping with intensity levels
    EMOTION_MAP = {
        "POSITIVE": {
            "very_high": "euphoria",
            "high": "joy",
            "medium": "contentment",
            "low": "mild_satisfaction"
        },
        "NEGATIVE": {
            "very_high": "rage",
            "high": "anger",
            "medium": "sadness",
            "low": "disappointment"
        },
        "NEUTRAL": {
            "default": "neutral"
        },
        "default": "neutral"
    }
    
    # Sentiment score thresholds
    SENTIMENT_THRESHOLDS = {
        "very_high": 0.9,
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
    
    # Cost estimation per 1K tokens (USD)
    COST_ESTIMATES = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }
    
    def __init__(self, 
                 api_keys: Optional[Dict[str, Union[str, List[str]]]] = None, 
                 config: Dict[str, Any] = None,
                 enable_async: bool = True):
        """
        Initialize the Enhanced NLP Engine with multi-provider support.
        
        Args:
            api_keys: Dictionary of API keys by provider
            config: Configuration settings
            enable_async: Enable asynchronous processing
        """
        # Setup logging with structured format
        self._setup_enhanced_logging()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize API providers
        self._initialize_api_providers(api_keys or {})
        
        # Enhanced configuration
        self.config = self._get_enhanced_config()
        if config:
            self.config.update(config)
        
        # Initialize caching system
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1-hour TTL
        
        # Initialize enhanced NLP components
        self.components = self._initialize_enhanced_nlp_components()
        
        # Initialize advanced memory systems
        self.context_memory = self._initialize_enhanced_memory_systems()
        
        # Initialize rate limiting
        self.rate_limits = self._initialize_rate_limiting()
        
        # Enable async processing
        self.enable_async = enable_async
        if enable_async:
            self.loop = asyncio.new_event_loop()
            self.executor = None
        
        # Performance metrics
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0,
            "total_cost_estimate": 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("Enhanced NLPEngine initialized successfully")
        
    def _setup_enhanced_logging(self):
        """Configure enhanced structured logging with performance tracking"""
        self.logger = logging.getLogger('EnhancedNLPEngine')
        if not self.logger.handlers:
            # Create custom formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler for persistent logging
            try:
                file_handler = logging.FileHandler('nlp_engine.log')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not create file handler: {e}")
            
            self.logger.setLevel(logging.INFO)
        
    def _initialize_api_providers(self, api_keys: Dict[str, Union[str, List[str]]]) -> None:
        """Initialize multiple AI providers with failover support."""
        self.providers = {}
        
        # OpenAI initialization
        openai_keys = api_keys.get('openai', []) or [os.getenv("OPENAI_API_KEY")]
        openai_keys = [key for key in openai_keys if key]
        if openai_keys:
            self.providers[AIProvider.OPENAI] = {
                'client': openai,
                'keys': openai_keys,
                'current_key_index': 0,
                'models': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'default_model': 'gpt-4-turbo'
            }
            openai.api_key = openai_keys[0]
            self.logger.info(f"OpenAI initialized with {len(openai_keys)} key(s)")
        
        # Anthropic (Claude) initialization
        anthropic_keys = api_keys.get('anthropic', []) or [os.getenv("ANTHROPIC_API_KEY")]
        anthropic_keys = [key for key in anthropic_keys if key]
        if anthropic_keys:
            self.providers[AIProvider.ANTHROPIC] = {
                'client': anthropic.Anthropic(api_key=anthropic_keys[0]),
                'keys': anthropic_keys,
                'current_key_index': 0,
                'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                'default_model': 'claude-3-sonnet-20240229'
            }
            self.logger.info(f"Anthropic initialized with {len(anthropic_keys)} key(s)")
        
        # Set primary provider based on availability and preference
        if AIProvider.ANTHROPIC in self.providers:
            self.primary_provider = AIProvider.ANTHROPIC
        elif AIProvider.OPENAI in self.providers:
            self.primary_provider = AIProvider.OPENAI
        else:
            self.primary_provider = None
            self.logger.warning("No API providers configured")
            
    def _get_enhanced_config(self) -> Dict[str, Any]:
        """Get enhanced default configuration settings."""
        return {
            "personality": "helpful, insightful, and emotionally intelligent",
            "primary_provider": "anthropic",
            "fallback_providers": ["openai", "huggingface"],
            "creativity_level": 0.7,
            "context_window": 10,
            "max_retries": 5,
            "enable_caching": True,
            "enable_semantic_search": True,
            "enable_toxicity_detection": True,
            "enable_intent_recognition": True,
            "enable_topic_modeling": True,
            "response_length_limit": 2000,
            "parallel_processing": True,
            "quality_threshold": 0.8,
            "cost_optimization": True,
            "privacy_mode": False
        }
            
    def _initialize_enhanced_nlp_components(self) -> Dict[str, Any]:
        """Initialize comprehensive NLP pipeline components."""
        components = {}
        
        try:
            # Advanced tokenizer
            components['tokenizer'] = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            # Semantic embedding model
            components['embeddings'] = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.logger.info("Semantic embeddings initialized")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            components['embeddings'] = None
            
        # Enhanced sentiment analysis with multiple models
        components['sentiment'] = self._initialize_multi_sentiment()
        
        # Named entity recognition with spaCy
        try:
            components['nlp'] = spacy.load("en_core_web_sm")
            self.logger.info("spaCy NLP pipeline initialized")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            try:
                components['ner'] = pipeline("ner", aggregation_strategy="simple")
                self.logger.info("Fallback HuggingFace NER initialized")
            except Exception as e2:
                self.logger.error(f"Failed to initialize any NER: {e2}")
                components['ner'] = None
                
        # Intent recognition
        try:
            components['intent'] = pipeline("zero-shot-classification")
            self.logger.info("Intent recognition initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize intent recognition: {e}")
            components['intent'] = None
            
        # Toxicity detection
        try:
            components['toxicity'] = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )
            self.logger.info("Toxicity detection initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize toxicity detection: {e}")
            components['toxicity'] = None
            
        # Text summarization
        try:
            components['summarizer'] = pipeline("summarization")
            self.logger.info("Summarization pipeline initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize summarization: {e}")
            components['summarizer'] = None
            
        # Topic modeling setup
        components['tfidf'] = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        return components
    
    def _initialize_multi_sentiment(self) -> List[Any]:
        """Initialize multiple sentiment analysis models for ensemble prediction."""
        sentiment_models = []
        
        models_to_try = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "distilbert-base-uncased-finetuned-sst-2-english",
            None  # Default pipeline
        ]
        
        for model_name in models_to_try:
            try:
                if model_name is None:
                    sentiment = pipeline("sentiment-analysis")
                else:
                    sentiment = pipeline("sentiment-analysis", model=model_name)
                sentiment_models.append(sentiment)
                self.logger.info(f"Sentiment model loaded: {model_name or 'default'}")
            except Exception as e:
                self.logger.warning(f"Failed to load sentiment model {model_name}: {e}")
                
        return sentiment_models if sentiment_models else [None]
    
    def _initialize_enhanced_memory_systems(self) -> Dict[str, Any]:
        """Initialize advanced memory systems with persistence and retrieval."""
        return {
            "short_term": deque(maxlen=self.config["context_window"]),
            "long_term": {},  # Persistent storage for important information
            "user_profiles": {},  # Enhanced user profiling
            "conversation_contexts": {},  # Session-based context tracking
            "semantic_memory": [],  # Vector database for semantic search
            "knowledge_graph": {},  # Relationship mapping between entities
        }
    
    def _initialize_rate_limiting(self) -> Dict[str, Any]:
        """Initialize rate limiting for different providers."""
        return {
            AIProvider.OPENAI: {"requests_per_minute": 60, "last_reset": time.time(), "current_count": 0},
            AIProvider.ANTHROPIC: {"requests_per_minute": 50, "last_reset": time.time(), "current_count": 0},
        }
    
    async def analyze_async(self, 
                           text: str, 
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           additional_context: Optional[Dict[str, Any]] = None) -> NLPAnalysisResult:
        """
        Perform comprehensive asynchronous analysis with parallel processing.
        """
        start_time = time.time()
        
        # Create analysis result object
        analysis = NLPAnalysisResult(
            text=text,
            timestamp=datetime.now().isoformat(),
            additional_context=additional_context or {}
        )
        
        # Check cache first
        cache_key = f"analysis_{hash(text)}"
        if self.config["enable_caching"] and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            self.logger.info("Returning cached analysis result")
            return NLPAnalysisResult(**cached_result)
        
        # Parallel processing of analysis tasks
        tasks = []
        
        # Language detection
        tasks.append(self._detect_language_async(analysis))
        
        # Sentiment analysis (ensemble)
        tasks.append(self._add_ensemble_sentiment_async(analysis))
        
        # Entity recognition
        tasks.append(self._add_advanced_entity_recognition_async(analysis))
        
        # Intent recognition
        if self.config["enable_intent_recognition"]:
            tasks.append(self._add_intent_recognition_async(analysis))
        
        # Topic modeling
        if self.config["enable_topic_modeling"]:
            tasks.append(self._add_topic_modeling_async(analysis))
        
        # Toxicity detection
        if self.config["enable_toxicity_detection"]:
            tasks.append(self._add_toxicity_detection_async(analysis))
        
        # Semantic embedding
        if self.config["enable_semantic_search"]:
            tasks.append(self._add_semantic_embedding_async(analysis))
        
        # Execute all tasks in parallel
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process emotion based on sentiment
        self._process_enhanced_emotion(analysis)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        analysis.metrics = AnalysisMetrics(
            processing_time=processing_time,
            confidence_score=self._calculate_confidence_score(analysis)
        )
        
        # Update memory systems
        await self._update_enhanced_memory_async(analysis, user_id, session_id)
        
        # Cache the result
        if self.config["enable_caching"]:
            self.cache[cache_key] = analysis.to_dict()
        
        return analysis
    
    def analyze(self, 
               text: str, 
               user_id: Optional[str] = None,
               session_id: Optional[str] = None,
               additional_context: Optional[Dict[str, Any]] = None) -> NLPAnalysisResult:
        """
        Synchronous wrapper for comprehensive analysis.
        """
        if self.enable_async:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, create a new thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                self.analyze_async(text, user_id, session_id, additional_context)
                            )
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(
                        self.analyze_async(text, user_id, session_id, additional_context)
                    )
            except Exception as e:
                self.logger.warning(f"Async analysis failed, falling back to sync: {e}")
        
        # Fallback to synchronous analysis
        return self._analyze_sync(text, user_id, session_id, additional_context)
    
    def _analyze_sync(self, text: str, user_id: Optional[str], 
                     session_id: Optional[str], additional_context: Optional[Dict[str, Any]]) -> NLPAnalysisResult:
        """Synchronous comprehensive analysis."""
        start_time = time.time()
        
        analysis = NLPAnalysisResult(
            text=text,
            timestamp=datetime.now().isoformat(),
            additional_context=additional_context or {}
        )
        
        # Check cache
        cache_key = f"analysis_{hash(text)}"
        if self.config["enable_caching"] and cache_key in self.cache:
            return NLPAnalysisResult(**self.cache[cache_key])
        
        # Sequential analysis
        self._detect_language_sync(analysis)
        self._add_ensemble_sentiment_sync(analysis)
        self._add_advanced_entity_recognition_sync(analysis)
        
        if self.config["enable_intent_recognition"]:
            self._add_intent_recognition_sync(analysis)
        
        if self.config["enable_topic_modeling"]:
            self._add_topic_modeling_sync(analysis)
        
        if self.config["enable_toxicity_detection"]:
            self._add_toxicity_detection_sync(analysis)
        
        if self.config["enable_semantic_search"]:
            self._add_semantic_embedding_sync(analysis)
        
        self._process_enhanced_emotion(analysis)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        analysis.metrics = AnalysisMetrics(
            processing_time=processing_time,
            confidence_score=self._calculate_confidence_score(analysis)
        )
        
        # Update memory
        self._update_enhanced_memory_sync(analysis, user_id, session_id)
        
        # Cache result
        if self.config["enable_caching"]:
            self.cache[cache_key] = analysis.to_dict()
        
        return analysis
    
    async def _detect_language_async(self, analysis: NLPAnalysisResult) -> None:
        """Asynchronous language detection."""
        try:
            from textblob import TextBlob
            blob = TextBlob(analysis.text)
            analysis.language = blob.detect_language()
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            analysis.language = 'en'  # Default to English
    
    def _detect_language_sync(self, analysis: NLPAnalysisResult) -> None:
        """Synchronous language detection."""
        try:
            from textblob import TextBlob
            blob = TextBlob(analysis.text)
            analysis.language = blob.detect_language()
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            analysis.language = 'en'
    
    async def _add_ensemble_sentiment_async(self, analysis: NLPAnalysisResult) -> None:
        """Asynchronous ensemble sentiment analysis."""
        sentiments = []
        
        for sentiment_model in self.components.get('sentiment', []):
            if sentiment_model:
                try:
                    result = sentiment_model(analysis.text)
                    sentiments.append(result[0])
                except Exception as e:
                    self.logger.warning(f"Sentiment model failed: {e}")
        
        if sentiments:
            # Ensemble prediction (average confidence scores)
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            # Take majority vote for label
            labels = [s['label'] for s in sentiments]
            majority_label = Counter(labels).most_common(1)[0][0]
            
            analysis.sentiment = {
                'label': majority_label,
                'score': avg_score,
                'ensemble_results': sentiments
            }
    
    def _add_ensemble_sentiment_sync(self, analysis: NLPAnalysisResult) -> None:
        """Synchronous ensemble sentiment analysis."""
        sentiments = []
        
        for sentiment_model in self.components.get('sentiment', []):
            if sentiment_model:
                try:
                    result = sentiment_model(analysis.text)
                    sentiments.append(result[0])
                except Exception as e:
                    self.logger.warning(f"Sentiment model failed: {e}")
        
        if sentiments:
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            labels = [s['label'] for s in sentiments]
            majority_label = Counter(labels).most_common(1)[0][0]
            
            analysis.sentiment = {
                'label': majority_label,
                'score': avg_score,
                'ensemble_results': sentiments
            }
    
    async def _add_advanced_entity_recognition_async(self, analysis: NLPAnalysisResult) -> None:
        """Advanced asynchronous entity recognition using spaCy."""
        nlp_component = self.components.get('nlp')
        if nlp_component:
            try:
                doc = nlp_component(analysis.text)
                entities = []
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': float(ent._.kb_id) if hasattr(ent._, 'kb_id') else 1.0
                    })
                analysis.entities = entities
            except Exception as e:
                self.logger.warning(f"Advanced NER failed: {e}")
                # Fallback to basic NER
                await self._fallback_ner_async(analysis)
        else:
            await self._fallback_ner_async(analysis)
    
    def _add_advanced_entity_recognition_sync(self, analysis: NLPAnalysisResult) -> None:
        """Advanced synchronous entity recognition using spaCy."""
        nlp_component = self.components.get('nlp')
        if nlp_component:
            try:
                doc = nlp_component(analysis.text)
                entities = []
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 1.0
                    })
                analysis.entities = entities
            except Exception as e:
                self.logger.warning(f"Advanced NER failed: {e}")
                self._fallback_ner_sync(analysis)
        else:
            self._fallback_ner_sync(analysis)
    
    async def _fallback_ner_async(self, analysis: NLPAnalysisResult) -> None:
        """Fallback NER using HuggingFace."""
        ner_component = self.components.get('ner')
        if ner_component:
            try:
                entities = ner_component(analysis.text)
                analysis.entities = entities
            except Exception as e:
                self.logger.warning(f"Fallback NER failed: {e}")
    
    def _fallback_ner_sync(self, analysis: NLPAnalysisResult) -> None:
        """Fallback NER using HuggingFace."""
        ner_component = self.components.get('ner')
        if ner_component:
            try:
                entities = ner_component(analysis.text)
                analysis.entities = entities
            except Exception as e:
                self.logger.warning(f"Fallback NER failed: {e}")
    
    async def _add_intent_recognition_async(self, analysis: NLPAnalysisResult) -> None:
        """Asynchronous intent recognition."""
        intent_component = self.components.get('intent')
        if intent_component:
            try:
                candidate_labels = [
                    'question', 'request', 'complaint', 'compliment', 
                    'information_seeking', 'task_completion', 'greeting', 'goodbye'
                ]
                result = intent_component(analysis.text, candidate_labels)
                analysis.intent = {
                    'intent': result['labels'][0],
                    'confidence': result['scores'][0],
                    'all_intents': list(zip(result['labels'], result['scores']))
                }
            except Exception as e:
                self.logger.warning(f"Intent recognition failed: {e}")
    
    async def _add_topic_modeling_async(self, analysis: NLPAnalysisResult) -> None:
        """Asynchronous topic modeling and keyword extraction."""
        try:
            # Extract keywords using TF-IDF
            tfidf = self.components.get('tfidf')
            if tfidf and hasattr(tfidf, 'vocabulary_'):
                # If already fitted, use transform
                tfidf_matrix = tfidf.transform([analysis.text])
            else:
                # Fit and transform (for single documents, we need a corpus)
                # Using a simple approach for keyword extraction
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf_single = TfidfVectorizer(max_features=10, stop_words='english')
                tfidf_matrix = tfidf_single.fit_transform([analysis.text])
                feature_names = tfidf_single.get_feature_names_out()
                
                # Get feature scores
                feature_scores = tfidf_matrix.toarray()[0]
                
                # Create keyword list with scores
                keywords = []
                for i, score in enumerate(feature_scores):
                    if score > 0:
                        keywords.append({
                            'keyword': feature_names[i],
                            'score': score
                        })
                
                # Sort by score and take top keywords
                keywords.sort(key=lambda x: x['score'], reverse=True)
                analysis.keywords = keywords[:5]
                analysis.topics = [kw['keyword'] for kw in keywords[:3]]
        except Exception as e:
            self.logger.warning(f"Topic modeling failed: {e}")
    
    def _add_topic_modeling_sync(self, analysis: NLPAnalysisResult) -> None:
        """Synchronous topic modeling and keyword extraction."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf_single = TfidfVectorizer(max_features=10, stop_words='english')
            tfidf_matrix = tfidf_single.fit_transform([analysis.text])
            feature_names = tfidf_single.get_feature_names_out()
            
            feature_scores = tfidf_matrix.toarray()[0]
            
            keywords = []
            for i, score in enumerate(feature_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': score
                    })
            
            keywords.sort(key=lambda x: x['score'], reverse=True)
            analysis.keywords = keywords[:5]
            analysis.topics = [kw['keyword'] for kw in keywords[:3]]
        except Exception as e:
            self.logger.warning(f"Topic modeling failed: {e}")
    
    async def _add_toxicity_detection_async(self, analysis: NLPAnalysisResult) -> None:
        """Asynchronous toxicity detection."""
        toxicity_component = self.components.get('toxicity')
        if toxicity_component:
            try:
                result = toxicity_component(analysis.text)
                # Handle different output formats
                if isinstance(result, list) and len(result) > 0:
                    toxic_score = next((r['score'] for r in result if r['label'] == 'TOXIC'), 0.0)
                    analysis.toxicity_score = toxic_score
                else:
                    analysis.toxicity_score = 0.0
            except Exception as e:
                self.logger.warning(f"Toxicity detection failed: {e}")
                analysis.toxicity_score = 0.0
    
    def _add_toxicity_detection_sync(self, analysis: NLPAnalysisResult) -> None:
        """Synchronous toxicity detection."""
        toxicity_component = self.components.get('toxicity')
        if toxicity_component:
            try:
                result = toxicity_component(analysis.text)
                if isinstance(result, list) and len(result) > 0:
                    toxic_score = next((r['score'] for r in result if r['label'] == 'TOXIC'), 0.0)
                    analysis.toxicity_score = toxic_score
                else:
                    analysis.toxicity_score = 0.0
            except Exception as e:
                self.logger.warning(f"Toxicity detection failed: {e}")
                analysis.toxicity_score = 0.0
    
    async def _add_semantic_embedding_async(self, analysis: NLPAnalysisResult) -> None:
        """Asynchronous semantic embedding generation."""
        embedding_model = self.components.get('embeddings')
        tokenizer = self.components.get('tokenizer')
        
        if embedding_model and tokenizer:
            try:
                # Tokenize and get embeddings
                inputs = tokenizer(analysis.text, return_tensors="pt", 
                                 truncation=True, padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = embedding_model(**inputs)
                    # Use the mean of last hidden states as sentence embedding
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                    analysis.semantic_embedding = embedding
            except Exception as e:
                self.logger.warning(f"Semantic embedding failed: {e}")
    
    def _add_semantic_embedding_sync(self, analysis: NLPAnalysisResult) -> None:
        """Synchronous semantic embedding generation."""
        try:
            import torch
            embedding_model = self.components.get('embeddings')
            tokenizer = self.components.get('tokenizer')
            
            if embedding_model and tokenizer:
                inputs = tokenizer(analysis.text, return_tensors="pt", 
                                 truncation=True, padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = embedding_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                    analysis.semantic_embedding = embedding
        except Exception as e:
            self.logger.warning(f"Semantic embedding failed: {e}")
    
    def _process_enhanced_emotion(self, analysis: NLPAnalysisResult) -> None:
        """Enhanced emotion processing with intensity levels."""
        sentiment_label = analysis.sentiment.get('label', 'NEUTRAL')
        sentiment_score = analysis.sentiment.get('score', 0.5)
        
        # Determine intensity level
        intensity = 'low'
        for threshold_name, threshold_value in self.SENTIMENT_THRESHOLDS.items():
            if sentiment_score >= threshold_value:
                intensity = threshold_name
                break
        
        # Map to emotion
        emotion_category = sentiment_label.upper()
        if emotion_category in self.EMOTION_MAP:
            emotion_mapping = self.EMOTION_MAP[emotion_category]
            analysis.emotion = emotion_mapping.get(intensity, emotion_mapping.get('default', 'neutral'))
        else:
            analysis.emotion = self.EMOTION_MAP['default']
    
    def _calculate_confidence_score(self, analysis: NLPAnalysisResult) -> float:
        """Calculate overall confidence score for the analysis."""
        scores = []
        
        # Sentiment confidence
        if analysis.sentiment and 'score' in analysis.sentiment:
            scores.append(analysis.sentiment['score'])
        
        # Intent confidence
        if isinstance(analysis.intent, dict) and 'confidence' in analysis.intent:
            scores.append(analysis.intent['confidence'])
        
        # Entity confidence (average)
        if analysis.entities:
            entity_scores = [e.get('confidence', 1.0) for e in analysis.entities]
            scores.append(sum(entity_scores) / len(entity_scores))
        
        # Return average confidence or default
        return sum(scores) / len(scores) if scores else 0.7
    
    async def _update_enhanced_memory_async(self, analysis: NLPAnalysisResult, 
                                           user_id: Optional[str], 
                                           session_id: Optional[str]) -> None:
        """Enhanced asynchronous memory update with semantic storage."""
        with self._lock:
            # Update short-term memory
            self.context_memory["short_term"].append(analysis.to_dict())
            
            # Update user-specific memory
            if user_id:
                await self._update_user_profile_async(user_id, analysis)
            
            # Update conversation context
            if session_id:
                await self._update_conversation_context_async(session_id, analysis, user_id)
            
            # Update semantic memory for future retrieval
            if analysis.semantic_embedding is not None:
                self.context_memory["semantic_memory"].append({
                    'text': analysis.text,
                    'embedding': analysis.semantic_embedding,
                    'timestamp': analysis.timestamp,
                    'user_id': user_id,
                    'session_id': session_id,
                    'metadata': analysis.to_dict()
                })
    
    def _update_enhanced_memory_sync(self, analysis: NLPAnalysisResult, 
                                    user_id: Optional[str], 
                                    session_id: Optional[str]) -> None:
        """Enhanced synchronous memory update."""
        with self._lock:
            self.context_memory["short_term"].append(analysis.to_dict())
            
            if user_id:
                self._update_user_profile_sync(user_id, analysis)
            
            if session_id:
                self._update_conversation_context_sync(session_id, analysis, user_id)
            
            if analysis.semantic_embedding is not None:
                self.context_memory["semantic_memory"].append({
                    'text': analysis.text,
                    'embedding': analysis.semantic_embedding,
                    'timestamp': analysis.timestamp,
                    'user_id': user_id,
                    'session_id': session_id,
                    'metadata': analysis.to_dict()
                })
    
    async def _update_user_profile_async(self, user_id: str, analysis: NLPAnalysisResult) -> None:
        """Enhanced asynchronous user profile updates."""
        if user_id not in self.context_memory["user_profiles"]:
            self.context_memory["user_profiles"][user_id] = {
                "created_at": datetime.now().isoformat(),
                "total_interactions": 0,
                "sentiment_history": [],
                "emotion_patterns": Counter(),
                "preferred_topics": Counter(),
                "language_preference": 'en',
                "interaction_times": [],
                "toxicity_flags": 0,
                "average_session_length": 0.0,
                "personality_traits": {},
                "conversation_contexts": []
            }
        
        profile = self.context_memory["user_profiles"][user_id]
        profile["total_interactions"] += 1
        profile["interaction_times"].append(datetime.now().isoformat())
        
        # Update sentiment history
        profile["sentiment_history"].append({
            'label': analysis.sentiment.get('label'),
            'score': analysis.sentiment.get('score'),
            'timestamp': analysis.timestamp
        })
        
        # Update emotion patterns
        profile["emotion_patterns"][analysis.emotion] += 1
        
        # Update preferred topics
        for topic in analysis.topics:
            profile["preferred_topics"][topic] += 1
        
        # Update language preference
        if analysis.language:
            profile["language_preference"] = analysis.language
        
        # Track toxicity
        if analysis.toxicity_score > 0.7:
            profile["toxicity_flags"] += 1
    
    def _update_user_profile_sync(self, user_id: str, analysis: NLPAnalysisResult) -> None:
        """Enhanced synchronous user profile updates."""
        if user_id not in self.context_memory["user_profiles"]:
            self.context_memory["user_profiles"][user_id] = {
                "created_at": datetime.now().isoformat(),
                "total_interactions": 0,
                "sentiment_history": [],
                "emotion_patterns": Counter(),
                "preferred_topics": Counter(),
                "language_preference": 'en',
                "interaction_times": [],
                "toxicity_flags": 0,
                "personality_traits": {},
            }
        
        profile = self.context_memory["user_profiles"][user_id]
        profile["total_interactions"] += 1
        profile["interaction_times"].append(datetime.now().isoformat())
        
        profile["sentiment_history"].append({
            'label': analysis.sentiment.get('label'),
            'score': analysis.sentiment.get('score'),
            'timestamp': analysis.timestamp
        })
        
        profile["emotion_patterns"][analysis.emotion] += 1
        
        for topic in analysis.topics:
            profile["preferred_topics"][topic] += 1
        
        if analysis.language:
            profile["language_preference"] = analysis.language
        
        if analysis.toxicity_score > 0.7:
            profile["toxicity_flags"] += 1
    
    async def _update_conversation_context_async(self, session_id: str, 
                                                analysis: NLPAnalysisResult, 
                                                user_id: Optional[str]) -> None:
        """Update conversation context for session tracking."""
        if session_id not in self.context_memory["conversation_contexts"]:
            self.context_memory["conversation_contexts"][session_id] = ConversationContext(
                user_id=user_id or "anonymous",
                session_id=session_id
            )
        
        context = self.context_memory["conversation_contexts"][session_id]
        context.turn_count += 1
        context.last_interaction = datetime.now()
        context.conversation_thread.append({
            'text': analysis.text,
            'analysis': analysis.to_dict(),
            'timestamp': analysis.timestamp
        })
    
    def _update_conversation_context_sync(self, session_id: str, 
                                         analysis: NLPAnalysisResult, 
                                         user_id: Optional[str]) -> None:
        """Update conversation context for session tracking."""
        if session_id not in self.context_memory["conversation_contexts"]:
            self.context_memory["conversation_contexts"][session_id] = ConversationContext(
                user_id=user_id or "anonymous",
                session_id=session_id
            )
        
        context = self.context_memory["conversation_contexts"][session_id]
        context.turn_count += 1
        context.last_interaction = datetime.now()
        context.conversation_thread.append({
            'text': analysis.text,
            'analysis': analysis.to_dict(),
            'timestamp': analysis.timestamp
        })
    
    @async_handle_errors()
    async def generate_response_async(self, 
                                     analysis_result: NLPAnalysisResult,
                                     user_id: Optional[str] = None,
                                     session_id: Optional[str] = None,
                                     provider: Optional[AIProvider] = None,
                                     model: Optional[str] = None) -> str:
        """
        Generate an enhanced AI response with multi-provider support and optimization.
        """
        start_time = time.time()
        
        # Select provider and model
        selected_provider = provider or self.primary_provider
        if not selected_provider or selected_provider not in self.providers:
            raise ValueError("No available AI provider configured")
        
        provider_config = self.providers[selected_provider]
        selected_model = model or provider_config['default_model']
        
        # Build enhanced prompt with full context
        prompt = await self._build_enhanced_prompt_async(analysis_result, user_id, session_id)
        
        # Check rate limits
        if not self._check_rate_limit(selected_provider):
            # Try fallback provider
            for fallback_provider in self.config.get('fallback_providers', []):
                fallback_enum = AIProvider(fallback_provider)
                if fallback_enum in self.providers and self._check_rate_limit(fallback_enum):
                    selected_provider = fallback_enum
                    provider_config = self.providers[selected_provider]
                    selected_model = provider_config['default_model']
                    break
            else:
                raise Exception("Rate limit exceeded for all providers")
        
        # Generate response based on provider
        response = await self._call_ai_provider_async(
            selected_provider, selected_model, prompt, analysis_result
        )
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_performance_metrics(
            selected_provider, selected_model, processing_time, 
            analysis_result, response, success=True
        )
        
        return response
    
    def generate_response(self, 
                         analysis_result: NLPAnalysisResult,
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         provider: Optional[AIProvider] = None,
                         model: Optional[str] = None) -> str:
        """
        Synchronous wrapper for enhanced response generation.
        """
        if self.enable_async:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                self.generate_response_async(
                                    analysis_result, user_id, session_id, provider, model
                                )
                            )
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(
                        self.generate_response_async(
                            analysis_result, user_id, session_id, provider, model
                        )
                    )
            except Exception as e:
                self.logger.warning(f"Async response generation failed, falling back to sync: {e}")
        
        # Fallback to synchronous generation
        return self._generate_response_sync(analysis_result, user_id, session_id, provider, model)
    
    def _generate_response_sync(self, analysis_result: NLPAnalysisResult,
                               user_id: Optional[str], session_id: Optional[str],
                               provider: Optional[AIProvider], model: Optional[str]) -> str:
        """Synchronous response generation."""
        start_time = time.time()
        
        selected_provider = provider or self.primary_provider
        if not selected_provider or selected_provider not in self.providers:
            raise ValueError("No available AI provider configured")
        
        provider_config = self.providers[selected_provider]
        selected_model = model or provider_config['default_model']
        
        prompt = self._build_enhanced_prompt_sync(analysis_result, user_id, session_id)
        
        if not self._check_rate_limit(selected_provider):
            for fallback_provider in self.config.get('fallback_providers', []):
                fallback_enum = AIProvider(fallback_provider)
                if fallback_enum in self.providers and self._check_rate_limit(fallback_enum):
                    selected_provider = fallback_enum
                    provider_config = self.providers[selected_provider]
                    selected_model = provider_config['default_model']
                    break
            else:
                raise Exception("Rate limit exceeded for all providers")
        
        response = self._call_ai_provider_sync(
            selected_provider, selected_model, prompt, analysis_result
        )
        
        processing_time = time.time() - start_time
        self._update_performance_metrics(
            selected_provider, selected_model, processing_time, 
            analysis_result, response, success=True
        )
        
        return response
    
    async def _build_enhanced_prompt_async(self, analysis: NLPAnalysisResult,
                                          user_id: Optional[str], 
                                          session_id: Optional[str]) -> Dict[str, str]:
        """Build comprehensive prompt with all available context."""
        system_components = []
        
        # Base personality and capabilities
        system_components.append(
            f"You are an advanced AI assistant with {self.config['personality']}. "
            "You have access to comprehensive analysis of the user's input including "
            "sentiment, emotions, intent, entities, and topics."
        )
        
        # Add emotional intelligence guidance
        emotion_guidance = self._get_enhanced_emotion_guidance(analysis.emotion, analysis.sentiment)
        if emotion_guidance:
            system_components.append(emotion_guidance)
        
        # Add user-specific context
        user_context = await self._get_enhanced_user_context_async(user_id)
        if user_context:
            system_components.append(user_context)
        
        # Add conversation context
        conversation_context = await self._get_conversation_context_async(session_id)
        if conversation_context:
            system_components.append(conversation_context)
        
        # Add analysis insights
        analysis_insights = self._generate_analysis_insights(analysis)
        if analysis_insights:
            system_components.append(analysis_insights)
        
        # Add response guidelines
        system_components.append(
            "Respond naturally, helpfully, and appropriately to the context. "
            "Maintain consistency with the conversation history and user preferences. "
            f"Keep responses under {self.config['response_length_limit']} characters."
        )
        
        return {
            "system": " ".join(system_components),
            "user": analysis.text,
            "context": analysis.to_dict()
        }
    
    def _build_enhanced_prompt_sync(self, analysis: NLPAnalysisResult,
                                   user_id: Optional[str], 
                                   session_id: Optional[str]) -> Dict[str, str]:
        """Synchronous enhanced prompt building."""
        system_components = []
        
        system_components.append(
            f"You are an advanced AI assistant with {self.config['personality']}. "
            "You have access to comprehensive analysis of the user's input."
        )
        
        emotion_guidance = self._get_enhanced_emotion_guidance(analysis.emotion, analysis.sentiment)
        if emotion_guidance:
            system_components.append(emotion_guidance)
        
        user_context = self._get_enhanced_user_context_sync(user_id)
        if user_context:
            system_components.append(user_context)
        
        conversation_context = self._get_conversation_context_sync(session_id)
        if conversation_context:
            system_components.append(conversation_context)
        
        analysis_insights = self._generate_analysis_insights(analysis)
        if analysis_insights:
            system_components.append(analysis_insights)
        
        system_components.append(
            "Respond naturally and helpfully. "
            f"Keep responses under {self.config['response_length_limit']} characters."
        )
        
        return {
            "system": " ".join(system_components),
            "user": analysis.text,
            "context": analysis.to_dict()
        }
    
    def _get_enhanced_emotion_guidance(self, emotion: str, sentiment: Dict[str, Any]) -> str:
        """Get enhanced emotional guidance for response generation."""
        base_guidance = {
            "anger": "The user seems angry. Respond with calm empathy and de-escalation.",
            "rage": "The user seems very angry. Prioritize de-escalation and safety.",
            "sadness": "The user seems sad. Offer compassionate support and understanding.",
            "grief": "The user may be grieving. Be gentle, supportive, and respectful.",
            "joy": "The user seems happy. Share in their positive energy appropriately.",
            "euphoria": "The user seems very excited. Be supportive but help ground expectations.",
            "confusion": "The user seems confused. Provide clear, step-by-step guidance.",
            "anxiety": "The user may be anxious. Be reassuring and offer practical help.",
            "fear": "The user seems worried. Acknowledge their concerns and provide comfort.",
            "neutral": "Maintain a balanced, helpful tone."
        }
        
        guidance = base_guidance.get(emotion.lower(), "")
        
        # Add sentiment-specific nuance
        sentiment_score = sentiment.get('score', 0.5)
        if sentiment_score > 0.8:
            guidance += " The user's sentiment is very strong, so acknowledge this intensity."
        elif sentiment_score < 0.3:
            guidance += " The user's sentiment is quite negative, so be extra supportive."
        
        return guidance
    
    async def _get_enhanced_user_context_async(self, user_id: Optional[str]) -> str:
        """Get comprehensive user context for personalization."""
        if not user_id or user_id not in self.context_memory["user_profiles"]:
            return ""
        
        profile = self.context_memory["user_profiles"][user_id]
        context_elements = []
        
        # Interaction history
        if profile["total_interactions"] > 10:
            context_elements.append(f"This is a returning user with {profile['total_interactions']} interactions.")
        
        # Preferred topics
        top_topics = profile["preferred_topics"].most_common(3)
        if top_topics:
            topics_str = ", ".join([topic for topic, _ in top_topics])
            context_elements.append(f"User frequently discusses: {topics_str}.")
        
        # Emotional patterns
        common_emotions = profile["emotion_patterns"].most_common(2)
        if common_emotions:
            emotions_str = ", ".join([emotion for emotion, _ in common_emotions])
            context_elements.append(f"User's typical emotional states: {emotions_str}.")
        
        # Language preference
        if profile["language_preference"] != 'en':
            context_elements.append(f"User's preferred language: {profile['language_preference']}.")
        
        # Sentiment trends
        if len(profile["sentiment_history"]) >= 5:
            recent_sentiments = profile["sentiment_history"][-5:]
            avg_sentiment = sum(s['score'] for s in recent_sentiments) / len(recent_sentiments)
            if avg_sentiment > 0.7:
                context_elements.append("User has been generally positive recently.")
            elif avg_sentiment < 0.4:
                context_elements.append("User has been experiencing negative sentiment recently.")
        
        return " ".join(context_elements)
    
    def _get_enhanced_user_context_sync(self, user_id: Optional[str]) -> str:
        """Synchronous enhanced user context."""
        if not user_id or user_id not in self.context_memory["user_profiles"]:
            return ""
        
        profile = self.context_memory["user_profiles"][user_id]
        context_elements = []
        
        if profile["total_interactions"] > 10:
            context_elements.append(f"Returning user with {profile['total_interactions']} interactions.")
        
        top_topics = profile["preferred_topics"].most_common(3)
        if top_topics:
            topics_str = ", ".join([topic for topic, _ in top_topics])
            context_elements.append(f"Frequently discusses: {topics_str}.")
        
        return " ".join(context_elements)
    
    async def _get_conversation_context_async(self, session_id: Optional[str]) -> str:
        """Get conversation context for session continuity."""
        if not session_id or session_id not in self.context_memory["conversation_contexts"]:
            return ""
        
        context = self.context_memory["conversation_contexts"][session_id]
        context_elements = []
        
        if context.turn_count > 1:
            context_elements.append(f"This is turn {context.turn_count} of the conversation.")
        
        # Recent conversation summary
        if len(context.conversation_thread) >= 2:
            recent_topics = []
            for turn in context.conversation_thread[-3:]:  # Last 3 turns
                analysis = turn.get('analysis', {})
                topics = analysis.get('topics', [])
                recent_topics.extend(topics)
            
            if recent_topics:
                unique_topics = list(set(recent_topics))[:3]
                context_elements.append(f"Recent conversation topics: {', '.join(unique_topics)}.")
        
        return " ".join(context_elements)
    
    def _get_conversation_context_sync(self, session_id: Optional[str]) -> str:
        """Synchronous conversation context."""
        if not session_id or session_id not in self.context_memory["conversation_contexts"]:
            return ""
        
        context = self.context_memory["conversation_contexts"][session_id]
        context_elements = []
        
        if context.turn_count > 1:
            context_elements.append(f"Turn {context.turn_count} of conversation.")
        
        return " ".join(context_elements)
    
    def _generate_analysis_insights(self, analysis: NLPAnalysisResult) -> str:
        """Generate insights from the analysis for context."""
        insights = []
        
        # Sentiment insight
        sentiment_label = analysis.sentiment.get('label', 'NEUTRAL')
        sentiment_score = analysis.sentiment.get('score', 0.5)
        insights.append(f"User sentiment: {sentiment_label} (confidence: {sentiment_score:.2f})")
        
        # Intent insight
        if isinstance(analysis.intent, dict):
            intent_info = analysis.intent.get('intent', 'unknown')
            intent_confidence = analysis.intent.get('confidence', 0.0)
            insights.append(f"Detected intent: {intent_info} (confidence: {intent_confidence:.2f})")
        
        # Entity insight
        if analysis.entities:
            entity_types = list(set([e.get('label', e.get('entity_group', 'MISC')) for e in analysis.entities]))
            insights.append(f"Key entities detected: {', '.join(entity_types[:3])}")
        
        # Topic insight
        if analysis.topics:
            insights.append(f"Main topics: {', '.join(analysis.topics[:3])}")
        
        # Toxicity warning
        if analysis.toxicity_score > 0.7:
            insights.append("WARNING: Potentially toxic content detected.")
        
        return "Analysis insights: " + "; ".join(insights) + "."
    
    def _check_rate_limit(self, provider: AIProvider) -> bool:
        """Check if provider is within rate limits."""
        if provider not in self.rate_limits:
            return True
        
        rate_limit_info = self.rate_limits[provider]
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - rate_limit_info["last_reset"] >= 60:
            rate_limit_info["current_count"] = 0
            rate_limit_info["last_reset"] = current_time
        
        # Check if under limit
        if rate_limit_info["current_count"] < rate_limit_info["requests_per_minute"]:
            rate_limit_info["current_count"] += 1
            return True
        
        return False
    
    async def _call_anthropic_async(self, model: str, prompt: Dict[str, str], 
                                   analysis: NLPAnalysisResult) -> str:
        """Call Anthropic Claude API asynchronously."""
        try:
            provider_config = self.providers[AIProvider.ANTHROPIC]
            client = provider_config['client']
            
            # Construct message for Claude
            system_message = prompt["system"]
            user_message = prompt["user"]
            
            # Add context if available
            if "context" in prompt:
                context_str = json.dumps(prompt["context"], indent=2)
                user_message += f"\n\nContext Analysis:\n{context_str}"
            
            # Create message
            message = await client.messages.create(
                model=model,
                max_tokens=min(4000, self.config["response_length_limit"] // 2),
                temperature=self.config["creativity_level"],
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            response = message.content[0].text
            
            # Calculate token usage and cost
            input_tokens = len(system_message.split()) + len(user_message.split())
            output_tokens = len(response.split())
            self._update_token_metrics(model, input_tokens, output_tokens)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            # Retry with different key if available
            return await self._retry_with_different_key_async(AIProvider.ANTHROPIC, model, prompt, analysis)
    
    async def _call_openai_async(self, model: str, prompt: Dict[str, str], 
                                analysis: NLPAnalysisResult) -> str:
        """Call OpenAI API asynchronously."""
        try:
            provider_config = self.providers[AIProvider.OPENAI]
            
            # Construct messages for OpenAI
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            # Add context
            if "context" in prompt:
                context_str = json.dumps(prompt["context"], indent=2)
                messages.append({"role": "user", "content": f"Analysis Context: {context_str}"})
            
            # Make API call
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                max_tokens=min(4000, self.config["response_length_limit"] // 2),
                temperature=self.config["creativity_level"]
            )
            
            response_text = response.choices[0].message.content
            
            # Update metrics
            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            self._update_token_metrics(model, input_tokens, output_tokens)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return await self._retry_with_different_key_async(AIProvider.OPENAI, model, prompt, analysis)
    
    def _call_ai_provider_sync(self, provider: AIProvider, model: str, 
                              prompt: Dict[str, str], 
                              analysis: NLPAnalysisResult) -> str:
        """Call the appropriate AI provider synchronously."""
        if provider == AIProvider.ANTHROPIC:
            return self._call_anthropic_sync(model, prompt, analysis)
        elif provider == AIProvider.OPENAI:
            return self._call_openai_sync(model, prompt, analysis)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_anthropic_sync(self, model: str, prompt: Dict[str, str], 
                           analysis: NLPAnalysisResult) -> str:
        """Call Anthropic Claude API synchronously."""
        try:
            provider_config = self.providers[AIProvider.ANTHROPIC]
            client = provider_config['client']
            
            system_message = prompt["system"]
            user_message = prompt["user"]
            
            if "context" in prompt:
                context_str = json.dumps(prompt["context"], indent=2)
                user_message += f"\n\nContext Analysis:\n{context_str}"
            
            message = client.messages.create(
                model=model,
                max_tokens=min(4000, self.config["response_length_limit"] // 2),
                temperature=self.config["creativity_level"],
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            response = message.content[0].text
            
            # Calculate metrics
            input_tokens = len(system_message.split()) + len(user_message.split())
            output_tokens = len(response.split())
            self._update_token_metrics(model, input_tokens, output_tokens)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            return self._retry_with_different_key_sync(AIProvider.ANTHROPIC, model, prompt, analysis)
    
    def _call_openai_sync(self, model: str, prompt: Dict[str, str], 
                         analysis: NLPAnalysisResult) -> str:
        """Call OpenAI API synchronously."""
        try:
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            if "context" in prompt:
                context_str = json.dumps(prompt["context"], indent=2)
                messages.append({"role": "user", "content": f"Analysis Context: {context_str}"})
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=min(4000, self.config["response_length_limit"] // 2),
                temperature=self.config["creativity_level"]
            )
            
            response_text = response.choices[0].message.content
            
            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            self._update_token_metrics(model, input_tokens, output_tokens)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return self._retry_with_different_key_sync(AIProvider.OPENAI, model, prompt, analysis)
    
    async def _retry_with_different_key_async(self, provider: AIProvider, model: str,
                                             prompt: Dict[str, str], 
                                             analysis: NLPAnalysisResult) -> str:
        """Retry API call with different key asynchronously."""
        provider_config = self.providers[provider]
        keys = provider_config['keys']
        current_index = provider_config['current_key_index']
        
        for i in range(1, len(keys)):
            try:
                # Switch to next key
                next_index = (current_index + i) % len(keys)
                next_key = keys[next_index]
                
                if provider == AIProvider.ANTHROPIC:
                    provider_config['client'] = anthropic.Anthropic(api_key=next_key)
                elif provider == AIProvider.OPENAI:
                    openai.api_key = next_key
                
                provider_config['current_key_index'] = next_index
                self.logger.info(f"Retrying with different {provider.value} key")
                
                # Retry the call
                if provider == AIProvider.ANTHROPIC:
                    return await self._call_anthropic_async(model, prompt, analysis)
                elif provider == AIProvider.OPENAI:
                    return await self._call_openai_async(model, prompt, analysis)
                    
            except Exception as e:
                self.logger.warning(f"Retry with key {next_index} failed: {e}")
                continue
        
        raise Exception(f"All {provider.value} keys exhausted")
    
    def _retry_with_different_key_sync(self, provider: AIProvider, model: str,
                                      prompt: Dict[str, str], 
                                      analysis: NLPAnalysisResult) -> str:
        """Retry API call with different key synchronously."""
        provider_config = self.providers[provider]
        keys = provider_config['keys']
        current_index = provider_config['current_key_index']
        
        for i in range(1, len(keys)):
            try:
                next_index = (current_index + i) % len(keys)
                next_key = keys[next_index]
                
                if provider == AIProvider.ANTHROPIC:
                    provider_config['client'] = anthropic.Anthropic(api_key=next_key)
                elif provider == AIProvider.OPENAI:
                    openai.api_key = next_key
                
                provider_config['current_key_index'] = next_index
                self.logger.info(f"Retrying with different {provider.value} key")
                
                if provider == AIProvider.ANTHROPIC:
                    return self._call_anthropic_sync(model, prompt, analysis)
                elif provider == AIProvider.OPENAI:
                    return self._call_openai_sync(model, prompt, analysis)
                    
            except Exception as e:
                self.logger.warning(f"Retry with key {next_index} failed: {e}")
                continue
        
        raise Exception(f"All {provider.value} keys exhausted")
    
    def _update_token_metrics(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Update token usage and cost metrics."""
        with self._lock:
            self.performance_metrics["total_tokens_used"] += input_tokens + output_tokens
            
            # Calculate cost estimate
            if model in self.COST_ESTIMATES:
                costs = self.COST_ESTIMATES[model]
                input_cost = (input_tokens / 1000) * costs["input"]
                output_cost = (output_tokens / 1000) * costs["output"]
                total_cost = input_cost + output_cost
                self.performance_metrics["total_cost_estimate"] += total_cost
    
    def _update_performance_metrics(self, provider: AIProvider, model: str, 
                                   processing_time: float, analysis: NLPAnalysisResult,
                                   response: str, success: bool) -> None:
        """Update comprehensive performance metrics."""
        with self._lock:
            self.performance_metrics["total_requests"] += 1
            
            if success:
                self.performance_metrics["successful_requests"] += 1
                
                # Update average response time
                total_successful = self.performance_metrics["successful_requests"]
                current_avg = self.performance_metrics["average_response_time"]
                self.performance_metrics["average_response_time"] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
            else:
                self.performance_metrics["failed_requests"] += 1
    
    def search_semantic_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search semantic memory using vector similarity."""
        if not self.config["enable_semantic_search"]:
            return []
        
        try:
            # Generate embedding for query
            query_analysis = self.analyze(query)
            if query_analysis.semantic_embedding is None:
                return []
            
            query_embedding = query_analysis.semantic_embedding
            
            # Calculate similarities
            similarities = []
            for memory in self.context_memory["semantic_memory"]:
                memory_embedding = memory["embedding"]
                if memory_embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        memory_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((similarity, memory))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [memory for _, memory in similarities[:top_k]]
        
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate a comprehensive conversation summary."""
        if session_id not in self.context_memory["conversation_contexts"]:
            return {}
        
        context = self.context_memory["conversation_contexts"][session_id]
        
        # Basic statistics
        summary = {
            "session_id": session_id,
            "user_id": context.user_id,
            "total_turns": context.turn_count,
            "duration": None,
            "topics_discussed": [],
            "emotions_detected": Counter(),
            "sentiments": [],
            "entities_mentioned": [],
            "intents_expressed": Counter()
        }
        
        # Calculate duration
        if len(context.conversation_thread) > 0:
            first_turn = context.conversation_thread[0]
            last_turn = context.conversation_thread[-1]
            first_time = datetime.fromisoformat(first_turn["timestamp"])
            last_time = datetime.fromisoformat(last_turn["timestamp"])
            summary["duration"] = str(last_time - first_time)
        
        # Analyze conversation content
        all_topics = []
        all_entities = []
        
        for turn in context.conversation_thread:
            analysis = turn.get("analysis", {})
            
            # Collect topics
            topics = analysis.get("topics", [])
            all_topics.extend(topics)
            
            # Collect emotions
            emotion = analysis.get("emotion", "neutral")
            summary["emotions_detected"][emotion] += 1
            
            # Collect sentiments
            sentiment = analysis.get("sentiment", {})
            if sentiment:
                summary["sentiments"].append({
                    "label": sentiment.get("label"),
                    "score": sentiment.get("score"),
                    "timestamp": turn["timestamp"]
                })
            
            # Collect entities
            entities = analysis.get("entities", [])
            for entity in entities:
                all_entities.append(entity.get("text", ""))
            
            # Collect intents
            intent = analysis.get("intent")
            if isinstance(intent, dict):
                intent_name = intent.get("intent", "unknown")
                summary["intents_expressed"][intent_name] += 1
        
        # Summarize topics and entities
        topic_counter = Counter(all_topics)
        summary["topics_discussed"] = [topic for topic, _ in topic_counter.most_common(10)]
        
        entity_counter = Counter(all_entities)
        summary["entities_mentioned"] = [entity for entity, _ in entity_counter.most_common(10)]
        
        return summary
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile."""
        if user_id not in self.context_memory["user_profiles"]:
            return {}
        
        profile = self.context_memory["user_profiles"][user_id].copy()
        
        # Convert Counter objects to regular dicts for JSON serialization
        profile["emotion_patterns"] = dict(profile["emotion_patterns"])
        profile["preferred_topics"] = dict(profile["preferred_topics"])
        
        # Calculate additional insights
        if profile["sentiment_history"]:
            recent_sentiments = profile["sentiment_history"][-10:]
            avg_sentiment = sum(s["score"] for s in recent_sentiments) / len(recent_sentiments)
            profile["recent_average_sentiment"] = avg_sentiment
            
            # Sentiment trend (improving/declining)
            if len(recent_sentiments) >= 5:
                first_half = recent_sentiments[:len(recent_sentiments)//2]
                second_half = recent_sentiments[len(recent_sentiments)//2:]
                
                first_avg = sum(s["score"] for s in first_half) / len(first_half)
                second_avg = sum(s["score"] for s in second_half) / len(second_half)
                
                trend = "improving" if second_avg > first_avg else "declining" if second_avg < first_avg else "stable"
                profile["sentiment_trend"] = trend
        
        return profile
    
    def export_conversation_history(self, session_id: str, format: str = "json") -> Union[str, Dict]:
        """Export conversation history in specified format."""
        if session_id not in self.context_memory["conversation_contexts"]:
            return {} if format == "json" else ""
        
        context = self.context_memory["conversation_contexts"][session_id]
        
        if format.lower() == "json":
            return {
                "session_info": {
                    "session_id": session_id,
                    "user_id": context.user_id,
                    "total_turns": context.turn_count,
                    "start_time": context.conversation_thread[0]["timestamp"] if context.conversation_thread else None,
                    "end_time": context.conversation_thread[-1]["timestamp"] if context.conversation_thread else None
                },
                "conversation": context.conversation_thread,
                "summary": self.get_conversation_summary(session_id)
            }
        
        elif format.lower() == "text":
            lines = [f"Conversation Export - Session: {session_id}"]
            lines.append(f"User: {context.user_id}")
            lines.append(f"Total Turns: {context.turn_count}")
            lines.append("-" * 50)
            
            for i, turn in enumerate(context.conversation_thread, 1):
                lines.append(f"\nTurn {i} ({turn['timestamp']})")
                lines.append(f"Input: {turn['text']}")
                
                analysis = turn.get('analysis', {})
                sentiment = analysis.get('sentiment', {})
                if sentiment:
                    lines.append(f"Sentiment: {sentiment.get('label')} ({sentiment.get('score', 0):.2f})")
                
                emotion = analysis.get('emotion')
                if emotion:
                    lines.append(f"Emotion: {emotion}")
                
                entities = analysis.get('entities', [])
                if entities:
                    entity_texts = [e.get('text', '') for e in entities[:3]]
                    lines.append(f"Entities: {', '.join(entity_texts)}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            metrics = self.performance_metrics.copy()
            
            # Calculate success rate
            total_requests = metrics["total_requests"]
            if total_requests > 0:
                metrics["success_rate"] = metrics["successful_requests"] / total_requests
                metrics["failure_rate"] = metrics["failed_requests"] / total_requests
            else:
                metrics["success_rate"] = 0.0
                metrics["failure_rate"] = 0.0
            
            # Add memory usage statistics
            metrics["memory_usage"] = {
                "short_term_memory": len(self.context_memory["short_term"]),
                "user_profiles": len(self.context_memory["user_profiles"]),
                "conversation_contexts": len(self.context_memory["conversation_contexts"]),
                "semantic_memory_entries": len(self.context_memory["semantic_memory"]),
                "cache_entries": len(self.cache)
            }
            
            # Add provider statistics
            provider_stats = {}
            for provider, config in self.providers.items():
                provider_stats[provider.value] = {
                    "available": True,
                    "models": config["models"],
                    "current_model": config["default_model"],
                    "api_keys_count": len(config["keys"])
                }
            metrics["providers"] = provider_stats
            
            return metrics
    
    def reset_memory(self, memory_type: str = "all") -> None:
        """Reset specific or all memory types."""
        with self._lock:
            if memory_type == "all" or memory_type == "short_term":
                self.context_memory["short_term"].clear()
                
            if memory_type == "all" or memory_type == "user_profiles":
                self.context_memory["user_profiles"].clear()
                
            if memory_type == "all" or memory_type == "conversation_contexts":
                self.context_memory["conversation_contexts"].clear()
                
            if memory_type == "all" or memory_type == "semantic_memory":
                self.context_memory["semantic_memory"].clear()
                
            if memory_type == "all" or memory_type == "cache":
                self.cache.clear()
        
        self.logger.info(f"Reset {memory_type} memory")
    
    def save_state(self, filepath: str) -> None:
        """Save engine state to file."""
        try:
            state = {
                "config": self.config,
                "context_memory": {
                    "user_profiles": self.context_memory["user_profiles"],
                    "conversation_contexts": {
                        k: {
                            "user_id": v.user_id,
                            "session_id": v.session_id,
                            "turn_count": v.turn_count,
                            "last_interaction": v.last_interaction.isoformat(),
                            "conversation_thread": v.conversation_thread,
                            "user_profile": v.user_profile
                        } for k, v in self.context_memory["conversation_contexts"].items()
                    }
                },
                "performance_metrics": self.performance_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"State saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self, filepath: str) -> None:
        """Load engine state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore config
            if "config" in state:
                self.config.update(state["config"])
            
            # Restore memory
            if "context_memory" in state:
                if "user_profiles" in state["context_memory"]:
                    self.context_memory["user_profiles"].update(state["context_memory"]["user_profiles"])
                
                if "conversation_contexts" in state["context_memory"]:
                    for session_id, context_data in state["context_memory"]["conversation_contexts"].items():
                        context = ConversationContext(
                            user_id=context_data["user_id"],
                            session_id=context_data["session_id"]
                        )
                        context.turn_count = context_data["turn_count"]
                        context.last_interaction = datetime.fromisoformat(context_data["last_interaction"])
                        context.conversation_thread = context_data["conversation_thread"]
                        context.user_profile = context_data["user_profile"]
                        
                        self.context_memory["conversation_contexts"][session_id] = context
            
            # Restore performance metrics
            if "performance_metrics" in state:
                self.performance_metrics.update(state["performance_metrics"])
            
            self.logger.info(f"State loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def __del__(self):
        """Cleanup method."""
        try:
            if hasattr(self, 'loop') and self.loop:
                if not self.loop.is_closed():
                    # Cancel any pending tasks
                    pending_tasks = asyncio.all_tasks(self.loop)
                    for task in pending_tasks:
                        task.cancel()
                    self.loop.close()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Cleanup warning: {e}")


# Example usage and testing
if __name__ == "__main__":
    def example_usage():
        """Demonstrate the Enhanced NLP Engine capabilities."""
        # Initialize the engine
        api_keys = {
            "anthropic": [os.getenv("ANTHROPIC_API_KEY")],
            "openai": [os.getenv("OPENAI_API_KEY")]
        }
        
        config = {
            "personality": "professional and empathetic",
            "creativity_level": 0.7,
            "enable_caching": True,
            "enable_semantic_search": True
        }
        
        engine = EnhancedNLPEngine(api_keys=api_keys, config=config)
        
        # Test text inputs
        test_inputs = [
            "I'm feeling really frustrated with this project. Nothing seems to be working!",
            "Can you help me understand how machine learning algorithms work?",
            "Thank you so much for your help! You've been incredibly helpful.",
            "What's the weather like today?",
            "I think the implementation needs to incorporate more robust error handling."
        ]
        
        print("Enhanced NLP Engine Demo")
        print("=" * 50)
        
        user_id = "demo_user_123"
        session_id = "demo_session_456"
        
        for i, text in enumerate(test_inputs, 1):
            print(f"\n--- Test {i} ---")
            print(f"Input: {text}")
            
            # Analyze the text
            analysis = engine.analyze(text, user_id=user_id, session_id=session_id)
            
            print(f"Sentiment: {analysis.sentiment}")
            print(f"Emotion: {analysis.emotion}")
            print(f"Intent: {analysis.intent}")
            print(f"Entities: {analysis.entities}")
            print(f"Topics: {analysis.topics}")
            print(f"Language: {analysis.language}")
            print(f"Toxicity Score: {analysis.toxicity_score}")
            
            # Generate response
            try:
                response = engine.generate_response(analysis, user_id=user_id, session_id=session_id)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Response generation failed: {e}")
            
            print("-" * 30)
        
        # Demonstrate advanced features
        print("\n--- Advanced Features Demo ---")
        
        # Get conversation summary
        summary = engine.get_conversation_summary(session_id)
        print(f"Conversation Summary: {json.dumps(summary, indent=2)}")
        
        # Get user profile
        profile = engine.get_user_profile(user_id)
        print(f"User Profile: {json.dumps(profile, indent=2)}")
        
        # Get performance metrics
        metrics = engine.get_performance_metrics()
        print(f"Performance Metrics: {json.dumps(metrics, indent=2)}")
        
        # Semantic search example
        search_results = engine.search_semantic_memory("machine learning", top_k=3)
        print(f"Semantic Search Results: {len(search_results)} results found")
        
        # Export conversation
        conversation_export = engine.export_conversation_history(session_id, format="text")
        print(f"Conversation Export Preview:\n{conversation_export[:500]}...")

    # Run example when script is executed directly
    example_usage()