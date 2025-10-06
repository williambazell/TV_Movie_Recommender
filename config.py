import os
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class ModelConfig(BaseSettings):
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model name")
    knn_neighbors: int = Field(default=20, description="Number of neighbors for KNN clustering")
    neural_net_hidden_size: int = Field(default=128, description="Hidden layer size for neural network")
    neural_net_dropout: float = Field(default=0.3, description="Dropout rate for neural network")
    learning_rate: float = Field(default=0.001, description="Learning rate for neural network")
    batch_size: int = Field(default=32, description="Batch size for training")
    epochs: int = Field(default=100, description="Number of training epochs")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")
    
    class Config:
        env_prefix = "MODEL_"

class APIConfig(BaseSettings):
    tmdb_api_key: str = Field(default="", description="TMDb API key")
    tmdb_base_url: str = Field(default="https://api.themoviedb.org/3", description="TMDb base URL")
    request_timeout: int = Field(default=10, description="Request timeout in seconds")
    rate_limit_delay: float = Field(default=0.25, description="Delay between requests")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    class Config:
        env_prefix = "API_"

class DatabaseConfig(BaseSettings):
    database_url: str = Field(default="sqlite:///./recommendations.db", description="Database URL")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL for caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    class Config:
        env_prefix = "DB_"

class RecommendationConfig(BaseSettings):
    max_catalog_size: int = Field(default=1000, description="Maximum catalog size")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    diversity_factor: float = Field(default=0.3, description="Diversity factor for recommendations")
    popularity_weight: float = Field(default=0.1, description="Weight for popularity in scoring")
    genre_weight: float = Field(default=0.3, description="Weight for genre matching")
    embedding_weight: float = Field(default=0.5, description="Weight for embedding similarity")
    length_weight: float = Field(default=0.1, description="Weight for length matching")
    
    class Config:
        env_prefix = "REC_"

class LoggingConfig(BaseSettings):
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="recommendations.log", description="Log file path")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    class Config:
        env_prefix = "LOG_"

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.recommendation = RecommendationConfig()
        self.logging = LoggingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.dict(),
            "api": self.api.dict(),
            "database": self.database.dict(),
            "recommendation": self.recommendation.dict(),
            "logging": self.logging.dict()
        }

config = Config()
