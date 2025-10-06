from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    MOVIE = "movie"
    TV = "tv"

class UserAction(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    PASS = "pass"
    WATCH = "watch"
    RATE = "rate"

class MediaItem(BaseModel):
    id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Title of the media")
    content_type: ContentType = Field(..., description="Type of content")
    overview: str = Field(default="", description="Plot summary")
    genres: List[str] = Field(default_factory=list, description="List of genres")
    runtime_min: Optional[int] = Field(None, description="Runtime in minutes")
    num_episodes: Optional[int] = Field(None, description="Number of episodes (for TV)")
    popularity: float = Field(default=0.0, description="Popularity score")
    rating: Optional[float] = Field(None, description="Average rating")
    release_date: Optional[str] = Field(None, description="Release date")
    url: Optional[str] = Field(None, description="External URL")
    poster_url: Optional[str] = Field(None, description="Poster image URL")
    backdrop_url: Optional[str] = Field(None, description="Backdrop image URL")
    cast: List[str] = Field(default_factory=list, description="Main cast members")
    directors: List[str] = Field(default_factory=list, description="Directors")
    keywords: List[str] = Field(default_factory=list, description="Keywords/tags")
    
    class Config:
        use_enum_values = True

class UserProfile(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    preferred_genres: List[str] = Field(default_factory=list, description="Preferred genres")
    preferred_content_type: ContentType = Field(default=ContentType.MOVIE, description="Preferred content type")
    preferred_runtime_min: Optional[int] = Field(None, description="Preferred runtime minimum")
    preferred_runtime_max: Optional[int] = Field(None, description="Preferred runtime maximum")
    preferred_episodes_min: Optional[int] = Field(None, description="Preferred episodes minimum")
    preferred_episodes_max: Optional[int] = Field(None, description="Preferred episodes maximum")
    diversity_preference: float = Field(default=0.5, description="Diversity preference (0-1)")
    novelty_preference: float = Field(default=0.5, description="Novelty preference (0-1)")
    popularity_preference: float = Field(default=0.5, description="Popularity preference (0-1)")
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    activity_frequency: int = Field(default=0, description="Activity frequency counter")
    
    class Config:
        use_enum_values = True

class UserInteraction(BaseModel):
    user_id: str = Field(..., description="User identifier")
    media_id: str = Field(..., description="Media item identifier")
    action: UserAction = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(None, description="Rating (1-10) if applicable")
    timestamp: datetime = Field(default_factory=datetime.now, description="Interaction timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    class Config:
        use_enum_values = True

class Recommendation(BaseModel):
    media_item: MediaItem = Field(..., description="Recommended media item")
    score: float = Field(..., description="Recommendation score")
    reasons: List[str] = Field(default_factory=list, description="Explanation for recommendation")
    confidence: float = Field(..., description="Confidence level (0-1)")
    diversity_score: float = Field(default=0.0, description="Diversity score")
    novelty_score: float = Field(default=0.0, description="Novelty score")
    
class RecommendationBatch(BaseModel):
    user_id: str = Field(..., description="User identifier")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    total_count: int = Field(..., description="Total number of recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    model_version: str = Field(default="1.0", description="Model version used")
    
class UserCluster(BaseModel):
    cluster_id: int = Field(..., description="Cluster identifier")
    user_ids: List[str] = Field(..., description="User IDs in this cluster")
    centroid_features: List[float] = Field(..., description="Cluster centroid features")
    size: int = Field(..., description="Number of users in cluster")
    created_at: datetime = Field(default_factory=datetime.now, description="Cluster creation time")
    
class ModelMetrics(BaseModel):
    model_name: str = Field(..., description="Model name")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    training_time: float = Field(..., description="Training time in seconds")
    inference_time: float = Field(..., description="Inference time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
class FeatureVector(BaseModel):
    media_id: str = Field(..., description="Media item identifier")
    embedding: List[float] = Field(..., description="Embedding vector")
    genre_features: List[float] = Field(default_factory=list, description="Genre features")
    content_features: List[float] = Field(default_factory=list, description="Content features")
    popularity_features: List[float] = Field(default_factory=list, description="Popularity features")
    temporal_features: List[float] = Field(default_factory=list, description="Temporal features")
    combined_features: List[float] = Field(default_factory=list, description="Combined feature vector")
    
class TrainingData(BaseModel):
    user_id: str = Field(..., description="User identifier")
    media_id: str = Field(..., description="Media item identifier")
    rating: float = Field(..., description="User rating")
    features: FeatureVector = Field(..., description="Feature vector")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
