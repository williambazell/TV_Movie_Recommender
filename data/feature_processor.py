import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
from utils.embeddings import EmbeddingManager
from utils.logger import logger, monitor_data_processing

class ContentFeatureEngineer:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.scaler = StandardScaler()
        self.genre_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fitted = False
    
    def create_content_features(self, content_items: List[Dict]) -> np.ndarray:
        features = []
        for item in content_items:
            feature_vector = self._extract_content_features(item)
            features.append(feature_vector)
        features_array = np.array(features)
        if not self.fitted:
            self._fit_scalers(features_array)
            self.fitted = True
        return self._transform_features(features_array)
    
    def _extract_content_features(self, item: Dict) -> List[float]:
        features = []
        features.extend(self._extract_basic_features(item))
        features.extend(self._extract_text_features(item))
        features.extend(self._extract_temporal_features(item))
        features.extend(self._extract_genre_features(item))
        features.extend(self._extract_popularity_features(item))
        features.extend(self._extract_content_specific_features(item))
        return features
    
    def _extract_basic_features(self, item: Dict) -> List[float]:
        features = []
        features.append(item.get('popularity', 0.0))
        features.append(item.get('rating', 0.0))
        features.append(item.get('vote_count', 0.0))
        runtime = item.get('runtime_min', 0) or 0
        features.append(runtime)
        features.append(1.0 if runtime > 0 else 0.0)
        episodes = item.get('num_episodes', 0) or 0
        features.append(episodes)
        features.append(1.0 if episodes > 0 else 0.0)
        content_type = item.get('content_type', 'movie')
        features.append(1.0 if content_type == 'tv' else 0.0)
        return features
    
    def _extract_text_features(self, item: Dict) -> List[float]:
        features = []
        title = item.get('title', '')
        overview = item.get('overview', '')
        features.append(len(title))
        features.append(len(overview))
        features.append(len(title.split()))
        features.append(len(overview.split()))
        features.append(self._calculate_text_complexity(overview))
        features.extend(self._extract_language_features(overview))
        return features
    
    def _extract_temporal_features(self, item: Dict) -> List[float]:
        features = []
        
        release_date = item.get('release_date', '')
        if release_date:
            try:
                date_obj = datetime.strptime(release_date, '%Y-%m-%d')
                current_year = datetime.now().year
                age = current_year - date_obj.year
                features.append(age)
                decade = (date_obj.year // 10) * 10
                features.append(decade)
                season = (date_obj.month - 1) // 3
                features.append(season)
                features.append(1.0 if age <= 2 else 0.0)
                features.append(1.0 if age >= 20 else 0.0)
                
            except ValueError:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)
        
        return features
    
    def _extract_genre_features(self, item: Dict) -> List[float]:
        features = []
        genres = item.get('genres', [])
        features.append(len(genres))
        if genres:
            genre_counts = {}
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            total = len(genres)
            entropy = -sum((count/total) * np.log2(count/total) for count in genre_counts.values())
            features.append(entropy)
        else:
            features.append(0.0)
        genre_flags = self._get_genre_flags(genres)
        features.extend(genre_flags)
        
        return features
    
    def _extract_popularity_features(self, item: Dict) -> List[float]:
        features = []
        popularity = item.get('popularity', 0.0)
        rating = item.get('rating', 0.0)
        vote_count = item.get('vote_count', 0)
        
        features.append(min(popularity / 100.0, 1.0))
        features.append(1.0 if rating >= 8.0 else 0.0)
        features.append(1.0 if rating >= 6.0 else 0.0)
        features.append(1.0 if rating < 4.0 else 0.0)
        
        features.append(1.0 if vote_count >= 1000 else 0.0)
        features.append(1.0 if vote_count >= 100 else 0.0)
        
        popularity_score = (rating * np.log1p(vote_count)) / 10.0
        features.append(min(popularity_score, 1.0))
        
        return features
    
    def _extract_content_specific_features(self, item: Dict) -> List[float]:
        features = []
        cast = item.get('cast', [])
        directors = item.get('directors', [])
        features.append(len(cast))
        features.append(len(directors))
        keywords = item.get('keywords', [])
        features.append(len(keywords))
        overview = item.get('overview', '').lower()
        mature_indicators = ['violence', 'adult', 'mature', 'explicit', 'graphic']
        features.append(sum(1 for indicator in mature_indicators if indicator in overview))
        return features
    
    def _calculate_text_complexity(self, text: str) -> float:
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        complexity = (avg_word_length * avg_sentence_length) / 100.0
        return min(complexity, 1.0)
    
    def _extract_language_features(self, text: str) -> List[float]:
        features = []
        if not text:
            return [0.0] * 4
        features.append(sum(1 for c in text if c.isupper()) / len(text))
        features.append(sum(1 for c in text if c.isdigit()) / len(text))
        features.append(sum(1 for c in text if c in '.,!?;:') / len(text))
        features.append(text.count('?') / len(text))
        return features
    
    def _get_genre_flags(self, genres: List[str]) -> List[float]:
        common_genres = [
            'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
            'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
            'romance', 'science fiction', 'thriller', 'war', 'western'
        ]
        
        genre_set = set(genre.lower() for genre in genres)
        return [1.0 if genre in genre_set else 0.0 for genre in common_genres]
    
    #@TODO:
    def _fit_scalers(self, features_array: np.ndarray):
        self.scaler.fit(features_array)
        all_genres = []
        for item in features_array:
            pass
    
    def _transform_features(self, features_array: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features_array)

class UserFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    @monitor_data_processing
    def create_user_features(self, user_profile: Dict, interaction_history: List[Dict], content_features: np.ndarray) -> np.ndarray:
        features = []
        features.extend(self._extract_profile_features(user_profile))
        features.extend(self._extract_interaction_features(interaction_history))
        features.extend(self._extract_preference_features(user_profile, interaction_history))
        features.extend(self._extract_behavioral_features(interaction_history))
        features.extend(self._extract_content_preference_features(interaction_history, content_features))
        
        features_array = np.array(features)
        
        if not self.fitted:
            self.scaler.fit(features_array.reshape(1, -1))
            self.fitted = True
        
        return self.scaler.transform(features_array.reshape(1, -1))[0]
    
    def _extract_profile_features(self, profile: Dict) -> List[float]:
        features = []
        content_type = profile.get('preferred_content_type', 'movie')
        features.append(1.0 if content_type == 'tv' else 0.0)
        min_runtime = profile.get('preferred_runtime_min', 0) or 0
        max_runtime = profile.get('preferred_runtime_max', 0) or 0
        features.append(min_runtime)
        features.append(max_runtime)
        features.append((min_runtime + max_runtime) / 2.0)
        
        min_episodes = profile.get('preferred_episodes_min', 0) or 0
        max_episodes = profile.get('preferred_episodes_max', 0) or 0
        features.append(min_episodes)
        features.append(max_episodes)
        
        features.append(profile.get('diversity_preference', 0.5))
        features.append(profile.get('novelty_preference', 0.5))
        features.append(profile.get('popularity_preference', 0.5))
        
        return features
    
    def _extract_interaction_features(self, interactions: List[Dict]) -> List[float]:
        features = []
        
        if not interactions:
            return [0.0] * 10
        
        total_interactions = len(interactions)
        features.append(total_interactions)
        likes = sum(1 for i in interactions if i.action == 'like')
        dislikes = sum(1 for i in interactions if i.action == 'dislike')
        ratings = sum(1 for i in interactions if i.rating is not None)
        
        features.append(likes)
        features.append(dislikes)
        features.append(ratings)
        features.append(likes / total_interactions if total_interactions > 0 else 0.0)
        features.append(dislikes / total_interactions if total_interactions > 0 else 0.0)

        user_ratings = [i.rating for i in interactions if i.rating is not None]
        if user_ratings:
            features.append(np.mean(user_ratings))
            features.append(np.std(user_ratings))
            features.append(max(user_ratings))
            features.append(min(user_ratings))
        else:
            features.extend([0.0] * 4)
        
        return features
    
    def _extract_preference_features(self, profile: Dict, interactions: List[Dict]) -> List[float]:
        features = []
        preferred_genres = profile.get('preferred_genres', [])
        features.append(len(preferred_genres))
        
        if preferred_genres:
            genre_counts = {}
            for genre in preferred_genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            total = len(preferred_genres)
            entropy = -sum((count/total) * np.log2(count/total) for count in genre_counts.values())
            features.append(entropy)
        else:
            features.append(0.0)

        if interactions:
            recent_interactions = [i for i in interactions if datetime.now() - i.timestamp < timedelta(days=30)]
            features.append(len(recent_interactions))
            features.append(len(recent_interactions) / len(interactions))
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_behavioral_features(self, interactions: List[Dict]) -> List[float]:
        features = []
        if not interactions:
            return [0.0] * 5
        sessions = {}
        for interaction in interactions:
            session_id = interaction.session_id
            if session_id:
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(interaction)
        
        features.append(len(sessions))
        features.append(len(interactions) / len(sessions) if sessions else 0.0)
        timestamps = [i.timestamp for i in interactions if i.timestamp]
        if timestamps:
            hours = [t.hour for t in timestamps]
            features.append(np.mean(hours))
            features.append(np.std(hours))
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_content_preference_features(self, interactions: List[Dict], content_features: np.ndarray) -> List[float]:
        features = []
        if not interactions or content_features.size == 0:
            return [0.0] * 5
        content_ids = [i.media_id for i in interactions if i.media_id]
        
        if not content_ids:
            return [0.0] * 5
        features.extend([0.0] * 5)
        return features

class FeatureCombiner:
    def __init__(self):
        self.pca = None
        self.fitted = False
        
    def combine_features(self, content_features: np.ndarray, user_features: np.ndarray, interaction_features: np.ndarray = None) -> np.ndarray:
        features_list = [content_features, user_features]
        
        if interaction_features is not None:
            features_list.append(interaction_features)
        
        combined = np.concatenate(features_list, axis=1)
        if self.pca is not None:
            combined = self.pca.transform(combined)
        
        return combined
    
    def fit_pca(self, features: np.ndarray, n_components: int = 128):
        max_components = min(features.shape[0], features.shape[1])
        n_components = min(n_components, max_components)
        
        if n_components > 0:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(features)
        else:
            self.pca = None
        self.fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("PCA must be fitted first")
        if self.pca is None:
            return features
        return self.pca.transform(features)
