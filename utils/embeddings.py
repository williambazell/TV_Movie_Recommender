import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from abc import ABC

class EmbeddingStrategy(ABC):
    def encode(self, texts: List[str]) -> np.ndarray:
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        pass

class SentenceTransformerStrategy(EmbeddingStrategy):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]

class TfidfStrategy(EmbeddingStrategy):
    def __init__(self, max_features: int = 2000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.dimension = max_features
        self._fitted = False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.vectorizer.fit(texts)
            self._fitted = True
        return self.vectorizer.transform(texts).toarray()
    
    def encode_single(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        return self.vectorizer.transform([text]).toarray()[0]

class CustomEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.model = None
        self._fitted = False
    
    def _build_model(self, vocab_size: int):
        self.model = nn.Sequential(
            nn.Embedding(vocab_size, self.dimension),
            nn.Linear(self.dimension, self.dimension),
            nn.ReLU(),
            nn.Linear(self.dimension, self.dimension),
            nn.Dropout(0.3)
        )
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Custom model must be fitted first")
        return np.random.randn(len(texts), self.dimension)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

class MultiModalEmbedding:
    def __init__(self, strategies: List[EmbeddingStrategy], weights: Optional[List[float]] = None):
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        self.dimension = sum(strategy.dimension for strategy in strategies)
        self.scaler = StandardScaler()
        self.pca = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            emb = strategy.encode(texts)
            embeddings.append(emb * weight)

        combined = np.concatenate(embeddings, axis=1)
        combined = self.scaler.fit_transform(combined)
        if self.pca is not None:
            combined = self.pca.transform(combined)
        
        return combined
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
    
    def fit_pca(self, embeddings: np.ndarray, n_components: int = 128):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)
        self.dimension = n_components

class EmbeddingManager:
    def __init__(self, strategy: str = "sentence_transformer", **kwargs):
        self.strategy_name = strategy
        self.embedder = self._create_embedder(strategy, **kwargs)
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _create_embedder(self, strategy: str, **kwargs) -> EmbeddingStrategy:
        if strategy == "sentence_transformer":
            return SentenceTransformerStrategy(kwargs.get('model_name', 'all-MiniLM-L6-v2'))
        elif strategy == "tfidf":
            return TfidfStrategy(max_features=kwargs.get('max_features', 2000), ngram_range=kwargs.get('ngram_range', (1, 2)))
        elif strategy == "multimodal":
            strategies = []
            if kwargs.get('use_sentence_transformer', True):
                strategies.append(SentenceTransformerStrategy())
            if kwargs.get('use_tfidf', True):
                strategies.append(TfidfStrategy())
            return MultiModalEmbedding(strategies)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        if use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.cache:
                    cached_embeddings.append((i, self.cache[text]))
                    self.cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.cache_misses += 1
            
            if uncached_texts:
                new_embeddings = self.embedder.encode(uncached_texts)
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache[text] = embedding
                all_embeddings = [None] * len(texts)
                for i, emb in cached_embeddings:
                    all_embeddings[i] = emb
                for i, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = emb
                
                return np.array(all_embeddings)
            else:
                return np.array([emb for _, emb in sorted(cached_embeddings)])
        else:
            return self.embedder.encode(texts)
    
    def encode_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        if use_cache and text in self.cache:
            self.cache_hits += 1
            return self.cache[text]
        
        embedding = self.embedder.encode_single(text)
        
        if use_cache:
            self.cache[text] = embedding
            self.cache_misses += 1
        
        return embedding
    
    def get_cache_stats(self) -> Dict[str, int]:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

def create_text_features(media_item: Dict) -> str:
    features = []
    if 'title' in media_item:
        features.append(media_item['title'])
    if 'genres' in media_item:
        features.extend(media_item['genres'])
    if 'overview' in media_item:
        features.append(media_item['overview'])
    if 'cast' in media_item:
        features.extend(media_item['cast'][:5])
    if 'directors' in media_item:
        features.extend(media_item['directors'])
    if 'keywords' in media_item:
        features.extend(media_item['keywords'])
    
    return ' '.join(features)

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(embeddings)

def find_similar_items(query_embedding: np.ndarray, item_embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    similarities = cosine_similarity([query_embedding], item_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_similarities = similarities[top_indices]
    return top_indices, top_similarities
