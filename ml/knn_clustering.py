import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from abc import ABC

class ClusteringAlgorithm(ABC):
    def fit(self, X: np.ndarray) -> 'ClusteringAlgorithm':
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def get_cluster_centers(self) -> np.ndarray:
        pass

class KMeansClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.cluster_centers_

class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'DBSCANClustering':
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.fit_predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.components_

class HierarchicalClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int = 8, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.fit_predict(X)
    
    def get_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        labels = self.model.labels_
        centers = []
        for cluster_id in range(self.n_clusters):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:
                centers.append(cluster_points.mean(axis=0))
        return np.array(centers)

class KNNClustering:
    def __init__(self, n_neighbors: int = 20,algorithm: str = 'auto',metric: str = 'cosine',clustering_method: str = 'kmeans',n_clusters: int = 8):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm,metric=metric)
        
        if clustering_method == 'kmeans':
            self.clusterer = KMeansClustering(n_clusters=n_clusters)
        elif clustering_method == 'dbscan':
            self.clusterer = DBSCANClustering()
        elif clustering_method == 'hierarchical':
            self.clusterer = HierarchicalClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        self.user_scaler = StandardScaler()
        self.content_scaler = StandardScaler()
        self.pca = None
        self.fitted = False
        self.user_clusters = {}
        self.content_clusters = {}
    
    def fit(self, user_features: np.ndarray, content_features: np.ndarray) -> 'KNNClustering':
        user_features_scaled = self.user_scaler.fit_transform(user_features)
        content_features_scaled = self.content_scaler.fit_transform(content_features)
        if self.pca is not None:
            user_features_scaled = self.pca.transform(user_features_scaled)
            content_features_scaled = self.pca.transform(content_features_scaled)
        self.knn_model.fit(user_features_scaled)
        n_users = user_features_scaled.shape[0]
        if n_users < self.n_clusters:
            user_clusters = np.zeros(n_users, dtype=int)
            content_clusters = np.zeros(content_features_scaled.shape[0], dtype=int)
            cluster_center = user_features_scaled.mean(axis=0).reshape(1, -1)
            self.cluster_centers = cluster_center
        else:
            self.clusterer.fit(user_features_scaled)
            user_clusters = self.clusterer.predict(user_features_scaled)
            content_clusters = self.clusterer.predict(content_features_scaled)
            self.cluster_centers = self.clusterer.get_cluster_centers()
        
        self.user_clusters = {
            'labels': user_clusters,
            'centers': self.cluster_centers
        }
        
        self.content_clusters = {
            'labels': content_clusters,
            'centers': self.cluster_centers
        }
        
        self.fitted = True
        return self
    
    def find_similar_users(self, user_features: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        user_features_scaled = self.user_scaler.transform(user_features.reshape(1, -1))
        if self.pca is not None:
            user_features_scaled = self.pca.transform(user_features_scaled)
        distances, indices = self.knn_model.kneighbors(user_features_scaled)
        return indices[0], distances[0]
    
    def find_similar_content(self, content_features: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        content_features_scaled = self.content_scaler.transform(content_features.reshape(1, -1))
        if self.pca is not None:
            content_features_scaled = self.pca.transform(content_features_scaled)
        
        distances, indices = self.knn_model.kneighbors(content_features_scaled)
        return indices[0], distances[0]
    
    def get_user_cluster(self, user_features: np.ndarray) -> int:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        user_features_scaled = self.user_scaler.transform(user_features.reshape(1, -1))
        if self.pca is not None:
            user_features_scaled = self.pca.transform(user_features_scaled)
        return self.clusterer.predict(user_features_scaled)[0]
    
    def get_content_cluster(self, content_features: np.ndarray) -> int:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        content_features_scaled = self.content_scaler.transform(content_features.reshape(1, -1))
        if self.pca is not None:
            content_features_scaled = self.pca.transform(content_features_scaled)
        
        return self.clusterer.predict(content_features_scaled)[0]
    
    def get_cluster_centers(self) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.clusterer.get_cluster_centers()
    
    def evaluate_clustering(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if len(np.unique(labels)) < 2:
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
        
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        return {'silhouette_score': silhouette, 'calinski_harabasz_score': calinski_harabasz}
    
    def optimize_clusters(self, features: np.ndarray, max_clusters: int = 20) -> Dict[str, Any]:
        silhouette_scores = []
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        
        return {
            'optimal_clusters': optimal_clusters,
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'cluster_range': list(cluster_range)
        }

class ContentDiscoveryKNN:
    def __init__(self, n_neighbors: int = 20, metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.content_features = None
        self.content_ids = None
        self.fitted = False
    
    def fit(self, content_features: np.ndarray, content_ids: List[str]) -> 'ContentDiscoveryKNN':
        self.content_features = content_features
        self.content_ids = content_ids
        self.knn_model.fit(content_features)
        self.fitted = True
        return self
    
    def discover_similar_content(self, query_features: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        distances, indices = self.knn_model.kneighbors(query_features.reshape(1, -1))
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            content_id = self.content_ids[idx]
            similarity = 1 - dist
            results.append((content_id, similarity))
        
        return results[:top_k]
    
    def discover_diverse_content(self, query_features: np.ndarray, top_k: int = 10, diversity_factor: float = 0.3) -> List[Tuple[str, float]]:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        distances, indices = self.knn_model.kneighbors(query_features.reshape(1, -1))
        candidates = [(self.content_ids[idx], 1 - dist) for idx, dist in zip(indices[0], distances[0])]
        selected = []
        remaining_candidates = candidates.copy()
        if remaining_candidates:
            selected.append(remaining_candidates.pop(0))
        while len(selected) < top_k and remaining_candidates:
            best_candidate = None
            best_score = -1
            for candidate_id, similarity in remaining_candidates:
                diversity_score = 0
                if len(selected) > 0:
                    selected_features = np.array([self.content_features[self.content_ids.index(sid)] for sid, _ in selected])
                    candidate_feature = self.content_features[self.content_ids.index(candidate_id)].reshape(1, -1)
                    similarities = np.mean([1 - self.knn_model.kneighbors(candidate_feature, n_neighbors = 1)[0][0] for _ in selected_features])
                    diversity_score = 1 - similarities           
                mmr_score = diversity_factor * similarity + (1 - diversity_factor) * diversity_score
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = (candidate_id, similarity)
            
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
        
        return selected

class UserKNN:
    def __init__(self, n_neighbors: int = 20, metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.user_features = None
        self.user_ids = None
        self.fitted = False
    
    def fit(self, user_features: np.ndarray, user_ids: List[str]) -> 'UserKNN':
        self.user_features = user_features
        self.user_ids = user_ids
        self.knn_model.fit(user_features)
        self.fitted = True
        return self
    
    def find_similar_users(self, user_features: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        distances, indices = self.knn_model.kneighbors(user_features.reshape(1, -1))
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            user_id = self.user_ids[idx]
            similarity = 1 - dist
            results.append((user_id, similarity))
        return results[:top_k]
    
    def get_user_segment(self, user_features: np.ndarray) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        similar_users = self.find_similar_users(user_features, top_k=self.n_neighbors)
        segment_features = np.array([self.user_features[self.user_ids.index(uid)] for uid, _ in similar_users])
        segment_center = np.mean(segment_features, axis=0)
        segment_diversity = np.std(segment_features, axis=0).mean()
        return {
            'similar_users': similar_users,
            'segment_center': segment_center,
            'segment_diversity': segment_diversity,
            'segment_size': len(similar_users)
        }
