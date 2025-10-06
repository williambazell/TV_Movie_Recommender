import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os
import joblib
from models import MediaItem, UserProfile, Recommendation, RecommendationBatch
from ml.knn_clustering import KNNClustering, ContentDiscoveryKNN, UserKNN
from ml.neural_network import NeuralRecommendationSystem
from data.feature_processor import ContentFeatureEngineer, UserFeatureEngineer, FeatureCombiner
from utils.embeddings import EmbeddingManager
from utils.logger import logger

class AdvancedRecommendationEngine:
    def __init__(self, embedding_strategy: str = 'sentence_transformer', neural_model_type: str = 'deep', knn_neighbors: int = 20):
        self.embedding_manager = EmbeddingManager(strategy=embedding_strategy)
        self.content_feature_engineer = ContentFeatureEngineer(self.embedding_manager)
        self.user_feature_engineer = UserFeatureEngineer()
        self.feature_combiner = FeatureCombiner()
        
        self.neural_system = NeuralRecommendationSystem(model_type=neural_model_type)
        self.knn_clustering = KNNClustering(n_neighbors=knn_neighbors)
        self.content_discovery = ContentDiscoveryKNN(n_neighbors=knn_neighbors)
        self.user_segmentation = UserKNN(n_neighbors=knn_neighbors)
        
        self.fitted = False
        self.content_catalog = []
        self.user_profiles = {}
        self.interaction_history = {}
        
        self.metrics = {
            'total_recommendations': 0,
            'avg_processing_time': 0.0,
            'model_accuracy': 0.0
        }
    
    def reset_for_fresh_recommendations(self):
        self.fitted = False
    
    def fit(self, content_catalog: List[Dict], user_profiles: Dict[str, UserProfile], interaction_history: Dict[str, List[Dict]]) -> 'AdvancedRecommendationEngine':
        logger.info("Starting recommendation engine training")
        self.content_catalog = content_catalog
        self.user_profiles = user_profiles
        self.interaction_history = interaction_history
        logger.info("Processing content features")
        content_features = self.content_feature_engineer.create_content_features(content_catalog)
        logger.info("Processing user features")
        user_features = []
        user_ids = []
        
        for user_id, profile in user_profiles.items():
            interactions = interaction_history.get(user_id, [])
            user_feature = self.user_feature_engineer.create_user_features(profile.dict(), interactions, content_features)
            user_features.append(user_feature)
            user_ids.append(user_id)

        user_features = np.array(user_features)
        logger.info("Fitting KNN clustering")
        self.knn_clustering.fit(user_features, content_features)
        content_ids = [item['id'] for item in content_catalog]
        content_neighbors = min(self.content_discovery.n_neighbors, len(content_features) - 1)
        if content_neighbors > 0:
            self.content_discovery.n_neighbors = content_neighbors
            self.content_discovery.fit(content_features, content_ids)
        
        user_neighbors = min(self.user_segmentation.n_neighbors, len(user_features) - 1)
        if user_neighbors > 0:
            self.user_segmentation.n_neighbors = user_neighbors
            self.user_segmentation.fit(user_features, user_ids)
        
        logger.info("Preparing neural network training data")
        training_features, training_targets = self._prepare_neural_training_data(content_features, user_features, interaction_history)
        
        if len(training_features) > 0:
            logger.info("Training neural network")
            self.neural_system.train(training_features, training_targets)
        if len(training_features) > 0:
            self.feature_combiner.fit_pca(training_features)
        self.fitted = True
        logger.info("Recommendation engine training completed")
        return self
    
    def _prepare_neural_training_data(self, content_features: np.ndarray, user_features: np.ndarray, interaction_history: Dict[str, List[Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        training_features = []
        training_targets = []
        
        for user_id, interactions in interaction_history.items():
            if user_id not in self.user_profiles:
                continue
            
            user_idx = list(self.user_profiles.keys()).index(user_id)
            user_feature = user_features[user_idx]
            
            for interaction in interactions:
                media_id = interaction.media_id
                if not media_id:
                    continue
                
                content_idx = None
                for i, item in enumerate(self.content_catalog):
                    if item['id'] == media_id:
                        content_idx = i
                        break
                
                if content_idx is None:
                    continue
                
                content_feature = content_features[content_idx]
                combined_feature = np.concatenate([user_feature, content_feature])
                training_features.append(combined_feature)
                if interaction.action == 'like':
                    target = 1.0
                elif interaction.action == 'dislike':
                    target = 0.0
                elif interaction.rating is not None:
                    target = interaction.rating / 10.0
                else:
                    continue
                training_targets.append(target)
        
        return np.array(training_features), np.array(training_targets)
    
    def recommend(self, user_id: str, num_recommendations: int = 10, diversity_factor: float = 0.3, include_explanations: bool = True) -> RecommendationBatch:
        if not self.fitted:
            raise ValueError("Engine must be fitted first")
        
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        
        start_time = datetime.now()
        user_profile = self.user_profiles[user_id]
        user_interactions = self.interaction_history.get(user_id, [])
        user_feature = self.user_feature_engineer.create_user_features(user_profile.dict(), user_interactions, self.content_feature_engineer.create_content_features(self.content_catalog))
        content_features = self.content_feature_engineer.create_content_features(self.content_catalog)
        recommendations = []

        neural_recs = self._get_neural_recommendations(user_feature, content_features, num_recommendations)
        knn_recs = self._get_knn_recommendations(user_feature, content_features, num_recommendations)
        discovery_recs = self._get_discovery_recommendations(user_feature, content_features, num_recommendations)
        segmentation_recs = self._get_segmentation_recommendations(user_id, user_feature, content_features, num_recommendations)
        all_recommendations = self._combine_recommendations(neural_recs, knn_recs, discovery_recs, segmentation_recs)
        logger.info(f"Recommendation counts - Neural: {len(neural_recs)}, KNN: {len(knn_recs)}, Discovery: {len(discovery_recs)}, Segmentation: {len(segmentation_recs)}")
        logger.info(f"Combined recommendations: {len(all_recommendations)}")
        
        final_recommendations = self._apply_diversity_and_filtering(all_recommendations, user_interactions, diversity_factor)
        final_recommendations = self._apply_content_type_filtering(final_recommendations, user_profile.preferred_content_type)
        final_recommendations = self._apply_interaction_filtering(final_recommendations, user_interactions)

        if user_profile.preferred_genres:
            final_recommendations = self._apply_genre_filtering(final_recommendations, user_profile.preferred_genres)
        
        if len(final_recommendations) < num_recommendations:
            logger.info(f"Only {len(final_recommendations)} recommendations after filtering, adding popular content")
            popular_content = self._get_popular_content_fallback(num_recommendations - len(final_recommendations), user_profile.preferred_content_type)
            final_recommendations.extend(popular_content)
        
        seen_content_ids = set()
        seen_titles = set()
        unique_final_recommendations = []
        for content_idx, score, method in final_recommendations:
            if content_idx < len(self.content_catalog):
                content_item = self.content_catalog[content_idx]
                content_id = content_item.get('id')
                content_title = content_item.get('title', '').lower().strip()
                if content_id not in seen_content_ids and content_title not in seen_titles:
                    unique_final_recommendations.append((content_idx, score, method))
                    seen_content_ids.add(content_id)
                    seen_titles.add(content_title)
        
        final_unique_recommendations = []
        seen_final_ids = set()
        seen_final_titles = set()
        
        for content_idx, score, method in unique_final_recommendations[:num_recommendations]:
            if content_idx < len(self.content_catalog):
                content_item = self.content_catalog[content_idx]
                content_id = content_item.get('id')
                content_title = content_item.get('title', '').lower().strip()
                if content_id not in seen_final_ids and content_title not in seen_final_titles:
                    final_unique_recommendations.append((content_idx, score, method))
                    seen_final_ids.add(content_id)
                    seen_final_titles.add(content_title)
        
        recommendation_objects = []
        for i, (content_idx, score, method) in enumerate(final_unique_recommendations):
            content_item = self.content_catalog[content_idx]
            media_item = MediaItem(
                id=content_item['id'],
                title=content_item['title'],
                content_type=content_item['content_type'],
                overview=content_item.get('overview', ''),
                genres=content_item.get('genres', []),
                runtime_min=content_item.get('runtime_min'),
                num_episodes=content_item.get('num_episodes'),
                popularity=content_item.get('popularity', 0.0),
                rating=content_item.get('rating'),
                release_date=content_item.get('release_date'),
                url=content_item.get('url'),
                poster_url=content_item.get('poster_url'),
                backdrop_url=content_item.get('backdrop_url'),
                cast=content_item.get('cast', []),
                directors=content_item.get('directors', []),
                keywords=content_item.get('keywords', [])
            )
            
            reasons = []
            if include_explanations:
                reasons = self._generate_explanation(user_profile, content_item, method, score)
            
            recommendation = Recommendation(
                media_item=media_item,
                score=score,
                reasons=reasons,
                confidence=min(score * 1.2, 1.0),
                diversity_score=self._calculate_diversity_score(content_item, recommendation_objects),
                novelty_score=self._calculate_novelty_score(content_item, user_interactions)
            )
            
            recommendation_objects.append(recommendation)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics['total_recommendations'] += len(recommendation_objects)
        self.metrics['avg_processing_time'] = ((self.metrics['avg_processing_time'] * (self.metrics['total_recommendations'] - len(recommendation_objects)) + processing_time) / self.metrics['total_recommendations'])
        
        return RecommendationBatch(
            user_id=user_id,
            recommendations=recommendation_objects,
            total_count=len(recommendation_objects),
            generated_at=datetime.now(),
            model_version="2.0"
        )
    
    def _get_neural_recommendations(self, user_feature: np.ndarray, content_features: np.ndarray, num_recommendations: int) -> List[Tuple[int, float, str]]:
        recommendations = []
        try:
            combined_features = []
            for content_feature in content_features:
                combined = np.concatenate([user_feature, content_feature])
                combined_features.append(combined)
            
            combined_features = np.array(combined_features)
            predictions = self.neural_system.predict(combined_features)
            top_indices = np.argsort(predictions)[::-1][:num_recommendations]
            
            for idx in top_indices:
                recommendations.append((idx, float(predictions[idx]), 'neural_network'))
            
            logger.info(f"Neural network generated {len(recommendations)} recommendations")
        
        except Exception as e:
            logger.warning(f"Neural network recommendations failed: {e}")
        
        return recommendations
    
    def _get_knn_recommendations(self, user_feature: np.ndarray, content_features: np.ndarray, num_recommendations: int) -> List[Tuple[int, float, str]]:
        recommendations = []
        try:
            try:
                similar_users = self.knn_clustering.find_similar_users(user_feature, top_k=10)
            except:
                similar_users = []
            
            try:
                similar_content = self.knn_clustering.find_similar_content(user_feature, top_k=num_recommendations)
            except:
                similar_content = []
                for i in range(min(num_recommendations, len(content_features))):
                    similarity = 0.5 + (i * 0.1)
                    similar_content.append((i, similarity))
            
            for content_idx, similarity in similar_content:
                recommendations.append((content_idx, float(similarity), 'knn_clustering'))
        
        except Exception as e:
            logger.warning(f"KNN recommendations failed: {e}")
        
        return recommendations
    
    def _get_discovery_recommendations(self, user_feature: np.ndarray, content_features: np.ndarray,num_recommendations: int) -> List[Tuple[int, float, str]]:
        recommendations = []
        try:
            user_similarity_scores = []
            for i, content_feature in enumerate(content_features):
                min_dim = min(len(user_feature), len(content_feature))
                user_subset = user_feature[:min_dim]
                content_subset = content_feature[:min_dim]
                dot_product = np.dot(user_subset, content_subset)
                norm_product = np.linalg.norm(user_subset) * np.linalg.norm(content_subset)
                if norm_product > 0:
                    base_similarity = dot_product / norm_product
                else:
                    base_similarity = 0.0
                content_item = self.content_catalog[i]
                if content_item.get('genres'):
                    genre_boost = 1.0
                    enhanced_similarity = base_similarity * genre_boost
                else:
                    enhanced_similarity = base_similarity
                user_similarity_scores.append((i, enhanced_similarity))
            user_similarity_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = self._apply_genre_balancing(user_similarity_scores, num_recommendations)
        except Exception as e:
            logger.warning(f"Content discovery recommendations failed: {e}")
        
        return recommendations
    
    def _get_segmentation_recommendations(self, user_id: str,user_feature: np.ndarray, content_features: np.ndarray, num_recommendations: int) -> List[Tuple[int, float, str]]:
        recommendations = []
        try:
            try:
                segment_info = self.user_segmentation.get_user_segment(user_feature)
                similar_users = segment_info.get('similar_users', [])
            except:
                similar_users = [(user_id, 1.0) for user_id in self.user_profiles.keys()]
            liked_content = set()
            for similar_user_id, similarity in similar_users:
                user_interactions = self.interaction_history.get(similar_user_id, [])
                for interaction in user_interactions:
                    if interaction.action == 'like':
                        liked_content.add(interaction.media_id)
            for content_idx, content_item in enumerate(self.content_catalog):
                if content_item['id'] in liked_content:
                    score = 0.8
                    recommendations.append((content_idx, score, 'user_segmentation'))
                else:
                    if content_item.get('popularity', 0) > 50:
                        score = 0.3
                        recommendations.append((content_idx, score, 'user_segmentation'))
        except Exception as e:
            logger.warning(f"User segmentation recommendations failed: {e}")
        return recommendations
    
    def _combine_recommendations(self, neural_recs: List[Tuple[int, float, str]], knn_recs: List[Tuple[int, float, str]], discovery_recs: List[Tuple[int, float, str]], segmentation_recs: List[Tuple[int, float, str]]) -> List[Tuple[int, float, str]]:
        method_weights = {
            'neural_network': 0.4,
            'knn_clustering': 0.3,
            'content_discovery': 0.2,
            'user_segmentation': 0.1
        }
        combined_scores = {}
        for recs in [neural_recs, knn_recs, discovery_recs, segmentation_recs]:
            for content_idx, score, method in recs:
                weight = method_weights.get(method, 0.1)
                weighted_score = score * weight
                boosted_score = self._apply_content_quality_boost(content_idx, weighted_score)
                if content_idx in combined_scores:
                    combined_scores[content_idx] = max(combined_scores[content_idx], boosted_score)
                else:
                    combined_scores[content_idx] = boosted_score
        sorted_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        seen_indices = set()
        unique_recommendations = []
        for idx, score in sorted_recommendations:
            if idx not in seen_indices:
                unique_recommendations.append((idx, score, 'combined'))
                seen_indices.add(idx)
        
        return unique_recommendations
    
    def _apply_diversity_and_filtering(self, recommendations: List[Tuple[int, float, str]], user_interactions: List[Dict], diversity_factor: float) -> List[Tuple[int, float, str]]:
        interacted_content = set()
        for interaction in user_interactions:
            interacted_content.add(interaction.media_id)
        filtered_recommendations = []
        for content_idx, score, method in recommendations:
            content_item = self.content_catalog[content_idx]
            if content_item['id'] not in interacted_content:
                filtered_recommendations.append((content_idx, score, method))
        if diversity_factor > 0:
            diverse_recommendations = self._apply_diversity_filtering(filtered_recommendations, diversity_factor)
            return diverse_recommendations
        
        return filtered_recommendations
    
    def _apply_diversity_filtering(self, recommendations: List[Tuple[int, float, str]], diversity_factor: float) -> List[Tuple[int, float, str]]:
        diverse_recommendations = []
        selected_genres = set()
        for content_idx, score, method in recommendations:
            content_item = self.content_catalog[content_idx]
            content_genres = set(content_item.get('genres', []))
            if len(diverse_recommendations) < 5:
                diverse_recommendations.append((content_idx, score, method))
                selected_genres.update(content_genres)
            else:
                genre_overlap = len(selected_genres & content_genres)
                total_genres = len(selected_genres)
                if total_genres == 0 or genre_overlap / total_genres < (1.0 - diversity_factor):
                    diverse_recommendations.append((content_idx, score, method))
                    selected_genres.update(content_genres)
        
        return diverse_recommendations
    
    def _get_popular_content_fallback(self, num_recommendations: int, content_type: str = None) -> List[Tuple[int, float, str]]:
        content_scores = []
        current_year = datetime.now().year
        for i, item in enumerate(self.content_catalog):
            if content_type and item.get('content_type', 'movie').lower() != content_type.lower():
                continue
            popularity = item.get('popularity', 0)
            rating = item.get('rating', 0)
            vote_count = item.get('vote_count', 0)
            release_date = item.get('release_date', '')
            base_score = min(0.9, max(0.1, popularity / 100.0))
            if rating > 0 and vote_count > 50:
                if rating >= 8.0:
                    base_score *= 1.3
                elif rating >= 7.0:
                    base_score *= 1.1
                elif rating < 5.0:
                    base_score *= 0.8
                    
            if release_date:
                try:
                    release_year = int(release_date.split('-')[0])
                    years_old = current_year - release_year
                    if years_old <= 2:
                        base_score *= 1.2
                    elif years_old <= 5:
                        base_score *= 1.1
                except (ValueError, IndexError):
                    pass
            
            content_scores.append((i, base_score))
        
        content_scores.sort(key=lambda x: x[1], reverse=True)
        fallback_recommendations = []
        for content_idx, score in content_scores[:num_recommendations]:
            fallback_recommendations.append((content_idx, score, 'popular_fallback'))
        
        return fallback_recommendations
    
    def _apply_genre_filtering(self, recommendations: List[Tuple[int, float, str]], preferred_genres: List[str]) -> List[Tuple[int, float, str]]:
        genre_mapping = {
            'anime': 'animation',
            'cartoon': 'animation',
            'animated': 'animation',
            'sci-fi': 'science fiction',
            'sci_fi': 'science fiction',
            'scifi': 'science fiction'
        }
        normalized_genres = []
        for genre in preferred_genres:
            normalized = genre_mapping.get(genre.lower(), genre.lower())
            normalized_genres.append(normalized)
        
        filtered_recommendations = []
        for content_idx, score, method in recommendations:
            content_item = self.content_catalog[content_idx]
            content_genres = [g.lower() for g in content_item.get('genres', [])]
            genre_match = any(
                preferred_genre in content_genres or 
                any(preferred_genre in content_genre for content_genre in content_genres) for preferred_genre in normalized_genres
            )
            
            if genre_match:
                boosted_score = min(1.0, score * 1.2)
                filtered_recommendations.append((content_idx, boosted_score, method))
            else:
                filtered_recommendations.append((content_idx, score * 0.8, method))
        
        filtered_recommendations.sort(key=lambda x: x[1], reverse=True)
        return filtered_recommendations
    
    def _apply_content_type_filtering(self, recommendations: List[Tuple[int, float, str]], preferred_content_type: str) -> List[Tuple[int, float, str]]:
        if not preferred_content_type:
            return recommendations
        
        filtered_recommendations = []
        for content_idx, score, method in recommendations:
            if content_idx < len(self.content_catalog):
                content_item = self.content_catalog[content_idx]
                content_type = content_item.get('content_type', 'movie')
                if content_type.lower() == preferred_content_type.lower():
                    filtered_recommendations.append((content_idx, score, method))
        
        return filtered_recommendations
    
    def _apply_interaction_filtering(self, recommendations: List[Tuple[int, float, str]], user_interactions: List[Dict]) -> List[Tuple[int, float, str]]:
        if not user_interactions:
            return recommendations
        interacted_content_ids = set()
        for interaction in user_interactions:
            if hasattr(interaction, 'media_id') and interaction.media_id:
                if hasattr(interaction, 'action') and interaction.action in ['like', 'dislike']:
                    interacted_content_ids.add(interaction.media_id)
        filtered_recommendations = []
        for content_idx, score, method in recommendations:
            if content_idx < len(self.content_catalog):
                content_item = self.content_catalog[content_idx]
                content_id = content_item.get('id')
                if content_id not in interacted_content_ids:
                    filtered_recommendations.append((content_idx, score, method))
        
        return filtered_recommendations
    
    def _apply_genre_balancing(self, similarity_scores: List[Tuple[int, float]], num_recommendations: int) -> List[Tuple[int, float, str]]:
        recommendations = []
        genre_counts = {}
        content_by_genre = {}
        for content_idx, score in similarity_scores:
            if content_idx < len(self.content_catalog):
                content_item = self.content_catalog[content_idx]
                genres = content_item.get('genres', [])
                primary_genre = genres[0].lower() if genres else 'unknown'
                
                if primary_genre not in content_by_genre:
                    content_by_genre[primary_genre] = []
                content_by_genre[primary_genre].append((content_idx, score))
        
        selected_genres = set()
        max_per_genre = max(1, num_recommendations // len(content_by_genre)) if content_by_genre else 1
        
        for content_idx, score in similarity_scores:
            if len(recommendations) >= num_recommendations:
                break
            if content_idx < len(self.content_catalog):
                content_item = self.content_catalog[content_idx]
                genres = content_item.get('genres', [])
                primary_genre = genres[0].lower() if genres else 'unknown'
                
                genre_count = genre_counts.get(primary_genre, 0)
                if (primary_genre not in selected_genres or 
                    genre_count < max_per_genre or 
                    len(selected_genres) < 3):
                    
                    recommendations.append((content_idx, score, 'content_discovery'))
                    genre_counts[primary_genre] = genre_count + 1
                    selected_genres.add(primary_genre)
        
        return recommendations
    
    def _apply_content_quality_boost(self, content_idx: int, base_score: float) -> float:
        if content_idx >= len(self.content_catalog):
            return base_score
        
        content_item = self.content_catalog[content_idx]
        boosted_score = base_score
        release_date = content_item.get('release_date', '')
        if release_date:
            try:
                from datetime import datetime
                release_year = int(release_date.split('-')[0])
                current_year = datetime.now().year
                years_old = current_year - release_year
                
                if years_old <= 1:
                    boosted_score *= 3.0
                elif years_old <= 2:
                    boosted_score *= 2.5
                elif years_old <= 3:
                    boosted_score *= 2.0
                elif years_old <= 5:
                    boosted_score *= 1.3
                else:
                    boosted_score *= 0.5
            except (ValueError, IndexError):
                pass
            
        rating = content_item.get('rating', 0)
        vote_count = content_item.get('vote_count', 0)
        
        if rating > 0 and vote_count > 30:
            if rating >= 8.5:
                boosted_score *= 2.5
            elif rating >= 8.0:
                boosted_score *= 2.2
            elif rating >= 7.5:
                boosted_score *= 1.8
            elif rating >= 7.0:
                boosted_score *= 1.5
            elif rating >= 6.5:
                boosted_score *= 1.2
            elif rating < 5.5:
                boosted_score *= 0.4
            elif rating < 6.0:
                boosted_score *= 0.6

        popularity = content_item.get('popularity', 0)
        if popularity > 200:
            boosted_score *= 1.8
        elif popularity > 100:
            boosted_score *= 1.5
        elif popularity > 50:
            boosted_score *= 1.3
        elif popularity > 20:
            boosted_score *= 1.1
        return min(1.0, boosted_score)
    
    def _generate_explanation(self, user_profile: UserProfile, content_item: Dict, method: str, score: float) -> List[str]:
        reasons = []
        user_genres = set(user_profile.preferred_genres)
        content_genres = set(content_item.get('genres', []))
        genre_overlap = user_genres & content_genres
        
        if genre_overlap:
            reasons.append(f"Matches your preferred genres: {', '.join(genre_overlap)}")
        
        if method == 'neural_network':
            reasons.append("Based on your viewing patterns and preferences")
        elif method == 'knn_clustering':
            reasons.append("Similar to content you've enjoyed")
        elif method == 'content_discovery':
            reasons.append("Similar to other content you might like")
        elif method == 'user_segmentation':
            reasons.append("Popular among users with similar tastes")

        if content_item.get('rating', 0) >= 8.0:
            reasons.append("Highly rated by critics and audiences")
        if content_item.get('popularity', 0) > 50:
            reasons.append("Currently popular and trending")
        
        return reasons
    
    def _calculate_diversity_score(self, content_item: Dict, existing_recommendations: List[Recommendation]) -> float:
        if not existing_recommendations:
            return 1.0
        
        content_genres = set(content_item.get('genres', []))
        existing_genres = set()
        
        for rec in existing_recommendations:
            existing_genres.update(rec.media_item.genres)

        genre_overlap = len(content_genres & existing_genres)
        total_genres = len(content_genres | existing_genres)
        return 1.0 - (genre_overlap / max(total_genres, 1))
    
    def _calculate_novelty_score(self, content_item: Dict, user_interactions: List[Dict]) -> float:
        if not user_interactions:
            return 1.0

        content_genres = set(content_item.get('genres', []))
        interacted_genres = set()
        
        #@TODO
        for interaction in user_interactions:
            pass

        genre_novelty = 1.0 - len(content_genres & interacted_genres) / max(len(content_genres), 1)
        return genre_novelty
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
    
    def save_model(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        self.neural_system.save_system(os.path.join(path, 'neural_system'))
        joblib.dump(self.knn_clustering, os.path.join(path, 'knn_clustering.pkl'))
        joblib.dump(self.content_discovery, os.path.join(path, 'content_discovery.pkl'))
        joblib.dump(self.user_segmentation, os.path.join(path, 'user_segmentation.pkl'))
        joblib.dump(self.content_feature_engineer, os.path.join(path, 'content_feature_engineer.pkl'))
        joblib.dump(self.user_feature_engineer, os.path.join(path, 'user_feature_engineer.pkl'))
        joblib.dump(self.feature_combiner, os.path.join(path, 'feature_combiner.pkl'))
        
        metadata = {
            'fitted': self.fitted,
            'metrics': self.metrics,
            'catalog_size': len(self.content_catalog),
            'user_count': len(self.user_profiles)
        }
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, default=str)
    
    def load_model(self, path: str):
        self.neural_system.load_system(os.path.join(path, 'neural_system'))
        self.knn_clustering = joblib.load(os.path.join(path, 'knn_clustering.pkl'))
        self.content_discovery = joblib.load(os.path.join(path, 'content_discovery.pkl'))
        self.user_segmentation = joblib.load(os.path.join(path, 'user_segmentation.pkl'))
        self.content_feature_engineer = joblib.load(os.path.join(path, 'content_feature_engineer.pkl'))
        self.user_feature_engineer = joblib.load(os.path.join(path, 'user_feature_engineer.pkl'))
        self.feature_combiner = joblib.load(os.path.join(path, 'feature_combiner.pkl'))
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.fitted = metadata['fitted']
        self.metrics = metadata['metrics']
