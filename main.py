import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from models import MediaItem, UserProfile, UserInteraction, UserAction, ContentType, RecommendationBatch
from recommendation_engine import AdvancedRecommendationEngine
from data.api import TMDbAPIClient, ContentProcessor
from data.feature_processor import ContentFeatureEngineer, UserFeatureEngineer
from user_profiling import UserProfilingSystem
from utils.embeddings import EmbeddingManager
from utils.logger import logger, metrics, condensed_logger, enable_verbose_api_logging, disable_verbose_api_logging
from config import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class RecommendationSystemApp:
    def __init__(self):
        self.api_client = TMDbAPIClient()
        self.content_processor = ContentProcessor(self.api_client)
        self.embedding_manager = EmbeddingManager()
        self.profiling_system = UserProfilingSystem()
        self.recommendation_engine = None
        self.content_catalog = []
        self.user_profiles = {}
        self.interaction_history = {}
        self._initialize_system()
    
    def _initialize_system(self):
        logger.info("Initializing recommendation system")
        self._load_content_catalog()
        self.recommendation_engine = AdvancedRecommendationEngine(embedding_strategy='sentence_transformer', neural_model_type='deep', knn_neighbors=20)
        logger.info("System initialization completed")
    
    def _load_content_catalog(self):
        catalog_file = 'content_catalog.json'
        if os.path.exists(catalog_file):
            logger.info("Loading existing content catalog")
            with open(catalog_file, 'r') as f:
                self.content_catalog = json.load(f)
        else:
            logger.info("Creating new content catalog")
            self._build_content_catalog()
            with open(catalog_file, 'w') as f:
                json.dump(self.content_catalog, f, default=str)
    
    def _build_content_catalog(self):
        logger.info("Building content catalog from TMDb API")
        movie_data = self.api_client.discover_content('movie', genres=['action', 'drama', 'comedy'], max_items=20)
        tv_data = self.api_client.discover_content('tv', genres=['drama', 'comedy', 'thriller'], max_items=20)
        movie_items = self.content_processor.batch_process_content(movie_data.get('results', []), 'movie', get_details=False)
        tv_items = self.content_processor.batch_process_content(tv_data.get('results', []), 'tv', get_details=False)
        self.content_catalog = movie_items + tv_items
        logger.info(f"Built catalog with {len(self.content_catalog)} items")
    
    def _build_content_catalog_full(self):
        condensed_logger.start_operation_group("content_catalog", "Building comprehensive content catalog")
        current_year = datetime.now().year
        condensed_logger.update_operation_group("content_catalog", "Fetching trending content...")
        trending_movies = self.api_client.get_trending_content('movie', 'week')
        trending_tv = self.api_client.get_trending_content('tv', 'week')
        condensed_logger.update_operation_group("content_catalog", "Fetching popular content...")
        popular_movies = self.api_client.get_popular_content('movie')
        popular_tv = self.api_client.get_popular_content('tv')
        condensed_logger.update_operation_group("content_catalog", "Fetching recent highly-rated content...")
        recent_movies = self.api_client.discover_content(
            'movie', 
            genres=['action', 'drama', 'comedy', 'animation', 'thriller', 'sci-fi'],
            year=current_year - 1,
            sort_by='vote_average.desc',
            max_items=25
        )
        
        recent_tv = self.api_client.discover_content(
            'tv', 
            genres=['drama', 'comedy', 'thriller', 'animation', 'sci-fi', 'action'],
            year=current_year - 1,
            sort_by='vote_average.desc',
            max_items=25
        )

        condensed_logger.update_operation_group("content_catalog", "Processing content with full details...")
        all_content = []
        if trending_movies.get('results'):
            trending_movie_items = self.content_processor.batch_process_content(trending_movies.get('results', []), 'movie', get_details=True)
            all_content.extend(trending_movie_items)
        
        if trending_tv.get('results'):
            trending_tv_items = self.content_processor.batch_process_content(trending_tv.get('results', []), 'tv', get_details=True)
            all_content.extend(trending_tv_items)
        
        if popular_movies.get('results'):
            popular_movie_items = self.content_processor.batch_process_content(popular_movies.get('results', []), 'movie', get_details=True)
            all_content.extend(popular_movie_items)
        
        if popular_tv.get('results'):
            popular_tv_items = self.content_processor.batch_process_content(popular_tv.get('results', []), 'tv', get_details=True)
            all_content.extend(popular_tv_items)
        
        if recent_movies.get('results'):
            recent_movie_items = self.content_processor.batch_process_content(recent_movies.get('results', []), 'movie', get_details=True)
            all_content.extend(recent_movie_items)
        
        if recent_tv.get('results'):
            recent_tv_items = self.content_processor.batch_process_content(recent_tv.get('results', []), 'tv', get_details=True)
            all_content.extend(recent_tv_items)

        condensed_logger.update_operation_group("content_catalog", "Removing duplicates...")
        seen_ids = set()
        unique_content = []
        for item in all_content:
            if item.get('id') not in seen_ids:
                unique_content.append(item)
                seen_ids.add(item.get('id'))
        
        self.content_catalog = unique_content
        condensed_logger.end_operation_group("content_catalog", f"! Built comprehensive catalog with {len(self.content_catalog)} unique items !")
    
    def _expand_catalog_for_user(self, user_preferences: Dict[str, Any]):
        condensed_logger.start_operation_group("catalog_expansion", "Expanding catalog with content discovery")
        preferred_genres = user_preferences.get('preferred_genres', [])
        content_type = user_preferences.get('preferred_content_type', 'movie')
        if not preferred_genres:
            return
        genre_specific_content = []
        current_year = datetime.now().year
        for genre in preferred_genres:
            recent_content = self.api_client.discover_content(
                content_type,
                genres=[genre],
                year=current_year - 1,
                sort_by='vote_average.desc',
                max_items=10
            )
            
            trending_content = self.api_client.discover_content(
                content_type,
                genres=[genre],
                sort_by='popularity.desc',
                max_items=10
            )

            all_genre_content = []
            if recent_content.get('results'):
                all_genre_content.extend(recent_content.get('results', []))
            if trending_content.get('results'):
                all_genre_content.extend(trending_content.get('results', []))
            
            if all_genre_content:
                processed_content = self.content_processor.batch_process_content(all_genre_content, content_type, get_details=True)
                genre_specific_content.extend(processed_content)
        
        if genre_specific_content:
            existing_ids = {item.get('id') for item in self.content_catalog}
            new_content = [item for item in genre_specific_content if item.get('id') not in existing_ids]
            self.content_catalog.extend(new_content)
            condensed_logger.end_operation_group("catalog_expansion", f"! Added {len(new_content)} genre-specific items to catalog !")
        else:
            condensed_logger.end_operation_group("catalog_expansion", "! No additional genre-specific content found !")
    
    def _get_similar_content_for_recommendations(self, content_ids: List[int], content_type: str) -> List[Dict]:
        similar_content = []
        for content_id in content_ids[:3]:
            try:
                similar_data = self.api_client.get_similar_content(content_id, content_type)
                if similar_data.get('results'):
                    processed_similar = self.content_processor.batch_process_content(similar_data.get('results', [])[:5], content_type, get_details=False)
                    similar_content.extend(processed_similar)
                rec_data = self.api_client.get_recommendations(content_id, content_type)
                if rec_data.get('results'):
                    processed_recs = self.content_processor.batch_process_content(rec_data.get('results', [])[:5], content_type, get_details=False)
                    similar_content.extend(processed_recs)
                    
            except Exception as e:
                logger.warning(f"Failed to get similar content for {content_id}: {e}")
                continue
        
        return similar_content
    
    def create_user_profile(self, user_id: str, preferences: Dict[str, Any] = None) -> UserProfile:
        logger.info(f"Creating profile for user {user_id}")
        profile = UserProfile(
            user_id=user_id,
            preferred_genres=preferences.get('genres', []) if preferences else [],
            preferred_content_type=ContentType(preferences.get('content_type', 'movie')) if preferences else ContentType.MOVIE,
            preferred_runtime_min=preferences.get('runtime_min') if preferences else None,
            preferred_runtime_max=preferences.get('runtime_max') if preferences else None,
            diversity_preference=preferences.get('diversity_preference', 0.5) if preferences else 0.5,
            novelty_preference=preferences.get('novelty_preference', 0.5) if preferences else 0.5,
            popularity_preference=preferences.get('popularity_preference', 0.5) if preferences else 0.5
        )
        
        self.user_profiles[user_id] = profile
        self.interaction_history[user_id] = []
        if hasattr(self, 'profiling_system'):
            self.profiling_system.user_profiles[user_id] = profile
            self.profiling_system.interaction_history[user_id] = []
        
        return profile
    
    def record_interaction(self, user_id: str, media_id: str, action: UserAction, rating: Optional[float] = None, session_id: Optional[str] = None):
        interaction = UserInteraction(
            user_id=user_id,
            media_id=media_id,
            action=action,
            rating=rating,
            session_id=session_id
        )
        
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        
        self.interaction_history[user_id].append(interaction)
        if hasattr(self, 'profiling_system'):
            if user_id not in self.profiling_system.user_profiles:
                self.profiling_system.user_profiles[user_id] = self.user_profiles.get(user_id)
                self.profiling_system.interaction_history[user_id] = []
            self.profiling_system.update_user_profile(user_id, interaction)
    
    def get_recommendations(self, user_id: str, num_recommendations: int = 10) -> RecommendationBatch:
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        condensed_logger.start_operation_group("recommendation_system", "Initializing recommendation system")
        self.recommendation_engine.reset_for_fresh_recommendations()
        condensed_logger.update_operation_group("recommendation_system", "Training ML models with latest interactions...")
        self.recommendation_engine.fit(
            self.content_catalog,
            self.user_profiles,
            self.interaction_history
        )
        condensed_logger.update_operation_group("recommendation_system", "Generating fresh personalized recommendations...")
        recommendations = self.recommendation_engine.recommend(
            user_id=user_id,
            num_recommendations=num_recommendations,
            diversity_factor=0.3,
            include_explanations=True
        )
        condensed_logger.end_operation_group("recommendation_system", f"! Generated {len(recommendations.recommendations)} unique recommendations !")
        return recommendations
    
    def record_user_interaction(self, user_id: str, content_id: int, action: str, rating: float = None):
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found")
        interaction = UserInteraction(
            user_id=user_id,
            media_id=content_id,
            action=UserAction(action),
            rating=rating,
            timestamp=datetime.now()
        )
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        self.interaction_history[user_id].append(interaction)
        if hasattr(self, 'profiling_system'):
            if user_id not in self.profiling_system.user_profiles:
                self.profiling_system.user_profiles[user_id] = self.user_profiles.get(user_id)
                self.profiling_system.interaction_history[user_id] = []
            self.profiling_system.update_user_profile(user_id, interaction)
        if hasattr(logger, 'verbose') and logger.verbose:
            logger.info(f"Recorded {action} for content {content_id} by user {user_id}")
    
    def interactive_demo(self):
        print("--- Advanced Recommendation System Demo ---")
        print("This system uses KNN clustering, neural networks, and deep/wide ML training")
        print()
        user_id = "demo_user"
        print("Creating demo user profile...")
        print("Please provide your preferences:")
        content_type = input("Preferred content type (movie/tv) [movie]: ").strip() or "movie"
        genres = input("Preferred genres (comma-separated) [action,drama]: ").strip() or "action,drama"
        genres_list = [g.strip() for g in genres.split(",")]
        preferences = {
            'content_type': content_type,
            'genres': genres_list,
            'diversity_preference': 0.5,
            'novelty_preference': 0.5,
            'popularity_preference': 0.5
        }
        profile = self.create_user_profile(user_id, preferences)
        print(f"Created profile for user {user_id}")
        print()
        print("Simulating user interactions...")
        self._simulate_user_interactions(user_id)
        print("Generating recommendations...")
        recommendations = self.get_recommendations(user_id, num_recommendations=10)
        print(f"\n--- Recommendations for {user_id} ---")
        print(f"Generated {len(recommendations.recommendations)} recommendations")
        print(f"Model version: {recommendations.model_version}")
        print()
        
        for i, rec in enumerate(recommendations.recommendations, 1):
            print(f"{i}. {rec.media_item.title} ({rec.media_item.content_type})")
            print(f"   Score: {rec.score:.3f} | Confidence: {rec.confidence:.3f}")
            print(f"   Genres: {', '.join(rec.media_item.genres)}")
            print(f"   Overview: {rec.media_item.overview[:100]}...")
            if rec.reasons:
                print(f"   Reasons: {'; '.join(rec.reasons)}")
            print()
        print("--- System Metrics ---")
        engine_metrics = self.recommendation_engine.get_metrics()
        for metric, value in engine_metrics.items():
            print(f"{metric}: {value}")

        cache_stats = self.embedding_manager.get_cache_stats()
        print(f"\nCache Statistics:")
        for stat, value in cache_stats.items():
            print(f"{stat}: {value}")
    
    def _simulate_user_interactions(self, user_id: str):
        popular_movies = [item for item in self.content_catalog if item.get('popularity', 0) > 50][:5]
        for movie in popular_movies:
            self.record_interaction(
                user_id=user_id,
                media_id=movie['id'],
                action=UserAction.LIKE,
                session_id="demo_session"
            )
        for i, movie in enumerate(popular_movies[:3]):
            rating = 8.0 + (i * 0.5)
            self.record_interaction(
                user_id=user_id,
                media_id=movie['id'],
                action=UserAction.RATE,
                rating=rating,
                session_id="demo_session"
            )
    
    def run_benchmark(self):
        print("--- Performance Benchmark ---")
        test_users = []
        for i in range(10):
            user_id = f"test_user_{i}"
            profile = self.create_user_profile(user_id)
            test_users.append(user_id)
            self._simulate_user_interactions(user_id)
        
        
        start_time = time.time()
        total_recommendations = 0
        
        for user_id in test_users:
            recommendations = self.get_recommendations(user_id, num_recommendations=5)
            total_recommendations += len(recommendations.recommendations)
        end_time = time.time()
        print(f"Generated {total_recommendations} recommendations for {len(test_users)} users")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Average time per user: {(end_time - start_time) / len(test_users):.2f} seconds")
        print(f"Recommendations per second: {total_recommendations / (end_time - start_time):.2f}")
    
    def save_system(self, path: str = "recommendation_system"):
        logger.info(f"Saving system to {path}")
        self.recommendation_engine.save_model(path)
        user_data = {
            'profiles': {uid: profile.dict() for uid, profile in self.user_profiles.items()},
            'interactions': {uid: [i.dict() for i in interactions] for uid, interactions in self.interaction_history.items()}
        }
        
        with open(os.path.join(path, 'user_data.json'), 'w') as f:
            json.dump(user_data, f, default=str)
        
        logger.info("System saved successfully")
    
    def load_system(self, path: str = "recommendation_system"):
        logger.info(f"Loading system from {path}")
        self.recommendation_engine.load_model(path)
        with open(os.path.join(path, 'user_data.json'), 'r') as f:
            user_data = json.load(f)
        for uid, profile_data in user_data['profiles'].items():
            self.user_profiles[uid] = UserProfile(**profile_data)
        for uid, interactions_data in user_data['interactions'].items():
            self.interaction_history[uid] = [UserInteraction(**i) for i in interactions_data]
        logger.info("System loaded successfully")
    
    def interactive_mode(self):
        print("--- Advanced Recommendation System - Interactive Mode ---")
        print("This mode demonstrates the following techniques:")
        print("- KNN Clustering for user segmentation")
        print("- Neural Networks for recommendation scoring") 
        print("- Advanced user profiling and behavioral analysis")
        print("- Real-time adaptation and learning")
        print("- Performance monitoring and metrics")
        print()
        print("---Configuration Options ---")
        print("Choose your system configuration:")
        print("1. Fast Mode (Demo settings)")
        print("2. Balanced Mode (Production settings)")
        print("3. Full Mode (Maximum features)")
        print("4. Custom Mode (Advanced configuration)")
        
        config_choice = input("Select configuration [1-4]: ").strip() or "2"
        
        if config_choice == "1":
            self._configure_fast_mode()
        elif config_choice == "2":
            self._configure_balanced_mode()
        elif config_choice == "3":
            self._configure_full_mode()
        elif config_choice == "4":
            self._configure_custom_mode()
        else:
            self._configure_balanced_mode()
            
        print("\n--- Advanced User Profiling ---")
        user_id = "interactive_user"
        user_preferences = self._setup_advanced_user_profiling(user_id)
        print("\n--- Building Content Catalog ---")
        print("Fetching content with full metadata...")
        self._build_content_catalog_full()
        print("Expanding catalog based on your preferences...")
        self._expand_catalog_for_user(user_preferences)
        print("\n--- Machine Learning Demonstration ---")
        self._demonstrate_ml_techniques()
        print("\n===== Interactive Recommendation System =====")
        self._interactive_recommendation_loop(user_id)
        print("\n--- Performance Analysis ---")
        self._display_performance_metrics()
    
    def _demonstrate_ml_techniques(self):
        print("Demonstrating ML techniques:")
        print("1. KNN Clustering for user segmentation")
        print("2. Neural Networks for recommendation scoring")
        print("3. Content feature engineering with embeddings")
        print("4. User behavioral analysis and profiling")
        print("5. Real-time model adaptation")
        print("! All ML components initialized and ready !")
    
    def _configure_fast_mode(self):
        print("Configuring Fast Mode...")
        pass
    
    def _configure_balanced_mode(self):
        print("Configuring Balanced Mode...")
        self.recommendation_engine = AdvancedRecommendationEngine(embedding_strategy='sentence_transformer', neural_model_type='deep', knn_neighbors=20)
    
    def _configure_full_mode(self):
        print("Configuring Full Mode...")
        self.recommendation_engine = AdvancedRecommendationEngine(embedding_strategy='sentence_transformer', neural_model_type='attention', knn_neighbors=50)
    
    def _configure_custom_mode(self):
        print("Configuring Custom Mode...")
        print("Available embedding strategies: sentence_transformer, tfidf, multimodal")
        embedding = input("Embedding strategy [sentence_transformer]: ").strip() or "sentence_transformer"
        
        print("Available neural models: deep, wide_deep, attention")
        model = input("Neural model type [deep]: ").strip() or "deep"
        
        try:
            neighbors = int(input("KNN neighbors [20]: ").strip() or "20")
        except ValueError:
            neighbors = 20
        
        self.recommendation_engine = AdvancedRecommendationEngine(embedding_strategy=embedding, neural_model_type=model, knn_neighbors=neighbors)
    
    def _setup_advanced_user_profiling(self, user_id: str):
        print("Setting up advanced user profiling...")
        print("\nPlease provide your detailed preferences:")
        content_type = input("Preferred content type (movie/tv) [movie]: ").strip() or "movie"
        genres_input = input("Preferred genres (comma-separated) [action,drama]: ").strip() or "action, drama"
        genres = [g.strip() for g in genres_input.split(',')]
        genre_mapping = {
            'anime': 'animation',
            'cartoon': 'animation',
            'animated': 'animation'
        }
        genres = [genre_mapping.get(genre.lower(), genre) for genre in genres]
        print("\nAdvanced preferences (press Enter for defaults):")
        diversity = input("Diversity preference (0.0-1.0) [0.5]: ").strip()
        diversity = float(diversity) if diversity else 0.5
        novelty = input("Novelty preference (0.0-1.0) [0.5]: ").strip()
        novelty = float(novelty) if novelty else 0.5
        popularity = input("Popularity preference (0.0-1.0) [0.5]: ").strip()
        popularity = float(popularity) if popularity else 0.5
        runtime_min = input("Minimum runtime in minutes [0]: ").strip()
        runtime_min = int(runtime_min) if runtime_min else None
        runtime_max = input("Maximum runtime in minutes [300]: ").strip()
        runtime_max = int(runtime_max) if runtime_max else None
        preferences = {
            'content_type': content_type,
            'genres': genres,
            'diversity_preference': diversity,
            'novelty_preference': novelty,
            'popularity_preference': popularity,
            'runtime_min': runtime_min,
            'runtime_max': runtime_max
        }
        
        profile = self.create_user_profile(user_id, preferences)
        print(f"Created advanced profile for user {user_id}")
        print("\nSimulating diverse user interactions for behavioral analysis...")
        self._simulate_advanced_interactions(user_id)
        return {
            'preferred_genres': genres,
            'preferred_content_type': content_type
        }
    
    def _simulate_advanced_interactions(self, user_id: str):
        interactions = [
            {'type': 'popular', 'action': 'like', 'count': 3},
            {'type': 'popular', 'action': 'rate', 'rating': 8.5, 'count': 2},
            {'type': 'niche', 'action': 'like', 'count': 2},
            {'type': 'niche', 'action': 'dislike', 'count': 1},
            {'type': 'genre_variety', 'action': 'rate', 'rating': 7.0, 'count': 4},
            {'type': 'recent', 'action': 'like', 'count': 2},
        ]
        
        for interaction_type in interactions:
            count = interaction_type['count']
            action = interaction_type['action']
            
            for _ in range(count):
                if interaction_type['type'] == 'popular':
                    content = [item for item in self.content_catalog 
                              if item.get('popularity', 0) > 50][:1]
                elif interaction_type['type'] == 'niche':
                    content = [item for item in self.content_catalog 
                              if item.get('popularity', 0) < 20][:1]
                elif interaction_type['type'] == 'genre_variety':
                    content = [item for item in self.content_catalog 
                              if len(item.get('genres', [])) > 2][:1]
                else:
                    content = [item for item in self.content_catalog 
                              if item.get('release_date', '') > '2020'][:1]
                
                if content:
                    item = content[0]
                    if action == 'rate':
                        rating = interaction_type.get('rating', 7.0)
                        self.record_interaction(user_id, item['id'], UserAction.RATE, rating)
                    else:
                        self.record_interaction(user_id, item['id'], UserAction(action))
    
    def _interactive_recommendation_loop(self, user_id: str):
        print("\n--- Interactive Recommendation Loop ---")
        print("Advanced commands available:")
        print("--> 'recommend <number>' - Get new recommendations")
        print("--> 'like <number>' - Like a recommendation (will be filtered from future recs)")
        print("--> 'dislike <number>' - Dislike a recommendation (will be filtered from future recs)") 
        print("--> 'rate <number> <rating>' - Rate a recommendation (1-10)")
        print("--> 'profile' - View your user profile analysis")
        print("--> 'metrics' - View system performance metrics")
        print("--> 'explain <number>' - Get explanation for a recommendation")
        print("--> 'diversity' - Adjust diversity settings")
        print("--> 'quit' - Exit the system")
        
        recommendations = None
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit':
                    break
                elif command.startswith('recommend '):
                    try:
                        num = int(command.split()[1])
                        print(f"\nGenerating {num} fresh recommendations...")
                        recommendations = self.get_recommendations(user_id, num_recommendations=num)
                        self._display_recommendations_interactive(recommendations)
                    except (ValueError, IndexError):
                        print("Invalid format. Use: recommend <number>")
                elif command == 'profile':
                    self._display_user_profile(user_id)
                elif command == 'metrics':
                    self._display_system_metrics()
                elif command.startswith('explain '):
                    try:
                        num = int(command.split()[1])
                        if recommendations and 1 <= num <= len(recommendations.recommendations):
                            rec = recommendations.recommendations[num-1]
                            self._explain_recommendation(rec)
                        else:
                            print("Invalid recommendation number or no recommendations available")
                    except (ValueError, IndexError):
                        print("Invalid format. Use: explain <number>")
                elif command == 'diversity':
                    self._adjust_diversity_settings(user_id)
                elif command.startswith(('like ', 'dislike ', 'rate ')):
                    if not recommendations:
                        print("No recommendations available. Use 'recommend <number>' first.")
                        continue
                    self._handle_interaction_command(command, user_id, recommendations)
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
    
    def _display_user_profile(self, user_id: str):
        if user_id not in self.user_profiles:
            print("User profile not found.")
            return
        
        profile = self.user_profiles[user_id]
        interactions = self.interaction_history.get(user_id, [])
        
        print(f"\n--- User Profile Analysis for {user_id} ---")
        print(f"Preferred Content Type: {profile.preferred_content_type}")
        print(f"Preferred Genres: {', '.join(profile.preferred_genres)}")
        print(f"Diversity Preference: {profile.diversity_preference:.2f}")
        print(f"Novelty Preference: {profile.novelty_preference:.2f}")
        print(f"Popularity Preference: {profile.popularity_preference:.2f}")
        print(f"Total Interactions: {len(interactions)}")
        
        if interactions:
            likes = sum(1 for i in interactions if i.action == 'like')
            dislikes = sum(1 for i in interactions if i.action == 'dislike')
            ratings = [i.rating for i in interactions if i.rating is not None]
            
            print(f"Likes: {likes}, Dislikes: {dislikes}")
            if ratings:
                print(f"Average Rating: {sum(ratings)/len(ratings):.2f}")
                print(f"Rating Range: {min(ratings):.1f} - {max(ratings):.1f}")
    
    def _display_system_metrics(self):
        print("\n--- System Performance Metrics ---")
        
        engine_metrics = self.recommendation_engine.get_metrics()
        for metric, value in engine_metrics.items():
            print(f"{metric}: {value}")
        
        cache_stats = self.embedding_manager.get_cache_stats()
        print(f"\nCache Statistics:")
        for stat, value in cache_stats.items():
            print(f"{stat}: {value}")

        print(f"\nContent Catalog:")
        print(f"Total Items: {len(self.content_catalog)}")
        movies = [item for item in self.content_catalog if item.get('content_type') == 'movie']
        tv_shows = [item for item in self.content_catalog if item.get('content_type') == 'tv']
        print(f"Movies: {len(movies)}, TV Shows: {len(tv_shows)}")
    
    def _explain_recommendation(self, recommendation):
        print(f"\n=== Recommendation Explanation ===")
        print(f"Title: {recommendation.media_item.title}")
        print(f"Score: {recommendation.score:.3f}")
        print(f"Confidence: {recommendation.confidence:.3f}")
        print(f"Reasons: {'; '.join(recommendation.reasons)}")
        print(f"\nDetailed Analysis:")
        print(f"- Content Type: {recommendation.media_item.content_type}")
        print(f"- Genres: {', '.join(recommendation.media_item.genres)}")
        print(f"- Popularity: {recommendation.media_item.popularity}")
        if hasattr(recommendation.media_item, 'rating'):
            print(f"- Rating: {recommendation.media_item.rating}")
    
    def _adjust_diversity_settings(self, user_id: str):
        print("\n--- Diversity Settings ---")
        profile = self.user_profiles[user_id]
        
        print(f"Current diversity preference: {profile.diversity_preference:.2f}")
        new_diversity = input("New diversity preference (0.0-1.0): ").strip()
        
        if new_diversity:
            try:
                diversity = float(new_diversity)
                if 0.0 <= diversity <= 1.0:
                    profile.diversity_preference = diversity
                    print(f"Diversity preference updated to {diversity:.2f}")
                else:
                    print("Diversity must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid number format")
    
    def _handle_interaction_command(self, command: str, user_id: str, recommendations):
        try:
            if command.startswith('like '):
                num = int(command.split()[1])
                if 1 <= num <= len(recommendations.recommendations):
                    rec = recommendations.recommendations[num-1]
                    self.record_user_interaction(user_id, rec.media_item.id, 'like')
                    print(f"* Liked: {rec.media_item.title} *")
                else:
                    print("Invalid recommendation number")
            elif command.startswith('dislike '):
                num = int(command.split()[1])
                if 1 <= num <= len(recommendations.recommendations):
                    rec = recommendations.recommendations[num-1]
                    self.record_user_interaction(user_id, rec.media_item.id, 'dislike')
                    print(f"! Disliked: {rec.media_item.title} !")
                else:
                    print("Invalid recommendation number")
            elif command.startswith('rate '):
                parts = command.split()
                num = int(parts[1])
                rating = float(parts[2])
                if 1 <= num <= len(recommendations.recommendations) and 1 <= rating <= 10:
                    rec = recommendations.recommendations[num-1]
                    self.record_user_interaction(user_id, rec.media_item.id, 'rate', rating)
                    print(f"- Rated {rec.media_item.title}: {rating}/10")
                else:
                    print("Invalid recommendation number or rating (1-10)")
        except (ValueError, IndexError):
            print("Invalid command format")
    
    def _display_performance_metrics(self):
        print("\n--- Final Performance Analysis ---")
        print("System Performance:")
        print(f"- Content Catalog Size: {len(self.content_catalog)} items")
        print(f"- User Profiles: {len(self.user_profiles)}")
        print(f"- Total Interactions: {sum(len(interactions) for interactions in self.interaction_history.values())}")
        print("\nMachine Learning Performance:")
        engine_metrics = self.recommendation_engine.get_metrics()
        for metric, value in engine_metrics.items():
            print(f"- {metric}: {value}")
        cache_stats = self.embedding_manager.get_cache_stats()
        print(f"\nCache Performance:")
        for stat, value in cache_stats.items():
            print(f"- {stat}: {value}")
    
    def _display_recommendations_interactive(self, recommendations):
        if not recommendations or not recommendations.recommendations:
            print("No recommendations available.")
            return
        
        print(f"\n--- Generated {len(recommendations.recommendations)} Recommendations ---")
        
        for i, rec in enumerate(recommendations.recommendations, 1):
            print(f"{i}. {rec.media_item.title} ({rec.media_item.content_type})")
            print(f"   Score: {rec.score:.3f} | Confidence: {rec.confidence:.3f}")
            print(f"   Genres: {', '.join(rec.media_item.genres)}")
            if hasattr(rec.media_item, 'overview') and rec.media_item.overview:
                overview = rec.media_item.overview[:100] + "..." if len(rec.media_item.overview) > 100 else rec.media_item.overview
                print(f"   Overview: {overview}")
            if rec.reasons:
                print(f"   Reasons: {'; '.join(rec.reasons)}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Advanced Recommendation System')
    parser.add_argument('--mode', choices=['demo', 'benchmark', 'interactive'], default='demo', help='Run mode')
    parser.add_argument('--save', action='store_true', help='Save system after running')
    parser.add_argument('--load', action='store_true', help='Load existing system')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (disable beautiful progress indicators)')
    parser.add_argument('--verbose-api', action='store_true', help='Enable verbose API call logging (for debugging)')
    
    args = parser.parse_args()
    
    if args.verbose:
        condensed_logger.verbose = True
    else:
        condensed_logger.verbose = False
    if args.verbose_api:
        enable_verbose_api_logging()
    else:
        disable_verbose_api_logging()
    app = RecommendationSystemApp()
    if args.load:
        try:
            app.load_system()
            print("Loaded existing system")
        except Exception as e:
            print(f"Failed to load system: {e}")
            print("Starting with fresh system")
    if args.mode == 'demo':
        app.interactive_demo()
    elif args.mode == 'benchmark':
        app.run_benchmark()
    elif args.mode == 'interactive':
        app.interactive_mode()
    if args.save:
        app.save_system()
        print("System saved")

if __name__ == "__main__":
    main()
