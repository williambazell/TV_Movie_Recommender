import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from models import UserProfile, UserInteraction, UserAction, ContentType
from utils.logger import logger

class BehavioralAnalyzer:
    def __init__(self):
        self.session_analyzer = SessionAnalyzer()
        self.preference_tracker = PreferenceTracker()
        self.engagement_analyzer = EngagementAnalyzer()
    
    def analyze_user_behavior(self, user_id: str, interactions: List[UserInteraction], time_window: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        recent_interactions = self._filter_recent_interactions(interactions, time_window)
        analysis = {
            'user_id': user_id,
            'session_patterns': self.session_analyzer.analyze_sessions(recent_interactions),
            'preference_evolution': self.preference_tracker.track_preferences(recent_interactions),
            'engagement_metrics': self.engagement_analyzer.calculate_metrics(recent_interactions),
            'behavioral_clusters': self._identify_behavioral_clusters(recent_interactions),
            'temporal_patterns': self._analyze_temporal_patterns(recent_interactions),
            'content_preferences': self._analyze_content_preferences(recent_interactions)
        }
        return analysis
    
    def _filter_recent_interactions(self, interactions: List[UserInteraction], time_window: timedelta) -> List[UserInteraction]:
        cutoff_time = datetime.now() - time_window
        return [i for i in interactions if i.timestamp >= cutoff_time]
    
    def _identify_behavioral_clusters(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if len(interactions) < 5:
            return {'cluster_type': 'insufficient_data', 'confidence': 0.0}
        features = self._extract_behavioral_features(interactions)
        cluster_type = self._classify_behavioral_cluster(features)
        return {
            'cluster_type': cluster_type,
            'confidence': 0.8,
            'features': features
        }
    
    def _extract_behavioral_features(self, interactions: List[UserInteraction]) -> Dict[str, float]:
        features = {}
        features['interaction_frequency'] = len(interactions) / 30.0
        actions = [i.action for i in interactions]
        action_counts = Counter(actions)
        total_actions = len(actions)
        
        features['like_ratio'] = action_counts.get(UserAction.LIKE, 0) / total_actions
        features['dislike_ratio'] = action_counts.get(UserAction.DISLIKE, 0) / total_actions
        features['pass_ratio'] = action_counts.get(UserAction.PASS, 0) / total_actions
        features['rating_ratio'] = action_counts.get(UserAction.RATE, 0) / total_actions
        
        ratings = [i.rating for i in interactions if i.rating is not None]
        if ratings:
            features['avg_rating'] = np.mean(ratings)
            features['rating_std'] = np.std(ratings)
            features['rating_range'] = max(ratings) - min(ratings)
        else:
            features['avg_rating'] = 0.0
            features['rating_std'] = 0.0
            features['rating_range'] = 0.0
        
        timestamps = [i.timestamp for i in interactions]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 for i in range(len(timestamps)-1)]
            features['avg_session_gap'] = np.mean(time_diffs)
            features['session_consistency'] = 1.0 / (np.std(time_diffs) + 1.0)
        else:
            features['avg_session_gap'] = 0.0
            features['session_consistency'] = 0.0
        
        return features
    
    def _classify_behavioral_cluster(self, features: Dict[str, float]) -> str:
        if features['like_ratio'] > 0.7:
            return 'enthusiast'
        elif features['dislike_ratio'] > 0.5:
            return 'selective'
        elif features['pass_ratio'] > 0.6:
            return 'explorer'
        elif features['rating_ratio'] > 0.3:
            return 'critic'
        else:
            return 'casual'
    
    def _analyze_temporal_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if not interactions:
            return {}
        hours = [i.timestamp.hour for i in interactions]
        hour_distribution = Counter(hours)
        weekdays = [i.timestamp.weekday() for i in interactions]
        weekday_distribution = Counter(weekdays)
        sessions = defaultdict(list)
        for interaction in interactions:
            session_id = interaction.session_id or 'default'
            sessions[session_id].append(interaction)
        session_lengths = [len(session) for session in sessions.values()]
        
        return {
            'peak_hours': [hour for hour, count in hour_distribution.most_common(3)],
            'peak_weekdays': [day for day, count in weekday_distribution.most_common(3)],
            'avg_session_length': np.mean(session_lengths) if session_lengths else 0,
            'session_frequency': len(sessions) / 30.0,
            'activity_consistency': 1.0 / (np.std(hours) + 1.0)
        }
    
    def _analyze_content_preferences(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        return {
            'preferred_genres': [],
            'preferred_content_types': [],
            'preferred_ratings': [],
            'preferred_popularity': []
        }

class SessionAnalyzer:
    def analyze_sessions(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if not interactions:
            return {}
        sessions = defaultdict(list)
        for interaction in interactions:
            session_id = interaction.session_id or 'default'
            sessions[session_id].append(interaction)
        session_lengths = [len(session) for session in sessions.values()]
        session_durations = []
        for session in sessions.values():
            if len(session) > 1:
                duration = (max(i.timestamp for i in session) - 
                           min(i.timestamp for i in session)).total_seconds() / 60
                session_durations.append(duration)
        
        return {
            'total_sessions': len(sessions),
            'avg_session_length': np.mean(session_lengths) if session_lengths else 0,
            'avg_session_duration': np.mean(session_durations) if session_durations else 0,
            'session_consistency': 1.0 / (np.std(session_lengths) + 1.0) if session_lengths else 0,
            'longest_session': max(session_lengths) if session_lengths else 0,
            'shortest_session': min(session_lengths) if session_lengths else 0
        }

class PreferenceTracker:
    def __init__(self, time_windows: List[timedelta] = None):
        self.time_windows = time_windows or [
            timedelta(days=7),
            timedelta(days=30),
            timedelta(days=90)
        ]
    
    def track_preferences(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        preference_evolution = {}
        for window in self.time_windows:
            window_interactions = self._filter_by_time_window(interactions, window)
            preferences = self._extract_preferences(window_interactions)
            preference_evolution[f"{window.days}_days"] = preferences

        stability = self._calculate_preference_stability(preference_evolution)
        return {
            'evolution': preference_evolution,
            'stability': stability,
            'trends': self._identify_preference_trends(preference_evolution)
        }
    
    def _filter_by_time_window(self, interactions: List[UserInteraction], window: timedelta) -> List[UserInteraction]:
        cutoff_time = datetime.now() - window
        return [i for i in interactions if i.timestamp >= cutoff_time]
    
    def _extract_preferences(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if not interactions:
            return {}
        actions = [i.action for i in interactions]
        action_counts = Counter(actions)
        total_actions = len(actions)
        preferences = {
            'like_ratio': action_counts.get(UserAction.LIKE, 0) / total_actions,
            'dislike_ratio': action_counts.get(UserAction.DISLIKE, 0) / total_actions,
            'pass_ratio': action_counts.get(UserAction.PASS, 0) / total_actions,
            'rating_ratio': action_counts.get(UserAction.RATE, 0) / total_actions
        }
        ratings = [i.rating for i in interactions if i.rating is not None]
        if ratings:
            preferences.update({
                'avg_rating': np.mean(ratings),
                'rating_std': np.std(ratings),
                'high_rating_ratio': sum(1 for r in ratings if r >= 8.0) / len(ratings),
                'low_rating_ratio': sum(1 for r in ratings if r <= 4.0) / len(ratings)
            })
        
        return preferences
    
    def _calculate_preference_stability(self, preference_evolution: Dict[str, Dict[str, Any]]) -> float:
        if len(preference_evolution) < 2:
            return 0.0
        windows = sorted(preference_evolution.keys())
        stability_scores = []
        
        for i in range(len(windows) - 1):
            current = preference_evolution[windows[i]]
            next_window = preference_evolution[windows[i + 1]]
            similarity = self._calculate_preference_similarity(current, next_window)
            stability_scores.append(similarity)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _calculate_preference_similarity(self, prefs1: Dict[str, Any], prefs2: Dict[str, Any]) -> float:
        common_keys = set(prefs1.keys()) & set(prefs2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = prefs1[key]
            val2 = prefs2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                similarity = 1.0 - abs(val1 - val2) / max(abs(val1) + abs(val2), 1.0)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _identify_preference_trends(self, preference_evolution: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        trends = {}
        metrics = ['like_ratio', 'avg_rating', 'high_rating_ratio']
        
        for metric in metrics:
            values = []
            for window, prefs in preference_evolution.items():
                if metric in prefs:
                    values.append(prefs[metric])
            
            if len(values) >= 2:
                if values[-1] > values[0] * 1.1:
                    trends[metric] = 'increasing'
                elif values[-1] < values[0] * 0.9:
                    trends[metric] = 'decreasing'
                else:
                    trends[metric] = 'stable'
        
        return trends

class EngagementAnalyzer:
    def calculate_metrics(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if not interactions:
            return {}
        total_interactions = len(interactions)
        unique_days = len(set(i.timestamp.date() for i in interactions))
        likes = sum(1 for i in interactions if i.action == UserAction.LIKE)
        dislikes = sum(1 for i in interactions if i.action == UserAction.DISLIKE)
        ratings = sum(1 for i in interactions if i.rating is not None)
        engagement_score = self._calculate_engagement_score(interactions)
        activity_patterns = self._analyze_activity_patterns(interactions)
        
        return {
            'total_interactions': total_interactions,
            'active_days': unique_days,
            'interaction_frequency': total_interactions / max(unique_days, 1),
            'like_ratio': likes / total_interactions,
            'dislike_ratio': dislikes / total_interactions,
            'rating_ratio': ratings / total_interactions,
            'engagement_score': engagement_score,
            'activity_patterns': activity_patterns
        }
    
    def _calculate_engagement_score(self, interactions: List[UserInteraction]) -> float:
        if not interactions:
            return 0.0
        
        action_weights = {
            UserAction.LIKE: 1.0,
            UserAction.DISLIKE: 0.5,
            UserAction.PASS: 0.2,
            UserAction.RATE: 1.5,
            UserAction.WATCH: 2.0
        }
        
        weighted_score = sum(action_weights.get(i.action, 0.1) for i in interactions)
        if len(interactions) > 1:
            time_span = (max(i.timestamp for i in interactions) - 
                        min(i.timestamp for i in interactions)).total_seconds() / 86400
            if time_span > 0:
                weighted_score /= time_span
        
        return min(weighted_score, 10.0)
    
    def _analyze_activity_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if not interactions:
            return {}

        hours = [i.timestamp.hour for i in interactions]
        weekdays = [i.timestamp.weekday() for i in interactions]
        hour_consistency = 1.0 / (np.std(hours) + 1.0)
        weekday_consistency = 1.0 / (np.std(weekdays) + 1.0)
        hour_distribution = Counter(hours)
        weekday_distribution = Counter(weekdays)
        
        return {
            'hour_consistency': hour_consistency,
            'weekday_consistency': weekday_consistency,
            'peak_hour': hour_distribution.most_common(1)[0][0] if hour_distribution else 0,
            'peak_weekday': weekday_distribution.most_common(1)[0][0] if weekday_distribution else 0,
            'activity_spread': len(set(hours)) / 24.0,  # How spread out activity is
            'weekend_activity': sum(1 for wd in weekdays if wd >= 5) / len(weekdays)
        }

class UserProfilingSystem:
    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.profile_updater = ProfileUpdater()
        self.anomaly_detector = AnomalyDetector()
    
    def create_user_profile(self, user_id: str, interactions: List[UserInteraction], initial_preferences: Dict[str, Any] = None) -> UserProfile:
        logger.info(f"Creating profile for user {user_id}")
        behavior_analysis = self.behavioral_analyzer.analyze_user_behavior(user_id, interactions)
        preferences = self._extract_preferences_from_behavior(behavior_analysis, initial_preferences)
        profile = UserProfile(
            user_id=user_id,
            preferred_genres=preferences.get('genres', []),
            preferred_content_type=preferences.get('content_type', ContentType.MOVIE),
            preferred_runtime_min=preferences.get('runtime_min'),
            preferred_runtime_max=preferences.get('runtime_max'),
            preferred_episodes_min=preferences.get('episodes_min'),
            preferred_episodes_max=preferences.get('episodes_max'),
            diversity_preference=preferences.get('diversity_preference', 0.5),
            novelty_preference=preferences.get('novelty_preference', 0.5),
            popularity_preference=preferences.get('popularity_preference', 0.5)
        )
        
        return profile
    
    def update_user_profile(self, profile: UserProfile, new_interactions: List[UserInteraction]) -> UserProfile:
        logger.info(f"Updating profile for user {profile.user_id}")
        behavior_analysis = self.behavioral_analyzer.analyze_user_behavior(profile.user_id, new_interactions)
        updated_preferences = self.profile_updater.update_preferences(profile, behavior_analysis)
        profile.updated_at = datetime.now()
        for key, value in updated_preferences.items():
            setattr(profile, key, value)
        return profile
    
    def detect_anomalies(self, user_id: str, interactions: List[UserInteraction]) -> Dict[str, Any]:
        return self.anomaly_detector.detect_anomalies(user_id, interactions)
    
    def _extract_preferences_from_behavior(self, behavior_analysis: Dict[str, Any], initial_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        preferences = initial_preferences or {}
        cluster_type = behavior_analysis.get('behavioral_clusters', {}).get('cluster_type', 'casual')
        if cluster_type == 'enthusiast':
            preferences.update({
                'diversity_preference': 0.3,
                'novelty_preference': 0.7,
                'popularity_preference': 0.6
            })
        elif cluster_type == 'selective':
            preferences.update({
                'diversity_preference': 0.2,
                'novelty_preference': 0.4,
                'popularity_preference': 0.8
            })
        elif cluster_type == 'explorer':
            preferences.update({
                'diversity_preference': 0.8,
                'novelty_preference': 0.9,
                'popularity_preference': 0.3
            })
        elif cluster_type == 'critic':
            preferences.update({
                'diversity_preference': 0.6,
                'novelty_preference': 0.5,
                'popularity_preference': 0.4
            })
        else:
            preferences.update({
                'diversity_preference': 0.5,
                'novelty_preference': 0.5,
                'popularity_preference': 0.7
            })
        
        return preferences

class ProfileUpdater:
    def update_preferences(self, profile: UserProfile, behavior_analysis: Dict[str, Any]) -> Dict[str, Any]:
        updates = {}
        preference_evolution = behavior_analysis.get('preference_evolution', {})
        if preference_evolution:
            trends = preference_evolution.get('trends', {})
            if 'like_ratio' in trends:
                if trends['like_ratio'] == 'increasing':
                    updates['diversity_preference'] = min(profile.diversity_preference + 0.1, 1.0)
                elif trends['like_ratio'] == 'decreasing':
                    updates['diversity_preference'] = max(profile.diversity_preference - 0.1, 0.0)
            
            if 'avg_rating' in trends:
                if trends['avg_rating'] == 'increasing':
                    updates['novelty_preference'] = min(profile.novelty_preference + 0.1, 1.0)
                elif trends['avg_rating'] == 'decreasing':
                    updates['novelty_preference'] = max(profile.novelty_preference - 0.1, 0.0)
        
        return updates

class AnomalyDetector:
    def detect_anomalies(self, user_id: str, interactions: List[UserInteraction]) -> Dict[str, Any]:
        if len(interactions) < 10:
            return {'anomalies': [], 'risk_score': 0.0}
        
        anomalies = []
        risk_score = 0.0
        recent_interactions = interactions[-10:]
        if self._detect_bot_behavior(recent_interactions):
            anomalies.append('bot_like_behavior')
            risk_score += 0.3
            
        if self._detect_rating_anomalies(recent_interactions):
            anomalies.append('rating_anomalies')
            risk_score += 0.2

        if self._detect_temporal_anomalies(recent_interactions):
            anomalies.append('temporal_anomalies')
            risk_score += 0.2
        
        return {
            'anomalies': anomalies,
            'risk_score': min(risk_score, 1.0),
            'confidence': 0.8
        }
    
    def _detect_bot_behavior(self, interactions: List[UserInteraction]) -> bool:
        if len(interactions) < 5:
            return False
        
        timestamps = [i.timestamp for i in interactions]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]

        if all(diff < 1.0 for diff in time_diffs):
            return True
        actions = [i.action for i in interactions]
        
        if len(set(actions)) == 1:
            return True
        
        return False
    
    def _detect_rating_anomalies(self, interactions: List[UserInteraction]) -> bool:
        ratings = [i.rating for i in interactions if i.rating is not None]
        if len(ratings) < 3:
            return False
        if all(r >= 9.0 for r in ratings) or all(r <= 2.0 for r in ratings):
            return True
        if np.std(ratings) < 0.5:
            return True
        return False
    
    def _detect_temporal_anomalies(self, interactions: List[UserInteraction]) -> bool:
        if len(interactions) < 5:
            return False
        hours = [i.timestamp.hour for i in interactions]
        night_hours = [h for h in hours if h < 6 or h > 22]
        if len(night_hours) / len(hours) > 0.8:
            return True
        
        return False

class UserProfilingSystem:
    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.profile_updater = ProfileUpdater()
        self.anomaly_detector = AnomalyDetector()
        self.user_profiles = {}
        self.interaction_history = {}
    
    def create_user_profile(self, user_id: str, preferences: Dict[str, Any]) -> UserProfile:
        profile = UserProfile(
            user_id=user_id,
            preferred_genres=preferences.get('preferred_genres', []),
            preferred_content_type=preferences.get('preferred_content_type', 'movie'),
            diversity_preference=preferences.get('diversity_preference', 0.5),
            novelty_preference=preferences.get('novelty_preference', 0.5),
            popularity_preference=preferences.get('popularity_preference', 0.5),
            created_at=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        self.interaction_history[user_id] = []
        
        logger.info(f"Creating profile for user {user_id}")
        return profile
    
    def update_user_profile(self, user_id: str, interaction: UserInteraction):
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found for profile update")
            return
        
        profile = self.user_profiles[user_id]
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        self.interaction_history[user_id].append(interaction)
        self._update_preferences_from_interaction(profile, interaction)
        self._update_behavioral_patterns(profile, interaction)
        if hasattr(logger, 'verbose') and logger.verbose:
            logger.info(f"Updating profile for user {user_id}")
    
    def _update_preferences_from_interaction(self, profile: UserProfile, interaction: UserInteraction):
        if interaction.action == UserAction.LIKE:
            profile.diversity_preference = min(profile.diversity_preference + 0.01, 1.0)
        elif interaction.action == UserAction.DISLIKE:
            profile.diversity_preference = max(profile.diversity_preference - 0.01, 0.0)
        if interaction.rating is not None:
            if interaction.rating >= 8.0:
                profile.novelty_preference = min(profile.novelty_preference + 0.02, 1.0)
            elif interaction.rating <= 4.0:
                profile.novelty_preference = max(profile.novelty_preference - 0.02, 0.0)
    
    def _update_behavioral_patterns(self, profile: UserProfile, interaction: UserInteraction):
        profile.last_activity = interaction.timestamp
        profile.activity_frequency += 1
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        return self.user_profiles.get(user_id)
    
    def get_interaction_history(self, user_id: str) -> List[UserInteraction]:
        return self.interaction_history.get(user_id, [])
    
    def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.user_profiles:
            return {}
        interactions = self.interaction_history.get(user_id, [])
        return self.behavioral_analyzer.analyze_user_behavior(user_id, interactions)
    
    def detect_anomalies(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.user_profiles:
            return {}
        interactions = self.interaction_history.get(user_id, [])
        return self.anomaly_detector.detect_anomalies(interactions)
