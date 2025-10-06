import requests
import time
import json
from typing import List, Dict, Optional
import redis
import hashlib
from utils.logger import logger
from config import config

class RateLimiter:
    def __init__(self, max_requests: int = 40, time_window: int = 10):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        self.requests.append(time.time())
    
    def wait_time(self) -> float:
        if self.can_make_request():
            return 0.0
        
        now = time.time()
        oldest_request = min(self.requests)
        return self.time_window - (now - oldest_request)

class CacheManager:
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or config.database.redis_url
        self.redis_client = None
        self._connect()
    
    #@TODO: REQUIRES REDIS API
    def _connect(self):
        try: 
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
            self.redis_client = None
            self.memory_cache = {}
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        if self.redis_client is None:
            cache_key = self._get_cache_key(endpoint, params)
            return self.memory_cache.get(cache_key)
        
        cache_key = self._get_cache_key(endpoint, params)
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def set(self, endpoint: str, params: Dict, data: Dict, ttl: int = None):
        ttl = ttl or config.database.cache_ttl
        
        if self.redis_client is None:
            cache_key = self._get_cache_key(endpoint, params)
            self.memory_cache[cache_key] = data
            return
        
        cache_key = self._get_cache_key(endpoint, params)
        try:
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

class TMDbAPIClient:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or config.api.tmdb_api_key
        self.base_url = base_url or config.api.tmdb_base_url
        if not self.api_key:
            logger.warning("TMDb API key not provided. Cannot pull catalog unless sample in \"content_catalog.json\"")
        self.rate_limiter = RateLimiter()
        self.cache_manager = CacheManager()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AdvancedRecommendationSystem/1.0'
        })
        self.genre_mappings = {
            'movie': {},
            'tv': {}
        }
        self._load_genre_mappings()
    
    def _load_genre_mappings(self):
        for content_type in ['movie', 'tv']:
            try:
                self.genre_mappings[content_type] = self._get_genre_mapping(content_type)
            except Exception as e:
                logger.warning(f"Failed to load {content_type} genre mappings: {e}")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        if not self.api_key:
            logger.error("TMDb API key not provided. Cannot make API requests.")
            return {"results": [], "total_results": 0, "total_pages": 0}
    
        cached_data = self.cache_manager.get(endpoint, params or {})
        if cached_data:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_time()
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        url = f"{self.base_url}/{endpoint}"
        request_params = {'api_key': self.api_key}
        if params:
            request_params.update(params)
        
        for attempt in range(config.api.max_retries):
            try:
                self.rate_limiter.record_request()
                response = self.session.get(
                    url, 
                    params=request_params, 
                    timeout=config.api.request_timeout
                )
                response.raise_for_status()
                
                data = response.json()
                self.cache_manager.set(endpoint, params or {}, data)
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < config.api.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def _get_genre_mapping(self, content_type: str) -> Dict[str, int]:
        endpoint = f"genre/{content_type}/list"
        data = self._make_request(endpoint)
        return {genre['name'].lower(): genre['id'] for genre in data.get('genres', [])}
    
    def discover_content(self, content_type: str = 'movie', genres: List[str] = None, year: int = None, page: int = 1, sort_by: str = 'popularity.desc', max_items: int = None) -> Dict:
        if max_items is None:
            endpoint = f"discover/{content_type}"
            params = {
                'language': 'en-US',
                'sort_by': sort_by,
                'page': page,
                'with_original_language': 'en'
            }
            
            if genres:
                genre_ids = []
                for genre in genres:
                    genre_lower = genre.lower()
                    if genre_lower in self.genre_mappings[content_type]:
                        genre_ids.append(self.genre_mappings[content_type][genre_lower])
                if genre_ids:
                    params['with_genres'] = ','.join(map(str, genre_ids))
            
            if year:
                if content_type == 'movie':
                    params['year'] = year
                else:
                    params['first_air_date_year'] = year
            
            return self._make_request(endpoint, params)
        else:
            all_results = []
            current_page = 1
            while len(all_results) < max_items:
                endpoint = f"discover/{content_type}"
                params = {
                    'language': 'en-US',
                    'sort_by': sort_by,
                    'page': current_page,
                    'with_original_language': 'en'
                }
                
                if genres:
                    genre_ids = []
                    for genre in genres:
                        genre_lower = genre.lower()
                        if genre_lower in self.genre_mappings[content_type]:
                            genre_ids.append(self.genre_mappings[content_type][genre_lower])
                    if genre_ids:
                        params['with_genres'] = ','.join(map(str, genre_ids))
                
                if year:
                    if content_type == 'movie':
                        params['year'] = year
                    else:
                        params['first_air_date_year'] = year
                
                response = self._make_request(endpoint, params)
                results = response.get('results', [])
                
                if not results:
                    break
                
                all_results.extend(results)
                current_page += 1
                if current_page > response.get('total_pages', 1):
                    break
                
            limited_results = all_results[:max_items]
            return {
                'results': limited_results,
                'total_results': len(limited_results),
                'total_pages': current_page - 1,
                'page': 1
            }
    
    def get_content_details(self, content_id: int, content_type: str = 'movie') -> Dict:
        endpoint = f"{content_type}/{content_id}"
        params = {
            'language': 'en-US',
            'append_to_response': 'credits,keywords,similar,recommendations'
        }
        return self._make_request(endpoint, params)
    
    def search_content(self, query: str, content_type: str = 'movie',page: int = 1) -> Dict:
        endpoint = f"search/{content_type}"
        params = {
            'query': query,
            'language': 'en-US',
            'page': page
        }
        return self._make_request(endpoint, params)
    
    def get_similar_content(self, content_id: int, content_type: str = 'movie') -> Dict:
        endpoint = f"{content_type}/{content_id}/similar"
        params = {'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_recommendations(self, content_id: int, content_type: str = 'movie') -> Dict:
        endpoint = f"{content_type}/{content_id}/recommendations"
        params = {'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_trending_content(self, content_type: str = 'movie', time_window: str = 'week') -> Dict:
        endpoint = f"trending/{content_type}/{time_window}"
        params = {
            'language': 'en-US',
            'with_original_language': 'en'
        }
        return self._make_request(endpoint, params)
    
    def get_popular_content(self, content_type: str = 'movie', page: int = 1) -> Dict:
        endpoint = f"{content_type}/popular"
        params = {
            'language': 'en-US',
            'page': page,
            'with_original_language': 'en'
        }
        return self._make_request(endpoint, params)
    
    def get_top_rated_content(self, content_type: str = 'movie', page: int = 1) -> Dict:
        """Get top-rated content."""
        endpoint = f"{content_type}/top_rated"
        params = {
            'language': 'en-US',
            'page': page
        }
        return self._make_request(endpoint, params)

class ContentProcessor:
    def __init__(self, api_client: TMDbAPIClient):
        self.api_client = api_client
    
    def process_content_item(self, item: Dict, content_type: str, get_details: bool = True) -> Dict:
        processed = {
            'id': f"{content_type}_{item['id']}",
            'title': item.get('title') or item.get('name', 'Unknown'),
            'content_type': content_type,
            'overview': item.get('overview', ''),
            'popularity': item.get('popularity', 0.0),
            'rating': item.get('vote_average', 0.0),
            'vote_count': item.get('vote_count', 0),
            'release_date': item.get('release_date') or item.get('first_air_date', ''),
            'poster_url': self._get_image_url(item.get('poster_path')),
            'backdrop_url': self._get_image_url(item.get('backdrop_path')),
            'genres': [],
            'cast': [],
            'directors': [],
            'keywords': [],
            'runtime_min': None,
            'num_episodes': None,
            'url': f"https://www.themoviedb.org/{content_type}/{item['id']}"
        }
        if get_details:
            try:
                details = self.api_client.get_content_details(item['id'], content_type)
                processed.update(self._extract_detailed_info(details, content_type))
            except Exception as e:
                logger.warning(f"Failed to get details for {item['id']}: {e}")
        
        return processed
    
    def _get_image_url(self, path: str) -> str:
        if not path:
            return None
        return f"https://image.tmdb.org/t/p/w500{path}"
    
    def _extract_detailed_info(self, details: Dict, content_type: str) -> Dict:
        info = {}
        if 'genres' in details:
            info['genres'] = [genre['name'].lower() for genre in details['genres']]
        if 'credits' in details:
            credits = details['credits']
            if 'cast' in credits:
                info['cast'] = [person['name'] for person in credits['cast'][:5]]
            if 'crew' in credits:
                directors = [person['name'] for person in credits['crew'] 
                           if person['job'] == 'Director']
                info['directors'] = directors[:3]

        if 'keywords' in details:
            if 'keywords' in details['keywords']:
                info['keywords'] = [kw['name'] for kw in details['keywords']['keywords'][:10]]
        if content_type == 'movie':
            info['runtime_min'] = details.get('runtime')
        else:
            info['runtime_min'] = details.get('episode_run_time', [None])[0] if details.get('episode_run_time') else None
            info['num_episodes'] = details.get('number_of_episodes')
    
        return info
    
    def batch_process_content(self, items: List[Dict], content_type: str, get_details: bool = True) -> List[Dict]:
        processed_items = []
        for item in items:
            try:
                processed = self.process_content_item(item, content_type, get_details)
                processed_items.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process item {item.get('id', 'unknown')}: {e}")
        return processed_items
