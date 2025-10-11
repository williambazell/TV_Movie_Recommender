# Advanced Recommendation System

A sophisticated movie and TV show recommendation engine that demonstrates advanced machine learning techniques and real-world application development. This system showcases expertise in neural networks, clustering algorithms, feature engineering, and scalable architecture design.

The application uses multiple ML approaches working together: KNN clustering for user segmentation, deep neural networks for preference learning, and advanced embedding techniques for content understanding. It processes real movie/TV data from TMDb API, learns from user interactions, and provides personalized recommendations with explanations.

Built with production-ready features including caching, logging, configuration management, and interactive modes for demonstration. The system adapts to user preferences in real-time and handles diverse content types with intelligent filtering and diversity optimization.

## Install and Run

### Prerequisites
- Python 3.8+
- TMDb API key (see https://developer.themoviedb.org/docs/getting-started or email williambazell@yahoo.com)

### Install dependencies
```bash
pip install -r requirements.txt
```
### Edit .env with your TMDb API key
```bash
cp env.example .env
```
Paste TMBD API key where prompted

## Running the Application

**Interactive Mode (Recommended)**
```bash
python main.py --mode interactive
```
Full-featured mode with user profiling, content discovery, and interactive recommendation system. Choose from Fast, Balanced, Full, or Custom configurations.

**Demo Mode**
```bash
python main.py --mode demo
```
Quick demonstration with simulated user interactions and basic recommendations.

**Save/Load System State**
```bash
python main.py --mode interactive --save
python main.py --mode interactive --load
```
Save trained models and user data, or load existing system state.

## ML Techniques in Recommendation App
- **KNN Clustering**: user segmentation and content discovery
- **Neural Networks**: deep learning models for recommendation scoring
- **Advanced Embeddings**: Multiple embedding strategies (Sentence Transformers, TF-IDF, Custom)
- **Feature Engineering**: Comprehensive feature extraction and combination

### Advanced/Specific Techniques
- **User Profiling**: Behavioral analysis and preference evolution tracking
- **Content Discovery**: Multi-modal content analysis and similarity matching
- **Diversity & Novelty**: Intelligent recommendation diversification
- **Real-time Adaptation**: Dynamic preference learning and model updates

### API Logging Control

- **Default**: API calls are silent (no individual API call logs)
- **Verbose API** (`--verbose-api`): Shows detailed API call logs for debugging
- **Combined**: Use `--verbose --verbose-api` for full debugging mode
