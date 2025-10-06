import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os
from abc import ABC
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

class RecommendationDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, user_ids: Optional[List[str]] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.user_ids = user_ids or []
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = { 'features': self.features[idx], 'target': self.targets[idx] }
        if self.user_ids:
            sample['user_id'] = self.user_ids[idx]
        return sample

class BaseRecommendationModel(nn.Module, ABC):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def save_model(self, path: str):
        torch.save({'model_state_dict': self.state_dict(),'input_dim': self.input_dim, 'hidden_dim': self.hidden_dim,'dropout': self.dropout}, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self

class DeepRecommendationModel(BaseRecommendationModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3, num_layers: int = 3):
        super().__init__(input_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()

class WideAndDeepModel(BaseRecommendationModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__(input_dim, hidden_dim, dropout)
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.combine = nn.Linear(2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return torch.sigmoid(self.combine(combined)).squeeze()

class AttentionRecommendationModel(BaseRecommendationModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3, num_heads: int = 4):
        super().__init__(input_dim, hidden_dim, dropout)
        max_heads = min(num_heads, input_dim)
        for i in range(max_heads, 0, -1):
            if input_dim % i == 0:
                self.num_heads = i
                break
        else:
            self.num_heads = 1
        if input_dim % self.num_heads != 0:
            self.projection_dim = ((input_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
            self.input_projection = nn.Linear(input_dim, self.projection_dim)
        else:
            self.projection_dim = input_dim
            self.input_projection = None
            
        self.attention = nn.MultiheadAttention(embed_dim=self.projection_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True)
        self.feature_processor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_projection is not None:
            x = self.input_projection(x)
        x_reshaped = x.unsqueeze(1)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.squeeze(1)
        processed = self.feature_processor(attn_out)
        return self.output_layers(processed).squeeze()

class NeuralRecommendationTrainer:
    def __init__(self, model: BaseRecommendationModel, learning_rate: float = 0.001, weight_decay: float = 1e-5, device: str = 'auto'):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(features)
            predictions = predictions.squeeze()
            targets = targets.squeeze()
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            mae = F.l1_loss(predictions, targets).item()
            total_mae += mae
            num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)
                predictions = self.model(features)
                predictions = predictions.squeeze()
                targets = targets.squeeze()
                loss = self.criterion(predictions, targets)
                mae = F.l1_loss(predictions, targets).item()
                total_loss += loss.item()
                total_mae += mae
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100, early_stopping_patience: int = 10, save_best: bool = True, model_path: str = 'best_model.pth') -> Dict[str, List[float]]:
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_mae = self.train_epoch(train_loader)

            val_loss, val_mae = self.validate(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.model.save_model(model_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self.history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        features_tensor = torch.FloatTensor(features).to(self.device)
        with torch.no_grad():
            predictions = self.model(features_tensor)
        return predictions.cpu().numpy()

class NeuralRecommendationSystem:
    def __init__(self, model_type: str = 'deep', input_dim: int = None, hidden_dim: int = 128, dropout: float = 0.3, learning_rate: float = 0.001):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.trainer = None
        
        if input_dim is not None:
            if model_type == 'deep':
                self.model = DeepRecommendationModel(input_dim, hidden_dim, dropout)
            elif model_type == 'wide_deep':
                self.model = WideAndDeepModel(input_dim, hidden_dim, dropout)
            elif model_type == 'attention':
                self.model = AttentionRecommendationModel(input_dim, hidden_dim, dropout)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.trainer = NeuralRecommendationTrainer(self.model, learning_rate=learning_rate)
        
        self.scaler = StandardScaler()
        self.fitted = False
    
    def prepare_data(self, features: np.ndarray, targets: np.ndarray, test_size: float = 0.2, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        features_scaled = self.scaler.fit_transform(features)
        X_train, X_val, y_train, y_val = train_test_split(features_scaled, targets, test_size=test_size, random_state=42)
        train_dataset = RecommendationDataset(X_train, y_train)
        val_dataset = RecommendationDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    def train(self, features: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 32, test_size: float = 0.2, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        if self.input_dim is None or self.model is None:
            self.input_dim = features.shape[1]
            if self.model_type == 'deep':
                self.model = DeepRecommendationModel(self.input_dim, self.hidden_dim, self.dropout)
            elif self.model_type == 'wide_deep':
                self.model = WideAndDeepModel(self.input_dim, self.hidden_dim, self.dropout)
            elif self.model_type == 'attention':
                self.model = AttentionRecommendationModel(self.input_dim, self.hidden_dim, self.dropout)
            else:
                self.model = DeepRecommendationModel(self.input_dim, self.hidden_dim, self.dropout)
            self.trainer = NeuralRecommendationTrainer(self.model, self.learning_rate)
            
        train_loader, val_loader = self.prepare_data(features, targets, test_size, batch_size)
        history = self.trainer.train(train_loader, val_loader, epochs, early_stopping_patience)
        
        self.fitted = True
        return history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained first")
        features_scaled = self.scaler.transform(features)
        return self.trainer.predict(features_scaled)
    
    def get_feature_importance(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained first")
        
        features_tensor = torch.FloatTensor(features).requires_grad_(True)
        self.model.eval()
        output = self.model(features_tensor)
        gradients = torch.autograd.grad(output.sum(), features_tensor, retain_graph=True)[0]
        return torch.abs(gradients).detach().numpy()
    
    def save_system(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, 'model.pth'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        config = {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    def load_system(self, path: str):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        if config['model_type'] == 'deep':
            self.model = DeepRecommendationModel(
                config['input_dim'], config['hidden_dim'], config['dropout'])
        elif config['model_type'] == 'wide_deep':
            self.model = WideAndDeepModel(config['input_dim'], config['hidden_dim'], config['dropout'])
        elif config['model_type'] == 'attention':
            self.model = AttentionRecommendationModel(config['input_dim'], config['hidden_dim'], config['dropout'])
        
        self.model.load_model(os.path.join(path, 'model.pth'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.trainer = NeuralRecommendationTrainer(self.model, config['learning_rate'])
        self.fitted = True
