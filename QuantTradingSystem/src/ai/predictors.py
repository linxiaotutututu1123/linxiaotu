"""
AI预测模块 - 深度学习模型
包含LSTM、Transformer等时序预测模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== 模型配置 ====================

@dataclass
class ModelConfig:
    """模型配置"""
    # 输入配置
    sequence_length: int = 60        # 输入序列长度
    feature_dim: int = 10            # 特征维度
    
    # 模型配置
    hidden_size: int = 128           # 隐藏层大小
    num_layers: int = 2              # 层数
    dropout: float = 0.2             # Dropout比例
    
    # 训练配置
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # 输出配置
    prediction_horizon: int = 1      # 预测时间跨度
    output_dim: int = 1              # 输出维度


# ==================== 特征工程 ====================

class FeatureEngineer:
    """
    特征工程
    从原始OHLCV数据中提取预测特征
    """
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self._scalers: Dict[str, Any] = {}
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        提取特征
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            特征数组 shape: (samples, features)
        """
        features = pd.DataFrame(index=df.index)
        
        # 价格特征
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 波动率特征
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # 动量特征
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # 布林带位置
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - sma20) / (2 * std20)
        
        # 成交量特征
        if 'volume' in df.columns:
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 高低价特征
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 时间特征
        if isinstance(df.index, pd.DatetimeIndex):
            features['day_of_week'] = df.index.dayofweek / 6
            features['hour'] = df.index.hour / 23 if hasattr(df.index, 'hour') else 0
        
        # 删除缺失值
        features = features.dropna()
        
        return features.values
    
    def create_sequences(
        self, 
        features: np.ndarray, 
        target: np.ndarray,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建训练序列
        
        Args:
            features: 特征数组
            target: 目标数组
            sequence_length: 序列长度
        
        Returns:
            (X, y) 训练数据
        """
        seq_len = sequence_length or self.sequence_length
        
        X, y = [], []
        for i in range(seq_len, len(features)):
            X.append(features[i-seq_len:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def normalize(self, data: np.ndarray, feature_name: str = "default") -> np.ndarray:
        """
        数据标准化
        """
        try:
            from sklearn.preprocessing import StandardScaler
            
            if feature_name not in self._scalers:
                self._scalers[feature_name] = StandardScaler()
                return self._scalers[feature_name].fit_transform(data)
            else:
                return self._scalers[feature_name].transform(data)
        except ImportError:
            # 简单标准化
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1
            return (data - mean) / std
    
    def inverse_normalize(self, data: np.ndarray, feature_name: str = "default") -> np.ndarray:
        """
        反标准化
        """
        try:
            if feature_name in self._scalers:
                return self._scalers[feature_name].inverse_transform(data)
        except:
            pass
        return data


# ==================== 模型基类 ====================

class BasePredictor(ABC):
    """预测模型基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def build_model(self):
        """构建模型"""
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    def save(self, path: str):
        """保存模型"""
        pass
    
    def load(self, path: str):
        """加载模型"""
        pass


# ==================== LSTM模型 ====================

class LSTMPredictor(BasePredictor):
    """
    LSTM预测模型
    适用于时序价格预测
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__(config or ModelConfig())
        self._history = None
    
    def build_model(self):
        """构建LSTM模型"""
        try:
            import torch
            import torch.nn as nn
            
            class LSTMModel(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
                    super().__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    
                    self.lstm = nn.LSTM(
                        input_dim, 
                        hidden_dim, 
                        num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0
                    )
                    
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, output_dim)
                    )
                
                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    lstm_out, _ = self.lstm(x)
                    # 取最后一个时间步
                    out = self.fc(lstm_out[:, -1, :])
                    return out
            
            self.model = LSTMModel(
                input_dim=self.config.feature_dim,
                hidden_dim=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_dim=self.config.output_dim,
                dropout=self.config.dropout
            )
            
            logger.info("LSTM model built successfully")
            
        except ImportError:
            logger.warning("PyTorch not installed, using simple model")
            self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """训练LSTM模型"""
        if self.model is None:
            self.build_model()
        
        if self.model is None:
            logger.warning("Model not available, skipping training")
            return
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            
            # 划分训练集和验证集
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # 转换为Tensor
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            
            # 早停
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                self.model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val).item()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs}, "
                               f"Train Loss: {train_loss/len(train_loader):.6f}, "
                               f"Val Loss: {val_loss:.6f}")
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.is_trained = True
            logger.info("LSTM training completed")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None or not self.is_trained:
            logger.warning("Model not trained")
            return np.zeros(len(X))
        
        try:
            import torch
            
            device = next(self.model.parameters()).device
            self.model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                predictions = self.model(X_tensor)
                return predictions.cpu().numpy().squeeze()
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.zeros(len(X))


# ==================== Transformer模型 ====================

class TransformerPredictor(BasePredictor):
    """
    Transformer预测模型
    利用注意力机制捕捉长期依赖
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__(config or ModelConfig())
    
    def build_model(self):
        """构建Transformer模型"""
        try:
            import torch
            import torch.nn as nn
            
            class PositionalEncoding(nn.Module):
                def __init__(self, d_model, max_len=5000):
                    super().__init__()
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
                    )
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    pe = pe.unsqueeze(0)
                    self.register_buffer('pe', pe)
                
                def forward(self, x):
                    return x + self.pe[:, :x.size(1)]
            
            class TransformerModel(nn.Module):
                def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout):
                    super().__init__()
                    self.input_projection = nn.Linear(input_dim, d_model)
                    self.pos_encoder = PositionalEncoding(d_model)
                    
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                    
                    self.fc = nn.Sequential(
                        nn.Linear(d_model, d_model // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model // 2, output_dim)
                    )
                
                def forward(self, x):
                    x = self.input_projection(x)
                    x = self.pos_encoder(x)
                    x = self.transformer(x)
                    # 使用最后一个时间步
                    out = self.fc(x[:, -1, :])
                    return out
            
            self.model = TransformerModel(
                input_dim=self.config.feature_dim,
                d_model=self.config.hidden_size,
                nhead=8,
                num_layers=self.config.num_layers,
                output_dim=self.config.output_dim,
                dropout=self.config.dropout
            )
            
            logger.info("Transformer model built successfully")
            
        except ImportError:
            logger.warning("PyTorch not installed")
            self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """训练（与LSTM类似）"""
        if self.model is None:
            self.build_model()
        
        if self.model is None:
            return
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
            
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                self.model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val).item()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs}, "
                               f"Train Loss: {train_loss/len(train_loader):.6f}, "
                               f"Val Loss: {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.is_trained = True
            logger.info("Transformer training completed")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None or not self.is_trained:
            return np.zeros(len(X))
        
        try:
            import torch
            
            device = next(self.model.parameters()).device
            self.model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                predictions = self.model(X_tensor)
                return predictions.cpu().numpy().squeeze()
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.zeros(len(X))


# ==================== 集成预测器 ====================

class EnsemblePredictor:
    """
    集成预测器
    组合多个模型的预测结果
    """
    
    def __init__(self, models: List[BasePredictor], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        
        # 验证权重
        assert len(self.weights) == len(self.models)
        assert abs(sum(self.weights) - 1.0) < 0.01
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """训练所有模型"""
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            model.train(X, y, validation_split)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """加权平均预测"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def update_weights(self, X: np.ndarray, y_true: np.ndarray):
        """根据验证集表现更新权重"""
        errors = []
        
        for model in self.models:
            pred = model.predict(X)
            mse = np.mean((pred - y_true) ** 2)
            errors.append(mse)
        
        # 误差越小，权重越大
        inv_errors = [1 / (e + 1e-6) for e in errors]
        total = sum(inv_errors)
        self.weights = [w / total for w in inv_errors]
        
        logger.info(f"Updated weights: {self.weights}")


# ==================== 预测管理器 ====================

class PredictionManager:
    """
    预测管理器
    管理模型训练、预测和评估
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.feature_engineer = FeatureEngineer(config.sequence_length if config else 60)
        self.predictor: Optional[BasePredictor] = None
        
        # 预测历史
        self._prediction_history: List[Dict] = []
    
    def train_model(
        self, 
        df: pd.DataFrame, 
        model_type: str = "lstm",
        target_column: str = "returns"
    ):
        """
        训练模型
        
        Args:
            df: OHLCV数据
            model_type: 模型类型 (lstm / transformer / ensemble)
            target_column: 目标列
        """
        # 提取特征
        features = self.feature_engineer.extract_features(df)
        
        # 计算目标（未来收益率）
        if target_column == "returns":
            target = df['close'].pct_change().shift(-1).dropna()
        elif target_column == "direction":
            target = (df['close'].shift(-1) > df['close']).astype(int)
        else:
            target = df[target_column].values
        
        # 对齐长度
        min_len = min(len(features), len(target))
        features = features[-min_len:]
        target = target[-min_len:].values
        
        # 标准化
        features = self.feature_engineer.normalize(features, "features")
        
        # 创建序列
        X, y = self.feature_engineer.create_sequences(features, target)
        
        # 更新配置
        self.config.feature_dim = features.shape[1]
        
        # 选择模型
        if model_type == "lstm":
            self.predictor = LSTMPredictor(self.config)
        elif model_type == "transformer":
            self.predictor = TransformerPredictor(self.config)
        elif model_type == "ensemble":
            self.predictor = EnsemblePredictor([
                LSTMPredictor(self.config),
                TransformerPredictor(self.config)
            ])
        
        # 训练
        self.predictor.train(X, y)
        
        logger.info(f"Model trained on {len(X)} samples")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        生成预测
        """
        if self.predictor is None or not self.predictor.is_trained:
            logger.warning("Model not trained")
            return np.array([])
        
        # 提取特征
        features = self.feature_engineer.extract_features(df)
        features = self.feature_engineer.normalize(features, "features")
        
        # 创建序列
        X = []
        for i in range(self.config.sequence_length, len(features)):
            X.append(features[i-self.config.sequence_length:i])
        X = np.array(X)
        
        if len(X) == 0:
            return np.array([])
        
        # 预测
        predictions = self.predictor.predict(X)
        
        # 记录预测
        self._prediction_history.append({
            "timestamp": datetime.now(),
            "predictions": predictions.tolist()
        })
        
        return predictions
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        评估预测性能
        """
        # 实际收益率
        actual = df['close'].pct_change().values[-len(predictions):]
        
        # 去除NaN
        mask = ~np.isnan(actual) & ~np.isnan(predictions)
        actual = actual[mask]
        predictions = predictions[mask]
        
        if len(actual) == 0:
            return {}
        
        metrics = {}
        
        # MSE
        metrics['mse'] = np.mean((actual - predictions) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 方向准确率
        direction_actual = (actual > 0).astype(int)
        direction_pred = (predictions > 0).astype(int)
        metrics['direction_accuracy'] = np.mean(direction_actual == direction_pred)
        
        # 相关系数
        if np.std(predictions) > 0 and np.std(actual) > 0:
            metrics['correlation'] = np.corrcoef(actual, predictions)[0, 1]
        else:
            metrics['correlation'] = 0
        
        # 信息系数(IC)
        metrics['ic'] = metrics['correlation']
        
        return metrics
