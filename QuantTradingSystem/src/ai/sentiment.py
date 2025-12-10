"""
AI预测模块 - 情感分析
从新闻、社交媒体等文本数据中分析市场情绪
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


# ==================== 情感分析配置 ====================

@dataclass
class SentimentConfig:
    """情感分析配置"""
    # 模型配置
    model_name: str = "bert-base-chinese"  # 中文BERT模型
    max_length: int = 512
    
    # 情感分类
    num_classes: int = 3  # 正面/中性/负面
    
    # 关键词配置
    use_lexicon: bool = True
    lexicon_weight: float = 0.3  # 词典情感权重
    
    # 聚合配置
    decay_factor: float = 0.9  # 时间衰减因子
    aggregation_window: int = 24  # 小时


# ==================== 中文金融情感词典 ====================

class FinancialLexicon:
    """
    金融领域情感词典
    """
    
    def __init__(self):
        # 正面词汇
        self.positive_words = {
            # 价格上涨
            '上涨', '涨停', '大涨', '飙升', '暴涨', '走高', '攀升', '冲高',
            '突破', '创新高', '新高', '强势', '领涨', '反弹', '回升',
            # 基本面
            '增长', '盈利', '利好', '利多', '看涨', '看多', '乐观',
            '改善', '提升', '增加', '扩张', '繁荣', '复苏', '回暖',
            # 资金
            '买入', '加仓', '抄底', '净流入', '做多', '增持',
            # 其他
            '超预期', '景气', '强劲', '稳健', '活跃', '火爆',
        }
        
        # 负面词汇
        self.negative_words = {
            # 价格下跌
            '下跌', '跌停', '大跌', '暴跌', '下挫', '走低', '回落',
            '跳水', '崩盘', '暴跌', '滑坡', '创新低', '新低', '弱势',
            # 基本面
            '亏损', '下滑', '利空', '利淡', '看跌', '看空', '悲观',
            '恶化', '下降', '减少', '收缩', '衰退', '萎缩', '疲软',
            # 资金
            '卖出', '减仓', '清仓', '净流出', '做空', '减持', '抛售',
            # 风险
            '风险', '危机', '违约', '暴雷', '爆仓', '停牌', '退市',
        }
        
        # 程度词
        self.degree_words = {
            '非常': 2.0, '极其': 2.0, '十分': 1.8, '特别': 1.8,
            '相当': 1.5, '比较': 1.2, '较为': 1.2, '略微': 0.8,
            '稍微': 0.8, '有点': 0.7, '略有': 0.7,
        }
        
        # 否定词
        self.negation_words = {'不', '没', '无', '非', '未', '别', '勿'}
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        基于词典分析情感
        
        Returns:
            {
                'positive': 正面情感得分,
                'negative': 负面情感得分,
                'sentiment': 综合情感得分 (-1 到 1)
            }
        """
        # 分词（简单按字符处理）
        text = text.lower()
        
        positive_score = 0
        negative_score = 0
        
        # 检查正面词
        for word in self.positive_words:
            count = text.count(word)
            if count > 0:
                # 检查是否有程度词或否定词修饰
                modifier = self._get_modifier(text, word)
                positive_score += count * modifier
        
        # 检查负面词
        for word in self.negative_words:
            count = text.count(word)
            if count > 0:
                modifier = self._get_modifier(text, word)
                negative_score += count * modifier
        
        # 归一化
        total = positive_score + negative_score
        if total > 0:
            positive_score /= total
            negative_score /= total
        
        sentiment = positive_score - negative_score
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'sentiment': sentiment
        }
    
    def _get_modifier(self, text: str, word: str) -> float:
        """获取修饰词的影响"""
        modifier = 1.0
        
        # 查找word前面的文本
        idx = text.find(word)
        if idx > 0:
            prefix = text[max(0, idx-10):idx]
            
            # 检查否定词
            for neg in self.negation_words:
                if neg in prefix:
                    modifier *= -1
                    break
            
            # 检查程度词
            for degree, weight in self.degree_words.items():
                if degree in prefix:
                    modifier *= weight
                    break
        
        return modifier


# ==================== 新闻情感分析器 ====================

class NewsSentimentAnalyzer:
    """
    新闻情感分析器
    分析新闻标题和内容的情感倾向
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.lexicon = FinancialLexicon()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载深度学习模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Sentiment model loaded successfully")
            
        except ImportError:
            logger.warning("transformers not installed, using lexicon only")
            self.model = None
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
            self.model = None
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        分析单条文本的情感
        
        Returns:
            {
                'sentiment': 情感得分 (-1 到 1),
                'confidence': 置信度,
                'positive_prob': 正面概率,
                'negative_prob': 负面概率
            }
        """
        results = {
            'sentiment': 0.0,
            'confidence': 0.0,
            'positive_prob': 0.0,
            'negative_prob': 0.0
        }
        
        if not text or len(text.strip()) == 0:
            return results
        
        # 词典分析
        lexicon_result = self.lexicon.analyze(text)
        
        # 深度学习模型分析
        if self.model is not None:
            try:
                import torch
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                
                # 假设输出顺序: [负面, 中性, 正面]
                negative_prob = probs[0].item()
                neutral_prob = probs[1].item() if len(probs) > 2 else 0
                positive_prob = probs[-1].item()
                
                model_sentiment = positive_prob - negative_prob
                model_confidence = max(probs).item()
                
                # 融合词典和模型结果
                results['sentiment'] = (
                    self.config.lexicon_weight * lexicon_result['sentiment'] +
                    (1 - self.config.lexicon_weight) * model_sentiment
                )
                results['confidence'] = model_confidence
                results['positive_prob'] = positive_prob
                results['negative_prob'] = negative_prob
                
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                results['sentiment'] = lexicon_result['sentiment']
                results['confidence'] = 0.5
        else:
            # 仅使用词典
            results['sentiment'] = lexicon_result['sentiment']
            results['confidence'] = abs(lexicon_result['sentiment'])
            results['positive_prob'] = lexicon_result['positive']
            results['negative_prob'] = lexicon_result['negative']
        
        return results
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """批量分析"""
        return [self.analyze_text(text) for text in texts]
    
    def analyze_news_df(self, df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
        """
        分析新闻DataFrame
        
        Args:
            df: 包含新闻文本的DataFrame
            text_column: 文本列名
        
        Returns:
            添加了情感列的DataFrame
        """
        results = self.analyze_batch(df[text_column].tolist())
        
        df = df.copy()
        df['sentiment'] = [r['sentiment'] for r in results]
        df['sentiment_confidence'] = [r['confidence'] for r in results]
        df['positive_prob'] = [r['positive_prob'] for r in results]
        df['negative_prob'] = [r['negative_prob'] for r in results]
        
        return df


# ==================== 市场情绪聚合器 ====================

class MarketSentimentAggregator:
    """
    市场情绪聚合器
    整合多源情感分析结果
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.news_analyzer = NewsSentimentAnalyzer(config)
        
        # 情感时间序列
        self._sentiment_history: List[Dict] = []
    
    def aggregate_sentiment(
        self, 
        news_data: List[Dict],
        symbol: str = None
    ) -> Dict[str, float]:
        """
        聚合多条新闻的情感
        
        Args:
            news_data: 新闻列表，每条包含 {text, timestamp, source, relevance}
            symbol: 相关标的
        
        Returns:
            聚合后的情感指标
        """
        if not news_data:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'bullish_ratio': 0.5
            }
        
        sentiments = []
        weights = []
        current_time = datetime.now()
        
        for news in news_data:
            # 分析情感
            text = news.get('text', news.get('title', ''))
            result = self.news_analyzer.analyze_text(text)
            
            # 计算权重
            weight = 1.0
            
            # 时间衰减
            news_time = news.get('timestamp')
            if news_time:
                if isinstance(news_time, str):
                    news_time = datetime.fromisoformat(news_time)
                hours_ago = (current_time - news_time).total_seconds() / 3600
                weight *= (self.config.decay_factor ** (hours_ago / 24))
            
            # 相关性权重
            relevance = news.get('relevance', 1.0)
            weight *= relevance
            
            # 来源权重（权威来源权重更高）
            source = news.get('source', '')
            if any(s in source for s in ['官方', '央行', '证监会', '财政部']):
                weight *= 1.5
            elif any(s in source for s in ['研报', '券商', '分析师']):
                weight *= 1.2
            
            sentiments.append(result['sentiment'])
            weights.append(weight)
        
        # 加权平均
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / total_weight
        else:
            weighted_sentiment = 0.0
        
        # 计算看涨比例
        bullish_count = sum(1 for s in sentiments if s > 0)
        bullish_ratio = bullish_count / len(sentiments) if sentiments else 0.5
        
        result = {
            'sentiment': weighted_sentiment,
            'confidence': np.mean([abs(s) for s in sentiments]) if sentiments else 0.0,
            'news_count': len(news_data),
            'bullish_ratio': bullish_ratio,
            'symbol': symbol,
            'timestamp': current_time.isoformat()
        }
        
        # 记录历史
        self._sentiment_history.append(result)
        
        return result
    
    def get_sentiment_signal(self, sentiment: float, threshold: float = 0.3) -> int:
        """
        将情感转换为交易信号
        
        Args:
            sentiment: 情感得分
            threshold: 阈值
        
        Returns:
            1: 看多, -1: 看空, 0: 中性
        """
        if sentiment > threshold:
            return 1
        elif sentiment < -threshold:
            return -1
        else:
            return 0
    
    def get_sentiment_history(
        self, 
        symbol: str = None,
        hours: int = 24
    ) -> pd.DataFrame:
        """获取情感历史"""
        if not self._sentiment_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._sentiment_history)
        
        if symbol:
            df = df[df['symbol'] == symbol]
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - pd.Timedelta(hours=hours)
            df = df[df['timestamp'] >= cutoff]
        
        return df


# ==================== 舆情监控器 ====================

class SentimentMonitor:
    """
    舆情监控器
    实时监控市场情绪变化
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or []
        self.aggregator = MarketSentimentAggregator()
        
        # 情感状态
        self._current_sentiment: Dict[str, float] = {}
        self._sentiment_alerts: List[Dict] = []
    
    def update(self, symbol: str, news_data: List[Dict]):
        """更新情感数据"""
        result = self.aggregator.aggregate_sentiment(news_data, symbol)
        
        old_sentiment = self._current_sentiment.get(symbol, 0.0)
        new_sentiment = result['sentiment']
        
        self._current_sentiment[symbol] = new_sentiment
        
        # 检测情感突变
        if abs(new_sentiment - old_sentiment) > 0.5:
            self._sentiment_alerts.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'old_sentiment': old_sentiment,
                'new_sentiment': new_sentiment,
                'change': new_sentiment - old_sentiment,
                'alert_type': 'sentiment_shift'
            })
            logger.warning(f"Sentiment shift for {symbol}: {old_sentiment:.2f} -> {new_sentiment:.2f}")
    
    def get_current_sentiment(self, symbol: str = None) -> Dict[str, float]:
        """获取当前情感"""
        if symbol:
            return {symbol: self._current_sentiment.get(symbol, 0.0)}
        return self._current_sentiment.copy()
    
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """获取情感警报"""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        return [
            alert for alert in self._sentiment_alerts
            if alert['timestamp'] >= cutoff
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取情感摘要"""
        if not self._current_sentiment:
            return {'overall': 0.0, 'bullish_symbols': [], 'bearish_symbols': []}
        
        overall = np.mean(list(self._current_sentiment.values()))
        bullish = [s for s, v in self._current_sentiment.items() if v > 0.3]
        bearish = [s for s, v in self._current_sentiment.items() if v < -0.3]
        
        return {
            'overall': overall,
            'bullish_symbols': bullish,
            'bearish_symbols': bearish,
            'symbol_count': len(self._current_sentiment),
            'last_update': datetime.now().isoformat()
        }
