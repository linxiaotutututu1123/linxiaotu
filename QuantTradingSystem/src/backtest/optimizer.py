"""
回测引擎 - 参数优化器
支持网格搜索、随机搜索、贝叶斯优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import warnings

from ..data.data_structures import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    param_type: str  # "int", "float", "choice"
    low: float = 0
    high: float = 100
    step: float = 1
    choices: List[Any] = None
    
    def sample_grid(self) -> List[Any]:
        """生成网格搜索的参数值"""
        if self.param_type == "choice":
            return self.choices
        elif self.param_type == "int":
            return list(range(int(self.low), int(self.high) + 1, int(self.step)))
        else:  # float
            return list(np.arange(self.low, self.high + self.step, self.step))
    
    def sample_random(self) -> Any:
        """随机采样"""
        if self.param_type == "choice":
            return np.random.choice(self.choices)
        elif self.param_type == "int":
            return np.random.randint(int(self.low), int(self.high) + 1)
        else:  # float
            return np.random.uniform(self.low, self.high)


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: PerformanceMetrics
    all_results: List[Dict[str, Any]]
    optimization_time: float


class ParameterOptimizer:
    """
    参数优化器
    支持多种优化算法
    """
    
    def __init__(
        self,
        objective: Callable[[Dict[str, Any]], Tuple[float, PerformanceMetrics]],
        parameter_spaces: List[ParameterSpace],
        maximize: bool = True,
        n_jobs: int = -1
    ):
        """
        Args:
            objective: 目标函数，输入参数字典，返回(分数, 绩效指标)
            parameter_spaces: 参数空间列表
            maximize: True=最大化目标，False=最小化目标
            n_jobs: 并行任务数，-1表示使用全部CPU
        """
        self.objective = objective
        self.parameter_spaces = parameter_spaces
        self.maximize = maximize
        self.n_jobs = n_jobs if n_jobs > 0 else None
        
        self._results: List[Dict[str, Any]] = []
    
    def grid_search(self) -> OptimizationResult:
        """
        网格搜索
        遍历所有参数组合
        """
        logger.info("Starting grid search optimization...")
        start_time = datetime.now()
        
        # 生成所有参数组合
        param_names = [p.name for p in self.parameter_spaces]
        param_values = [p.sample_grid() for p in self.parameter_spaces]
        all_combinations = list(product(*param_values))
        
        logger.info(f"Total combinations: {len(all_combinations)}")
        
        results = []
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        best_metrics = None
        
        # 执行搜索
        for i, values in enumerate(all_combinations):
            params = dict(zip(param_names, values))
            
            try:
                score, metrics = self.objective(params)
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                # 更新最优
                if self.maximize:
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(all_combinations)}, Best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error evaluating params {params}: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Grid search completed in {elapsed:.2f}s")
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=results,
            optimization_time=elapsed
        )
    
    def random_search(self, n_iter: int = 100) -> OptimizationResult:
        """
        随机搜索
        随机采样n_iter次
        """
        logger.info(f"Starting random search optimization ({n_iter} iterations)...")
        start_time = datetime.now()
        
        results = []
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        best_metrics = None
        
        for i in range(n_iter):
            # 随机采样参数
            params = {p.name: p.sample_random() for p in self.parameter_spaces}
            
            try:
                score, metrics = self.objective(params)
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                # 更新最优
                if self.maximize:
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Progress: {i+1}/{n_iter}, Best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error evaluating params {params}: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Random search completed in {elapsed:.2f}s")
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=results,
            optimization_time=elapsed
        )
    
    def bayesian_optimization(self, n_iter: int = 100, n_initial: int = 10) -> OptimizationResult:
        """
        贝叶斯优化
        使用高斯过程作为代理模型
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from scipy.optimize import minimize
            from scipy.stats import norm
        except ImportError:
            logger.warning("sklearn not installed, falling back to random search")
            return self.random_search(n_iter)
        
        logger.info(f"Starting Bayesian optimization ({n_iter} iterations)...")
        start_time = datetime.now()
        
        # 参数空间边界
        bounds = []
        param_names = []
        for p in self.parameter_spaces:
            if p.param_type == "choice":
                bounds.append((0, len(p.choices) - 1))
            else:
                bounds.append((p.low, p.high))
            param_names.append(p.name)
        
        bounds = np.array(bounds)
        
        # 初始采样
        X_observed = []
        y_observed = []
        results = []
        
        logger.info(f"Initial sampling ({n_initial} points)...")
        for _ in range(n_initial):
            x = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
            X_observed.append(x)
            
            params = self._x_to_params(x, param_names)
            
            try:
                score, metrics = self.objective(params)
                y_observed.append(score if self.maximize else -score)
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
            except Exception as e:
                y_observed.append(float('-inf'))
                logger.warning(f"Error: {e}")
        
        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)
        
        # 高斯过程优化
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        best_metrics = None
        
        for i in range(n_iter - n_initial):
            # 拟合高斯过程
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_observed, y_observed)
            
            # 找到下一个采样点（最大化采集函数）
            x_next = self._acquisition_max(gp, bounds, y_observed.max())
            
            # 评估
            params = self._x_to_params(x_next, param_names)
            
            try:
                score, metrics = self.objective(params)
                y_next = score if self.maximize else -score
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                # 更新观测
                X_observed = np.vstack([X_observed, x_next])
                y_observed = np.append(y_observed, y_next)
                
                # 更新最优
                actual_score = score
                if self.maximize:
                    if actual_score > best_score:
                        best_score = actual_score
                        best_params = params
                        best_metrics = metrics
                else:
                    if actual_score < best_score:
                        best_score = actual_score
                        best_params = params
                        best_metrics = metrics
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {n_initial + i + 1}/{n_iter}, Best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Bayesian optimization completed in {elapsed:.2f}s")
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_results=results,
            optimization_time=elapsed
        )
    
    def _x_to_params(self, x: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
        """将数值向量转换为参数字典"""
        params = {}
        for i, (name, p) in enumerate(zip(param_names, self.parameter_spaces)):
            if p.param_type == "choice":
                idx = int(np.clip(round(x[i]), 0, len(p.choices) - 1))
                params[name] = p.choices[idx]
            elif p.param_type == "int":
                params[name] = int(round(x[i]))
            else:
                params[name] = float(x[i])
        return params
    
    def _acquisition_max(
        self, 
        gp, 
        bounds: np.ndarray, 
        y_best: float,
        n_restarts: int = 25
    ) -> np.ndarray:
        """
        最大化采集函数（Expected Improvement）
        """
        from scipy.optimize import minimize
        from scipy.stats import norm
        
        def ei(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            
            if sigma == 0:
                return 0
            
            z = (mu - y_best) / sigma
            ei_value = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei_value  # 负号因为minimize
        
        best_x = None
        best_ei = float('inf')
        
        for _ in range(n_restarts):
            x0 = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
            
            try:
                result = minimize(
                    ei,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x
            except:
                pass
        
        if best_x is None:
            best_x = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
        
        return best_x


# ==================== 过拟合检测 ====================

class OverfittingDetector:
    """
    过拟合检测器
    分析策略参数敏感性和泛化能力
    """
    
    def __init__(self, sensitivity_threshold: float = 0.30):
        """
        Args:
            sensitivity_threshold: 参数敏感性阈值
        """
        self.sensitivity_threshold = sensitivity_threshold
    
    def analyze_sensitivity(
        self,
        optimization_results: List[Dict[str, Any]],
        target_metric: str = "sharpe_ratio"
    ) -> Dict[str, float]:
        """
        分析参数敏感性
        
        Args:
            optimization_results: 优化结果列表
            target_metric: 目标指标
        
        Returns:
            各参数的敏感性分数
        """
        if len(optimization_results) < 10:
            logger.warning("Not enough results for sensitivity analysis")
            return {}
        
        # 提取参数和分数
        params_list = [r['params'] for r in optimization_results]
        scores = [r['score'] for r in optimization_results]
        
        if not params_list:
            return {}
        
        param_names = list(params_list[0].keys())
        sensitivity = {}
        
        for param_name in param_names:
            # 计算参数变化与分数变化的相关性
            param_values = [p.get(param_name, 0) for p in params_list]
            
            # 标准化
            param_std = np.std(param_values)
            score_std = np.std(scores)
            
            if param_std > 0 and score_std > 0:
                correlation = np.corrcoef(param_values, scores)[0, 1]
                sensitivity[param_name] = abs(correlation)
            else:
                sensitivity[param_name] = 0.0
        
        return sensitivity
    
    def check_overfitting(
        self,
        in_sample_metrics: PerformanceMetrics,
        out_sample_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """
        检查过拟合
        对比样本内和样本外表现
        
        Args:
            in_sample_metrics: 样本内绩效
            out_sample_metrics: 样本外绩效
        
        Returns:
            过拟合检测报告
        """
        report = {
            "is_overfitted": False,
            "reasons": [],
            "degradation": {}
        }
        
        # 检查各指标衰减
        metrics_to_check = [
            ("sharpe_ratio", 0.50),      # 夏普衰减超过50%
            ("annual_return", 0.50),     # 收益衰减超过50%
            ("win_rate", 0.20),          # 胜率衰减超过20%
        ]
        
        for metric_name, threshold in metrics_to_check:
            in_value = getattr(in_sample_metrics, metric_name, 0)
            out_value = getattr(out_sample_metrics, metric_name, 0)
            
            if in_value != 0:
                degradation = (in_value - out_value) / abs(in_value)
            else:
                degradation = 0
            
            report["degradation"][metric_name] = degradation
            
            if degradation > threshold:
                report["is_overfitted"] = True
                report["reasons"].append(
                    f"{metric_name} degraded by {degradation:.1%} (threshold: {threshold:.0%})"
                )
        
        # 检查最大回撤增加
        dd_increase = out_sample_metrics.max_drawdown - in_sample_metrics.max_drawdown
        if dd_increase > 0.10:  # 回撤增加超过10%
            report["is_overfitted"] = True
            report["reasons"].append(
                f"Max drawdown increased by {dd_increase:.1%}"
            )
        report["degradation"]["max_drawdown_increase"] = dd_increase
        
        return report
    
    def suggest_improvements(self, report: Dict[str, Any]) -> List[str]:
        """
        根据过拟合报告提供改进建议
        """
        suggestions = []
        
        if not report.get("is_overfitted"):
            return ["策略泛化能力良好，继续监控"]
        
        for reason in report.get("reasons", []):
            if "sharpe" in reason.lower():
                suggestions.append("增加回测时间跨度，覆盖多种市场状态")
                suggestions.append("减少策略参数数量，降低复杂度")
            
            if "return" in reason.lower():
                suggestions.append("检查是否存在数据窥探(data snooping)")
                suggestions.append("使用Walk-forward分析验证策略稳定性")
            
            if "drawdown" in reason.lower():
                suggestions.append("加强风险控制，降低仓位")
                suggestions.append("增加止损机制")
        
        # 通用建议
        suggestions.extend([
            "确保至少200次以上交易样本",
            "使用蒙特卡洛模拟评估策略鲁棒性",
            "考虑使用正则化或参数约束"
        ])
        
        return list(set(suggestions))  # 去重
