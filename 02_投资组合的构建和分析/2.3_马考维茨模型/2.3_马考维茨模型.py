import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.optimize import minimize

class MarkowitzModel:
    """
    马考维茨模型类，用于优化投资组合

    Attributes:
        returns (pd.DataFrame): 各资产的历史回报率数据
        covariance_matrix (np.ndarray): 资产之间的协方差矩阵
        mean_returns (np.ndarray): 各资产的平均回报率
    """

    def __init__(self, returns: pd.DataFrame):
        """
        初始化马考维茨模型

        Args:
            returns (pd.DataFrame): 各资产的历史回报率数据
        """
        self.returns = returns
        self.covariance_matrix = returns.cov()
        self.mean_returns = returns.mean()

    def calculate_portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        计算投资组合的预期回报和风险（标准差）

        Args:
            weights (np.ndarray): 投资组合中各资产的权重

        Returns:
            Tuple[float, float]: 投资组合的预期回报和风险（标准差）
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        return portfolio_return, portfolio_std_dev

    def generate_random_portfolios(self, num_portfolios: int) -> pd.DataFrame:
        """
        生成随机投资组合，计算其预期回报和风险

        Args:
            num_portfolios (int): 要生成的随机投资组合数量

        Returns:
            pd.DataFrame: 包含随机投资组合的预期回报、风险和权重的数据框
        """
        results = np.zeros((num_portfolios, 3 + len(self.returns.columns)))

        for i in range(num_portfolios):
            weights = np.random.random(len(self.returns.columns))
            weights /= np.sum(weights)
            portfolio_return, portfolio_std_dev = self.calculate_portfolio_performance(weights)
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_std_dev
            results[i, 2] = portfolio_return / portfolio_std_dev  # 夏普比率
            results[i, 3:] = weights

        columns = ['Return', 'Risk', 'Sharpe Ratio'] + [f'Weight {symbol}' for symbol in self.returns.columns]
        return pd.DataFrame(results, columns=columns)

    def optimize_portfolio(self, risk_free_rate: float = 0.0) -> Tuple[np.ndarray, float, float]:
        """
        使用夏普比率优化投资组合

        Args:
            risk_free_rate (float): 无风险利率，默认为0

        Returns:
            Tuple[np.ndarray, float, float]: 最优投资组合的权重、预期回报和风险（标准差）
        """
        num_assets = len(self.returns.columns)
        args = (self.mean_returns, self.covariance_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))

        result = minimize(self.neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x, *self.calculate_portfolio_performance(result.x)

    def neg_sharpe_ratio(self, weights: np.ndarray, mean_returns: np.ndarray, covariance_matrix: np.ndarray, risk_free_rate: float) -> float:
        """
        计算负的夏普比率（用于优化）

        Args:
            weights (np.ndarray): 投资组合中各资产的权重
            mean_returns (np.ndarray): 各资产的平均回报率
            covariance_matrix (np.ndarray): 资产之间的协方差矩阵
            risk_free_rate (float): 无风险利率

        Returns:
            float: 负的夏普比率
        """
        p_ret, p_var = self.calculate_portfolio_performance(weights)
        return -(p_ret - risk_free_rate) / p_var

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = {
        'AAPL': np.random.normal(0.001, 0.01, 1000),
        'MSFT': np.random.normal(0.001, 0.01, 1000),
        'GOOGL': np.random.normal(0.001, 0.01, 1000),
    }
    returns = pd.DataFrame(data)
    
    # 初始化马考维茨模型
    model = MarkowitzModel(returns)
    
    # 生成随机投资组合并输出
    portfolios = model.generate_random_portfolios(5000)
    print(portfolios.head())
    
    # 优化投资组合并输出
    optimal_weights, expected_return, expected_risk = model.optimize_portfolio()
    print("Optimal Weights:", optimal_weights)
    print("Expected Return:", expected_return)
    print("Expected Risk:", expected_risk)