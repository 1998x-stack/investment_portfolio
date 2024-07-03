import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple


class PortfolioOptimizer:
    """投资组合优化器类，用于管理和优化投资组合。

    该类使用现代投资组合理论，通过最小化风险或最大化收益来优化投资组合。
    
    Attributes:
        returns (np.ndarray): 每个资产的预期回报率。
        cov_matrix (np.ndarray): 资产的协方差矩阵。
    """

    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray) -> None:
        """
        初始化 PortfolioOptimizer 实例。

        Args:
            returns (np.ndarray): 每个资产的预期回报率。
            cov_matrix (np.ndarray): 资产的协方差矩阵。
        """
        self.returns = returns
        self.cov_matrix = cov_matrix

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        计算投资组合的预期回报率和风险（标准差）。

        Args:
            weights (np.ndarray): 投资组合中每个资产的权重。

        Returns:
            Tuple[float, float]: 投资组合的预期回报率和风险。
        """
        expected_return = np.dot(weights, self.returns)
        expected_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return expected_return, expected_risk

    def minimize_risk(self, target_return: float) -> np.ndarray:
        """
        在给定目标回报率下，最小化投资组合的风险。

        Args:
            target_return (float): 目标回报率。

        Returns:
            np.ndarray: 最优投资组合的权重。
        """
        num_assets = len(self.returns)
        args = (self.returns, self.cov_matrix)

        def risk(weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> float:
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = ({'type': 'eq', 'fun': lambda weights: np.dot(weights, returns) - target_return},
                       {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        result = minimize(risk, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def maximize_return(self, risk_tolerance: float) -> np.ndarray:
        """
        在给定风险容忍度下，最大化投资组合的回报率。

        Args:
            risk_tolerance (float): 风险容忍度（风险的最大值）。

        Returns:
            np.ndarray: 最优投资组合的权重。
        """
        num_assets = len(self.returns)
        args = (self.returns, self.cov_matrix)

        def neg_return(weights: np.ndarray, returns: np.ndarray) -> float:
            return -np.dot(weights, returns)

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'ineq', 'fun': lambda weights: risk_tolerance - np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        result = minimize(neg_return, initial_guess, args=(self.returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def display_portfolio(self, weights: np.ndarray) -> None:
        """
        打印投资组合的详细信息。

        Args:
            weights (np.ndarray): 投资组合中每个资产的权重。
        """
        expected_return, expected_risk = self.portfolio_performance(weights)
        print(f"投资组合权重: {weights}")
        print(f"预期回报率: {expected_return:.2f}")
        print(f"风险 (标准差): {expected_risk:.2f}")

# 示例数据
returns = np.array([0.12, 0.10, 0.15, 0.09])
cov_matrix = np.array([
    [0.005, -0.010, 0.004, -0.002],
    [-0.010, 0.040, -0.002, 0.004],
    [0.004, -0.002, 0.023, 0.002],
    [-0.002, 0.004, 0.002, 0.012]
])

# 创建 PortfolioOptimizer 实例
optimizer = PortfolioOptimizer(returns, cov_matrix)

# 目标回报率下最小化风险
target_return = 0.10
weights_min_risk = optimizer.minimize_risk(target_return)
optimizer.display_portfolio(weights_min_risk)

# 风险容忍度下最大化回报率
risk_tolerance = 0.15
weights_max_return = optimizer.maximize_return(risk_tolerance)
optimizer.display_portfolio(weights_max_return)