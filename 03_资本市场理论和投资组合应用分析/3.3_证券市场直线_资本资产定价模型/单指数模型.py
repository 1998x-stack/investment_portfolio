import numpy as np
from scipy import stats
from typing import List, Tuple

class SingleIndexModel:
    """单指数模型类，用于估计资产的系统风险和预期回报率。

    该类通过回归分析估计资产的贝塔系数和预期回报率，并与市场回报率进行比较。

    Attributes:
        asset_returns (np.ndarray): 资产回报率的时间序列数据。
        market_returns (np.ndarray): 市场回报率的时间序列数据。
        alpha (float): 回归截距。
        beta (float): 贝塔系数，衡量资产相对于市场的系统风险。
        residuals (np.ndarray): 回归残差。
    """

    def __init__(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> None:
        """
        初始化SingleIndexModel实例。

        Args:
            asset_returns (np.ndarray): 资产回报率的时间序列数据。
            market_returns (np.ndarray): 市场回报率的时间序列数据。
        """
        self.asset_returns = asset_returns
        self.market_returns = market_returns
        self.alpha = 0.0
        self.beta = 0.0
        self.residuals = np.array([])
        self._perform_regression()

    def _perform_regression(self) -> None:
        """执行回归分析，估计模型参数alpha和beta，并计算残差。"""
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.market_returns, self.asset_returns)
        self.alpha = intercept
        self.beta = slope
        self.residuals = self.asset_returns - (self.alpha + self.beta * self.market_returns)
        print(f"回归结果: alpha={self.alpha:.4f}, beta={self.beta:.4f}")

    def expected_return(self, risk_free_rate: float, market_return: float) -> float:
        """
        计算资产的预期回报率。

        Args:
            risk_free_rate (float): 无风险利率。
            market_return (float): 市场预期回报率。

        Returns:
            float: 资产的预期回报率。
        """
        expected_return = risk_free_rate + self.beta * (market_return - risk_free_rate)
        print(f"预期回报率: {expected_return:.4f}")
        return expected_return

    def summary(self) -> None:
        """打印模型的详细信息。"""
        print("单指数模型分析结果：")
        print(f"alpha（回归截距）: {self.alpha:.4f}")
        print(f"beta（贝塔系数）: {self.beta:.4f}")
        print(f"残差标准差: {np.std(self.residuals):.4f}")

# 示例数据
asset_returns = np.array([0.12, 0.10, 0.15, 0.09, 0.11])
market_returns = np.array([0.10, 0.08, 0.12, 0.07, 0.09])

# 创建SingleIndexModel实例
sim = SingleIndexModel(asset_returns, market_returns)

# 打印模型的详细信息
sim.summary()

# 计算资产的预期回报率
risk_free_rate = 0.03
market_return = 0.10
sim.expected_return(risk_free_rate, market_return)
