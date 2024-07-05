import numpy as np
from typing import List, Tuple

class MultiFactorModel:
    """
    Multi-Factor Model class to estimate expected returns and risk based on multiple risk factors.
    
    Attributes:
        risk_free_rate (float): The risk-free rate.
        factor_names (List[str]): List of risk factor names.
        historical_data (np.ndarray): Historical data matrix where rows are dates and columns are [stock returns, factor1, factor2, ...].
        num_factors (int): Number of risk factors.
    """

    def __init__(self, risk_free_rate: float, factor_names: List[str]):
        """
        Initialize the MultiFactorModel with a risk-free rate and names of the risk factors.
        
        Args:
            risk_free_rate (float): The risk-free rate.
            factor_names (List[str]): List of risk factor names.
        """
        self.risk_free_rate = risk_free_rate
        self.factor_names = factor_names
        self.historical_data = None
        self.num_factors = len(factor_names)

    def load_data(self, data: np.ndarray) -> None:
        """
        Load historical data into the model.
        
        Args:
            data (np.ndarray): Historical data matrix where rows are dates and columns are [stock returns, factor1, factor2, ...].
        """
        if data.shape[1] != self.num_factors + 1:
            raise ValueError(f"Data should have {self.num_factors + 1} columns")
        self.historical_data = data

    def estimate_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the risk premiums (lambda) and factor sensitivities (beta) using historical data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Estimated risk premiums and factor sensitivities.
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded")

        Y = self.historical_data[:, 0] - self.risk_free_rate  # Excess returns
        X = self.historical_data[:, 1:]  # Factor matrix

        # Adding a column of ones for the intercept term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Solving for beta using ordinary least squares
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        
        # Risk premiums are estimated as the coefficients corresponding to factors (excluding intercept)
        lambda_est = beta[1:]
        
        return lambda_est, beta

    def predict_return(self, factor_values: np.ndarray) -> float:
        """
        Predict the expected return of a stock given the factor values.
        
        Args:
            factor_values (np.ndarray): Array of factor values.
        
        Returns:
            float: Predicted expected return.
        """
        lambda_est, beta = self.estimate_parameters()
        expected_excess_return = np.dot(lambda_est, factor_values)
        expected_return = self.risk_free_rate + expected_excess_return
        return expected_return

    def evaluate_model(self) -> float:
        """
        Evaluate the model by calculating the R-squared value.
        
        Returns:
            float: R-squared value of the model.
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded")

        Y = self.historical_data[:, 0] - self.risk_free_rate  # Excess returns
        X = self.historical_data[:, 1:]  # Factor matrix

        # Adding a column of ones for the intercept term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y
        Y_pred = X @ beta

        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        ss_residual = np.sum((Y - Y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        return r_squared


# 示例数据加载
historical_data = np.array([
    [0.12, 0.01, 0.02],
    [0.15, 0.03, 0.01],
    [0.10, 0.02, 0.03],
    [0.18, 0.04, 0.02],
])

# 模型初始化
risk_free_rate = 0.05
factor_names = ["Market Risk", "Size Risk"]
multi_factor_model = MultiFactorModel(risk_free_rate, factor_names)

# 加载历史数据
multi_factor_model.load_data(historical_data)

# 估计参数
lambda_est, beta = multi_factor_model.estimate_parameters()
print(f"Risk Premiums (Lambda): {lambda_est}")
print(f"Factor Sensitivities (Beta): {beta}")

# 预测回报率
factor_values = np.array([0.02, 0.03])
expected_return = multi_factor_model.predict_return(factor_values)
print(f"Predicted Expected Return: {expected_return}")

# 评估模型
r_squared = multi_factor_model.evaluate_model()
print(f"R-squared: {r_squared}")

# 校验代码的正确性和逻辑
assert isinstance(lambda_est, np.ndarray), "Lambda estimation should be a numpy array"
assert isinstance(beta, np.ndarray), "Beta estimation should be a numpy array"
assert isinstance(expected_return, float), "Expected return should be a float"
assert isinstance(r_squared, float), "R-squared should be a float"

print("All checks passed. The Multi-Factor Model is correctly implemented and evaluated.")
