"""CQF (Certificate in Quantitative Finance) metrics calculations.

This module implements quantitative finance metrics based on the CQF curriculum:
- Derivatives Pricing: Black-Scholes, Greeks, Binomial Trees
- Stochastic Calculus: Monte Carlo simulation, Path-dependent options
- Fixed Income Derivatives: Interest rate models
- Exotic Options: Asian, Barrier, Lookback
- Portfolio Optimization: Mean-Variance, Black-Litterman
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize


@dataclass
class OptionGreeks:
    """Option Greeks."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class BlackScholesResult:
    """Black-Scholes pricing result."""

    call_price: float
    put_price: float
    greeks: OptionGreeks


@dataclass
class PortfolioOptimizationResult:
    """Portfolio optimization result."""

    weights: NDArray[np.float64]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class CQFMetrics:
    """CQF curriculum-based quantitative finance calculator."""

    # ==================== BLACK-SCHOLES MODEL ====================

    @staticmethod
    def black_scholes_d1(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate d1 for Black-Scholes model.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            d1 value.
        """
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def black_scholes_d2(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate d2 for Black-Scholes model.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            d2 value.
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)

    @staticmethod
    def black_scholes_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate Black-Scholes call option price.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Call option price.
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        d2 = CQFMetrics.black_scholes_d2(S, K, T, r, sigma)
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    @staticmethod
    def black_scholes_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate Black-Scholes put option price.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Put option price.
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        d2 = CQFMetrics.black_scholes_d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    # ==================== GREEKS ====================

    @staticmethod
    def delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate option delta.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: "call" or "put".

        Returns:
            Delta.
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        if option_type == "call":
            return float(stats.norm.cdf(d1))
        else:
            return float(stats.norm.cdf(d1) - 1)

    @staticmethod
    def gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate option gamma.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Gamma.
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        return float(stats.norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate option theta (per day).

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: "call" or "put".

        Returns:
            Theta (per day).
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        d2 = CQFMetrics.black_scholes_d2(S, K, T, r, sigma)

        term1 = -S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
            theta_annual = term1 + term2
        else:
            term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
            theta_annual = term1 + term2

        return float(theta_annual / 365)  # Daily theta

    @staticmethod
    def vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate option vega (per 1% change in volatility).

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Vega (per 1% vol change).
        """
        d1 = CQFMetrics.black_scholes_d1(S, K, T, r, sigma)
        return float(S * stats.norm.pdf(d1) * np.sqrt(T) / 100)

    @staticmethod
    def rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate option rho (per 1% change in rate).

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: "call" or "put".

        Returns:
            Rho (per 1% rate change).
        """
        d2 = CQFMetrics.black_scholes_d2(S, K, T, r, sigma)

        if option_type == "call":
            return float(K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100)
        else:
            return float(-K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100)

    @staticmethod
    def calculate_all_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> OptionGreeks:
        """Calculate all option Greeks.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            option_type: "call" or "put".

        Returns:
            OptionGreeks dataclass.
        """
        return OptionGreeks(
            delta=CQFMetrics.delta(S, K, T, r, sigma, option_type),
            gamma=CQFMetrics.gamma(S, K, T, r, sigma),
            theta=CQFMetrics.theta(S, K, T, r, sigma, option_type),
            vega=CQFMetrics.vega(S, K, T, r, sigma),
            rho=CQFMetrics.rho(S, K, T, r, sigma, option_type),
        )

    # ==================== IMPLIED VOLATILITY ====================

    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """Calculate implied volatility using Newton-Raphson.

        Args:
            option_price: Market price of option.
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            option_type: "call" or "put".
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.

        Returns:
            Implied volatility.
        """
        sigma = 0.2  # Initial guess

        for _ in range(max_iterations):
            if option_type == "call":
                price = CQFMetrics.black_scholes_call(S, K, T, r, sigma)
            else:
                price = CQFMetrics.black_scholes_put(S, K, T, r, sigma)

            vega = CQFMetrics.vega(S, K, T, r, sigma) * 100  # Undo the /100

            diff = option_price - price
            if abs(diff) < tolerance:
                return sigma

            if vega == 0:
                break

            sigma = sigma + diff / vega

        return sigma

    # ==================== BINOMIAL TREE ====================

    @staticmethod
    def binomial_tree_european(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int,
        option_type: str = "call",
    ) -> float:
        """Price European option using binomial tree.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            n_steps: Number of time steps.
            option_type: "call" or "put".

        Returns:
            Option price.
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)

        # Build price tree at maturity
        prices = np.array([S * (u ** (n_steps - i)) * (d**i) for i in range(n_steps + 1)])

        # Calculate option values at maturity
        if option_type == "call":
            values = np.maximum(prices - K, 0)
        else:
            values = np.maximum(K - prices, 0)

        # Backward induction
        for step in range(n_steps - 1, -1, -1):
            values = np.exp(-r * dt) * (p * values[:-1] + (1 - p) * values[1:])

        return float(values[0])

    @staticmethod
    def binomial_tree_american(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int,
        option_type: str = "put",
    ) -> float:
        """Price American option using binomial tree.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            n_steps: Number of time steps.
            option_type: "call" or "put".

        Returns:
            Option price.
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)

        # Build full price tree
        price_tree = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                price_tree[j, i] = S * (u ** (i - j)) * (d**j)

        # Initialize option values at maturity
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        if option_type == "call":
            option_tree[:, n_steps] = np.maximum(price_tree[:, n_steps] - K, 0)
        else:
            option_tree[:, n_steps] = np.maximum(K - price_tree[:, n_steps], 0)

        # Backward induction with early exercise check
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                hold_value = np.exp(-r * dt) * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )
                if option_type == "call":
                    exercise_value = max(price_tree[j, i] - K, 0)
                else:
                    exercise_value = max(K - price_tree[j, i], 0)
                option_tree[j, i] = max(hold_value, exercise_value)

        return float(option_tree[0, 0])

    # ==================== MONTE CARLO SIMULATION ====================

    @staticmethod
    def monte_carlo_european(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_simulations: int = 100000,
        option_type: str = "call",
    ) -> tuple[float, float]:
        """Price European option using Monte Carlo simulation.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            n_simulations: Number of simulations.
            option_type: "call" or "put".

        Returns:
            Tuple of (price, standard error).
        """
        # Generate random paths
        Z = np.random.standard_normal(n_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Discount to present
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

        return float(price), float(std_error)

    @staticmethod
    def monte_carlo_asian(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int = 252,
        n_simulations: int = 100000,
        option_type: str = "call",
        average_type: str = "arithmetic",
    ) -> tuple[float, float]:
        """Price Asian option using Monte Carlo simulation.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            n_steps: Number of averaging periods.
            n_simulations: Number of simulations.
            option_type: "call" or "put".
            average_type: "arithmetic" or "geometric".

        Returns:
            Tuple of (price, standard error).
        """
        dt = T / n_steps

        # Generate paths
        Z = np.random.standard_normal((n_simulations, n_steps))
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S

        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
            )

        # Calculate averages
        if average_type == "arithmetic":
            averages = np.mean(paths[:, 1:], axis=1)
        else:
            averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(averages - K, 0)
        else:
            payoffs = np.maximum(K - averages, 0)

        # Discount to present
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

        return float(price), float(std_error)

    @staticmethod
    def monte_carlo_barrier(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        n_steps: int = 252,
        n_simulations: int = 100000,
        option_type: str = "call",
        barrier_type: str = "down-and-out",
    ) -> tuple[float, float]:
        """Price barrier option using Monte Carlo simulation.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.
            barrier: Barrier level.
            n_steps: Number of time steps.
            n_simulations: Number of simulations.
            option_type: "call" or "put".
            barrier_type: "down-and-out", "down-and-in", "up-and-out", "up-and-in".

        Returns:
            Tuple of (price, standard error).
        """
        dt = T / n_steps

        # Generate paths
        Z = np.random.standard_normal((n_simulations, n_steps))
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S

        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
            )

        ST = paths[:, -1]

        # Determine if barrier was hit
        if "down" in barrier_type:
            barrier_hit = np.min(paths, axis=1) <= barrier
        else:
            barrier_hit = np.max(paths, axis=1) >= barrier

        # Calculate vanilla payoffs
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Apply barrier condition
        if "out" in barrier_type:
            payoffs = payoffs * (~barrier_hit)
        else:  # "in"
            payoffs = payoffs * barrier_hit

        # Discount to present
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

        return float(price), float(std_error)

    # ==================== INTEREST RATE MODELS ====================

    @staticmethod
    def vasicek_rate(
        r0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_simulations: int = 10000,
    ) -> NDArray[np.float64]:
        """Simulate interest rates using Vasicek model.

        Args:
            r0: Initial short rate.
            kappa: Mean reversion speed.
            theta: Long-term mean rate.
            sigma: Volatility.
            T: Time horizon.
            n_steps: Number of time steps.
            n_simulations: Number of simulations.

        Returns:
            Array of simulated rate paths.
        """
        dt = T / n_steps
        rates = np.zeros((n_simulations, n_steps + 1))
        rates[:, 0] = r0

        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_simulations)
            rates[:, t] = (
                rates[:, t - 1]
                + kappa * (theta - rates[:, t - 1]) * dt
                + sigma * np.sqrt(dt) * Z
            )

        return rates

    @staticmethod
    def cir_rate(
        r0: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_simulations: int = 10000,
    ) -> NDArray[np.float64]:
        """Simulate interest rates using CIR model.

        Args:
            r0: Initial short rate.
            kappa: Mean reversion speed.
            theta: Long-term mean rate.
            sigma: Volatility.
            T: Time horizon.
            n_steps: Number of time steps.
            n_simulations: Number of simulations.

        Returns:
            Array of simulated rate paths.
        """
        dt = T / n_steps
        rates = np.zeros((n_simulations, n_steps + 1))
        rates[:, 0] = r0

        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_simulations)
            rates[:, t] = np.maximum(
                rates[:, t - 1]
                + kappa * (theta - rates[:, t - 1]) * dt
                + sigma * np.sqrt(np.maximum(rates[:, t - 1], 0) * dt) * Z,
                0,
            )

        return rates

    @staticmethod
    def zero_coupon_bond_vasicek(
        r: float,
        kappa: float,
        theta: float,
        sigma: float,
        T: float,
    ) -> float:
        """Price zero-coupon bond under Vasicek model.

        Args:
            r: Current short rate.
            kappa: Mean reversion speed.
            theta: Long-term mean rate.
            sigma: Volatility.
            T: Time to maturity.

        Returns:
            Bond price.
        """
        B = (1 - np.exp(-kappa * T)) / kappa
        A = np.exp(
            (theta - sigma**2 / (2 * kappa**2)) * (B - T)
            - sigma**2 * B**2 / (4 * kappa)
        )
        return float(A * np.exp(-B * r))

    # ==================== PORTFOLIO OPTIMIZATION ====================

    @staticmethod
    def mean_variance_optimization(
        expected_returns: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        target_return: float | None = None,
    ) -> PortfolioOptimizationResult:
        """Perform mean-variance optimization.

        Args:
            expected_returns: Expected returns for each asset.
            cov_matrix: Covariance matrix.
            risk_free_rate: Risk-free rate.
            target_return: Target portfolio return (for efficient frontier).

        Returns:
            PortfolioOptimizationResult dataclass.
        """
        n_assets = len(expected_returns)

        def portfolio_volatility(weights: NDArray[np.float64]) -> float:
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def neg_sharpe_ratio(weights: NDArray[np.float64]) -> float:
            ret = np.dot(weights, expected_returns)
            vol = portfolio_volatility(weights)
            return -(ret - risk_free_rate) / vol if vol > 0 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if target_return is not None:
            constraints.append(
                {"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return}
            )

        bounds = [(0, 1) for _ in range(n_assets)]
        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        optimal_return = np.dot(optimal_weights, expected_returns)
        optimal_vol = portfolio_volatility(optimal_weights)
        optimal_sharpe = (optimal_return - risk_free_rate) / optimal_vol if optimal_vol > 0 else 0

        return PortfolioOptimizationResult(
            weights=optimal_weights,
            expected_return=float(optimal_return),
            volatility=float(optimal_vol),
            sharpe_ratio=float(optimal_sharpe),
        )

    @staticmethod
    def efficient_frontier(
        expected_returns: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        n_points: int = 100,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Calculate efficient frontier.

        Args:
            expected_returns: Expected returns for each asset.
            cov_matrix: Covariance matrix.
            n_points: Number of points on the frontier.

        Returns:
            Tuple of (returns, volatilities, weights).
        """
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_returns = []
        frontier_vols = []
        frontier_weights = []

        for target_ret in target_returns:
            try:
                result = CQFMetrics.mean_variance_optimization(
                    expected_returns, cov_matrix, target_return=target_ret
                )
                frontier_returns.append(result.expected_return)
                frontier_vols.append(result.volatility)
                frontier_weights.append(result.weights)
            except Exception:
                continue

        return (
            np.array(frontier_returns),
            np.array(frontier_vols),
            np.array(frontier_weights),
        )

    @staticmethod
    def black_litterman(
        market_weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        risk_aversion: float,
        P: NDArray[np.float64],
        Q: NDArray[np.float64],
        omega: NDArray[np.float64],
        tau: float = 0.05,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Black-Litterman expected returns.

        Args:
            market_weights: Market capitalization weights.
            cov_matrix: Covariance matrix.
            risk_aversion: Risk aversion coefficient.
            P: Views matrix (K x N).
            Q: View returns vector (K x 1).
            omega: Uncertainty matrix for views (K x K).
            tau: Scaling factor for prior uncertainty.

        Returns:
            Tuple of (posterior expected returns, posterior covariance).
        """
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        # Posterior expected returns
        tau_cov = tau * cov_matrix
        M1 = np.linalg.inv(tau_cov)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(M1, pi) + np.dot(P.T, np.dot(np.linalg.inv(omega), Q))

        posterior_returns = np.dot(np.linalg.inv(M1 + M2), M3)

        # Posterior covariance
        posterior_cov = np.linalg.inv(M1 + M2)

        return posterior_returns, posterior_cov + cov_matrix

    @staticmethod
    def calculate_black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> BlackScholesResult:
        """Calculate comprehensive Black-Scholes result.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            BlackScholesResult dataclass.
        """
        call_price = CQFMetrics.black_scholes_call(S, K, T, r, sigma)
        put_price = CQFMetrics.black_scholes_put(S, K, T, r, sigma)
        greeks = CQFMetrics.calculate_all_greeks(S, K, T, r, sigma, "call")

        return BlackScholesResult(
            call_price=call_price,
            put_price=put_price,
            greeks=greeks,
        )
