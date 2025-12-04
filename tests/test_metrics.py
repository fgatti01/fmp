"""Tests for CFA, FRM, and CQF metrics calculations."""

import numpy as np
import pytest

from fmp_analytics.metrics.cfa import CFAMetrics
from fmp_analytics.metrics.frm import FRMMetrics
from fmp_analytics.metrics.cqf import CQFMetrics


class TestCFAMetrics:
    """Tests for CFA metrics."""

    def test_present_value(self):
        """Test present value calculation."""
        # $1000 in 5 years at 5% = $783.53
        pv = CFAMetrics.present_value(1000, 0.05, 5)
        assert abs(pv - 783.53) < 0.01

    def test_future_value(self):
        """Test future value calculation."""
        # $1000 today for 5 years at 5% = $1276.28
        fv = CFAMetrics.future_value(1000, 0.05, 5)
        assert abs(fv - 1276.28) < 0.01

    def test_npv(self):
        """Test NPV calculation."""
        cash_flows = [-1000, 300, 400, 500, 200]
        npv = CFAMetrics.npv(cash_flows, 0.10)
        assert npv > 0  # Positive NPV

    def test_irr(self):
        """Test IRR calculation."""
        cash_flows = [-1000, 300, 400, 500, 200]
        irr = CFAMetrics.irr(cash_flows)
        assert 0.10 < irr < 0.20  # Reasonable IRR

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        sharpe = CFAMetrics.sharpe_ratio(returns, 0.02)
        assert isinstance(sharpe, float)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        sortino = CFAMetrics.sortino_ratio(returns, 0.02)
        assert isinstance(sortino, float)

    def test_beta(self):
        """Test beta calculation."""
        market_returns = np.random.normal(0.0005, 0.01, 252)
        stock_returns = 1.2 * market_returns + np.random.normal(0, 0.005, 252)
        beta = CFAMetrics.beta(stock_returns, market_returns)
        assert 1.0 < beta < 1.4  # Should be close to 1.2

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        returns = np.array([0.01, 0.02, -0.05, -0.03, 0.02, 0.01])
        mdd = CFAMetrics.max_drawdown(returns)
        assert mdd > 0

    def test_bond_price(self):
        """Test bond pricing."""
        # $1000 par, 5% coupon, 5% YTM = par value
        price = CFAMetrics.bond_price(1000, 0.05, 0.05, 10, 2)
        assert abs(price - 1000) < 1

    def test_modified_duration(self):
        """Test modified duration."""
        mod_dur = CFAMetrics.modified_duration(1000, 0.05, 0.05, 10, 2)
        assert 0 < mod_dur < 10

    def test_portfolio_return(self):
        """Test portfolio return calculation."""
        weights = np.array([0.4, 0.3, 0.3])
        returns = np.array([0.10, 0.08, 0.12])
        port_return = CFAMetrics.portfolio_return(weights, returns)
        assert abs(port_return - 0.10) < 0.001

    def test_portfolio_variance(self):
        """Test portfolio variance calculation."""
        weights = np.array([0.5, 0.5])
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
        variance = CFAMetrics.portfolio_variance(weights, cov_matrix)
        assert variance > 0


class TestFRMMetrics:
    """Tests for FRM metrics."""

    def test_historical_var(self):
        """Test historical VaR calculation."""
        returns = np.random.normal(0, 0.02, 1000)
        var = FRMMetrics.historical_var(returns, 0.95)
        assert var > 0

    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        returns = np.random.normal(0, 0.02, 1000)
        var = FRMMetrics.parametric_var(returns, 0.95)
        assert var > 0

    def test_conditional_var(self):
        """Test CVaR calculation."""
        returns = np.random.normal(0, 0.02, 1000)
        cvar = FRMMetrics.conditional_var(returns, 0.95)
        var = FRMMetrics.historical_var(returns, 0.95)
        assert cvar >= var  # CVaR >= VaR

    def test_expected_loss(self):
        """Test expected loss calculation."""
        el = FRMMetrics.expected_loss(0.02, 0.45, 1000000)
        assert el == 0.02 * 0.45 * 1000000

    def test_unexpected_loss(self):
        """Test unexpected loss calculation."""
        ul = FRMMetrics.unexpected_loss(0.02, 0.45, 1000000)
        assert ul > 0

    def test_credit_var(self):
        """Test credit VaR calculation."""
        cvar = FRMMetrics.credit_var(0.02, 0.45, 1000000)
        assert cvar > 0

    def test_liquidity_coverage_ratio(self):
        """Test LCR calculation."""
        lcr = FRMMetrics.liquidity_coverage_ratio(1000000, 800000)
        assert abs(lcr - 1.25) < 0.01

    def test_bid_ask_spread(self):
        """Test bid-ask spread calculation."""
        spread = FRMMetrics.bid_ask_spread(99.50, 100.50)
        assert abs(spread - 0.01) < 0.001  # 1% spread


class TestCQFMetrics:
    """Tests for CQF metrics."""

    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing."""
        # At-the-money option
        call = CQFMetrics.black_scholes_call(100, 100, 1, 0.05, 0.2)
        assert call > 0
        assert call < 100  # Less than stock price

    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing."""
        put = CQFMetrics.black_scholes_put(100, 100, 1, 0.05, 0.2)
        assert put > 0
        assert put < 100

    def test_put_call_parity(self):
        """Test put-call parity."""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        call = CQFMetrics.black_scholes_call(S, K, T, r, sigma)
        put = CQFMetrics.black_scholes_put(S, K, T, r, sigma)
        # C - P = S - K*e^(-rT)
        parity_diff = call - put - (S - K * np.exp(-r * T))
        assert abs(parity_diff) < 0.001

    def test_delta_call(self):
        """Test call delta."""
        delta = CQFMetrics.delta(100, 100, 1, 0.05, 0.2, "call")
        assert 0 < delta < 1

    def test_delta_put(self):
        """Test put delta."""
        delta = CQFMetrics.delta(100, 100, 1, 0.05, 0.2, "put")
        assert -1 < delta < 0

    def test_gamma(self):
        """Test gamma."""
        gamma = CQFMetrics.gamma(100, 100, 1, 0.05, 0.2)
        assert gamma > 0

    def test_vega(self):
        """Test vega."""
        vega = CQFMetrics.vega(100, 100, 1, 0.05, 0.2)
        assert vega > 0

    def test_theta_call(self):
        """Test call theta."""
        theta = CQFMetrics.theta(100, 100, 1, 0.05, 0.2, "call")
        assert theta < 0  # Time decay

    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        # Calculate option price with known vol
        true_vol = 0.25
        price = CQFMetrics.black_scholes_call(100, 100, 1, 0.05, true_vol)

        # Back out implied vol
        iv = CQFMetrics.implied_volatility(price, 100, 100, 1, 0.05, "call")
        assert abs(iv - true_vol) < 0.001

    def test_binomial_european_convergence(self):
        """Test binomial tree converges to Black-Scholes."""
        bs_price = CQFMetrics.black_scholes_call(100, 100, 1, 0.05, 0.2)
        bin_price = CQFMetrics.binomial_tree_european(100, 100, 1, 0.05, 0.2, 100, "call")
        assert abs(bs_price - bin_price) < 0.5  # Within 50 cents

    def test_monte_carlo_european(self):
        """Test Monte Carlo option pricing."""
        price, std_err = CQFMetrics.monte_carlo_european(100, 100, 1, 0.05, 0.2)
        bs_price = CQFMetrics.black_scholes_call(100, 100, 1, 0.05, 0.2)
        assert abs(price - bs_price) < 0.5  # Close to BS

    def test_mean_variance_optimization(self):
        """Test portfolio optimization."""
        expected_returns = np.array([0.10, 0.08, 0.12])
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
        result = CQFMetrics.mean_variance_optimization(expected_returns, cov_matrix)

        assert np.isclose(np.sum(result.weights), 1.0)  # Weights sum to 1
        assert all(w >= 0 for w in result.weights)  # No short selling
        assert result.sharpe_ratio > 0


class TestIntegration:
    """Integration tests across modules."""

    def test_portfolio_risk_metrics(self):
        """Test calculating portfolio risk metrics."""
        # Generate correlated returns
        cov = np.array([[0.04, 0.01], [0.01, 0.03]])
        mean = [0.001, 0.0008]
        returns = np.random.multivariate_normal(mean, cov / 252, 252)

        weights = np.array([0.6, 0.4])
        portfolio_returns = np.dot(returns, weights)

        # CFA metrics
        sharpe = CFAMetrics.sharpe_ratio(portfolio_returns, 0.02)
        vol = CFAMetrics.annualized_volatility(portfolio_returns)

        # FRM metrics
        var = FRMMetrics.historical_var(portfolio_returns, 0.95)
        cvar = FRMMetrics.conditional_var(portfolio_returns, 0.95)

        assert isinstance(sharpe, float)
        assert vol > 0
        assert var > 0
        assert cvar >= var

    def test_option_greeks_consistency(self):
        """Test that Greeks are consistent with finite differences."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.2
        eps = 0.01

        # Delta via finite difference
        call_up = CQFMetrics.black_scholes_call(S + eps, K, T, r, sigma)
        call_down = CQFMetrics.black_scholes_call(S - eps, K, T, r, sigma)
        delta_fd = (call_up - call_down) / (2 * eps)

        # Analytical delta
        delta = CQFMetrics.delta(S, K, T, r, sigma, "call")

        assert abs(delta - delta_fd) < 0.01
