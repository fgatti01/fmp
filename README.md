# FMP Financial Analytics Platform

A comprehensive financial analytics platform with FMP API integration, CFA/FRM/CQF metrics calculations, and AI-powered analysis using Agno agents.

## Features

### Data Integration
- **FMP API Client**: Full integration with Financial Modeling Prep API
  - Stock quotes and company profiles
  - Historical prices and technical indicators
  - Financial statements (income, balance sheet, cash flow)
  - Financial ratios and key metrics
  - Calendars (earnings, dividends, economic events)
  - Market movers and sector performance

### Financial Metrics

#### CFA (Chartered Financial Analyst) Metrics
- **Portfolio Management**: Sharpe ratio, Sortino ratio, Treynor ratio, Jensen's Alpha
- **Fixed Income**: Duration (Macaulay, Modified), Convexity, YTM, DV01
- **Equity Valuation**: DCF, Gordon Growth Model, Residual Income
- **Corporate Finance**: WACC, CAPM, EVA, Free Cash Flow

#### FRM (Financial Risk Manager) Metrics
- **Market Risk**: VaR (Historical, Parametric, Monte Carlo), CVaR/ES
- **Credit Risk**: PD, LGD, EAD, Expected Loss, Unexpected Loss
- **Portfolio Risk**: Component VaR, Marginal VaR, Risk Budgeting
- **Stress Testing**: Scenario analysis, Historical scenarios

#### CQF (Certificate in Quantitative Finance) Metrics
- **Derivatives Pricing**: Black-Scholes, Binomial Trees
- **Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Monte Carlo**: European, Asian, Barrier options
- **Interest Rate Models**: Vasicek, CIR
- **Portfolio Optimization**: Mean-Variance, Efficient Frontier, Black-Litterman

### AI-Powered Analysis
- **Agno Agent**: Intelligent financial analyst with reasoning capabilities
- **Natural Language Interface**: Ask questions about stocks, portfolios, and markets
- **Automated Analysis**: Stock screening, portfolio optimization, risk assessment

### Portfolio Management
- **Portfolio Tracker**: Real-time portfolio monitoring
- **Transaction Management**: Buy, sell, dividend tracking
- **Performance Attribution**: Return and risk decomposition
- **Multiple Portfolios**: Manage and compare portfolios

## Installation

```bash
# Clone the repository
git clone https://github.com/fgatti01/fmp.git
cd fmp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,viz]"
```

## Configuration

Create a `.env` file in the project root:

```env
# FMP API Configuration
FMP_API_KEY=your_fmp_api_key_here
FMP_BASE_URL=https://financialmodelingprep.com/api

# OpenAI Configuration (for Agno agent)
OPENAI_API_KEY=your_openai_api_key_here

# Logging
LOG_LEVEL=INFO
```

## Usage

### Command Line Interface

```bash
# Get stock quote
fmp quote AAPL,MSFT,GOOGL

# Get company profile
fmp profile AAPL

# Analyze a stock
fmp analyze AAPL

# Optimize portfolio
fmp optimize AAPL,MSFT,GOOGL,AMZN

# DCF valuation
fmp dcf AAPL --growth 0.08 --terminal 0.02 --discount 0.10

# Market movers
fmp movers

# Sector performance
fmp sectors

# Interactive AI chat
fmp chat
```

### Python API

```python
import asyncio
from fmp_analytics.pipeline import DataPipeline, AnalysisPipeline
from fmp_analytics.metrics import CFAMetrics, FRMMetrics, CQFMetrics

async def main():
    async with DataPipeline() as data:
        # Get stock data
        quote = await data.get_quote("AAPL")
        print(f"AAPL Price: ${quote.price}")

        # Get historical returns
        returns = await data.get_returns("AAPL")

        # Calculate metrics
        sharpe = CFAMetrics.sharpe_ratio(returns.values, risk_free_rate=0.05)
        print(f"Sharpe Ratio: {sharpe:.2f}")

        # Analyze stock
        analysis = AnalysisPipeline(data)
        result = await analysis.analyze_stock("AAPL")
        print(f"Overall Score: {result.overall_score}/100")

asyncio.run(main())
```

### Agno Agent

```python
from fmp_analytics.agent import FinancialAgent

# Create agent
agent = FinancialAgent()

# Ask questions
response = agent.chat("Analyze Apple stock and tell me if it's a good investment")
print(response)

# Get portfolio optimization
response = agent.chat("Find optimal weights for AAPL, MSFT, GOOGL, AMZN")
print(response)
```

### Portfolio Tracker

```python
from fmp_analytics.portfolio import PortfolioTracker

# Create portfolio
tracker = PortfolioTracker()

# Add transactions
tracker.buy("AAPL", quantity=10, price=150.00)
tracker.buy("MSFT", quantity=5, price=300.00)
tracker.sell("AAPL", quantity=5, price=160.00)

# Update market values
await tracker.update_market_values()

# Get metrics
metrics = await tracker.calculate_metrics()
print(f"Portfolio Value: ${metrics.total_value:,.2f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

## Project Structure

```
fmp/
├── src/fmp_analytics/
│   ├── api/                  # FMP API client
│   │   ├── client.py         # Core HTTP client
│   │   └── endpoints/        # API endpoint modules
│   ├── models/               # Pydantic data models
│   ├── metrics/              # Financial metrics
│   │   ├── cfa.py           # CFA metrics
│   │   ├── frm.py           # FRM metrics
│   │   └── cqf.py           # CQF metrics
│   ├── pipeline/             # Data and analysis pipelines
│   │   ├── data_pipeline.py  # Data ingestion
│   │   └── analysis_pipeline.py  # Analysis workflows
│   ├── agent/                # Agno AI agent
│   │   ├── tools.py         # Agent tools
│   │   └── financial_agent.py  # Agent implementation
│   ├── portfolio/            # Portfolio management
│   │   ├── tracker.py       # Portfolio tracker
│   │   └── manager.py       # Multi-portfolio manager
│   └── cli.py               # Command-line interface
├── tests/                    # Test suite
├── pyproject.toml           # Project configuration
└── README.md                # Documentation
```

## API Coverage

The platform integrates with the following FMP API endpoints:

| Category | Endpoints |
|----------|-----------|
| Stock Lists | stock/list, etf/list, tradable/list, stock-screener |
| Company Info | profile, key-executives, company-outlook, stock_peers |
| Market Data | quote, quotes, gainers, losers, actives, sector-performance |
| Historical | historical-price-full, historical-chart, technical_indicator |
| Financials | income-statement, balance-sheet, cash-flow, ratios, key-metrics |
| Valuation | dcf, enterprise-values, financial-score |
| Calendars | earnings, dividends, ipo, economic, stock-splits |
| Forex/Crypto | fx, quotes/crypto, historical forex/crypto |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/fmp_analytics --cov-report=html

# Run specific test file
pytest tests/test_metrics.py
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## Acknowledgments

- [Financial Modeling Prep](https://financialmodelingprep.com/) for the financial data API
- [Agno](https://github.com/agno-agi/agno) for the AI agent framework
- CFA Institute, GARP (FRM), and CQF Institute for the financial metrics frameworks
